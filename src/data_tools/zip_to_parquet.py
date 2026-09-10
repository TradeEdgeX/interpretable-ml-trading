"""
数据转换工具：将 ZIP 格式的交易数据转换为 Parquet 格式

这个模块提供了 DataConverter 类，用于批量将 Binance aggTrades ZIP 文件
转换为 Parquet 格式，提高后续数据处理的效率。
"""

import os
import pandas as pd
import numpy as np
import zipfile
import glob
from datetime import datetime
import shutil
import logging
import re
from typing import Optional, Dict, List
from pathlib import Path

# 设置日志
logger = logging.getLogger(__name__)


def is_binance_um_monthly_aggtrade_zip(filename: str) -> bool:
    """True for UM Vision *monthly* aggTrades ``*-aggTrades-YYYY-MM.zip``.

    Daily archives ``*-aggTrades-YYYY-MM-DD.zip`` return False.
    """

    u = os.path.basename(filename).upper()
    if not u.endswith(".ZIP"):
        return False
    return bool(re.search(r"-AGGTRADES-\d{4}-\d{2}\.ZIP$", u)) and not bool(
        re.search(r"-AGGTRADES-\d{4}-\d{2}-\d{2}\.ZIP$", u)
    )


class DataConverter:
    """数据转换器：将 Binance aggTrades ZIP → 原始 tick Parquet

    简化设计：
    - agg_data/ 存放原始 ZIP（永久保留，不删除）
    - parquet_data/ 存放转换后的 Parquet
    - 转换时检查 parquet 是否存在，不存在才转换
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        *,
        backup_dir: Optional[str] = None,
        force: bool = False,
        aggregate_freq: str = "1s",
    ):
        """
        初始化数据转换器

        Args:
            input_dir: ZIP 文件输入目录（转换后保留源文件）
            output_dir: Parquet 文件输出目录
            force: 是否强制重新转换
            aggregate_freq: 聚合频率，pandas resample 格式（默认: "1s"）
                Examples: "1s" (1秒), "1T" (1分钟), "5T" (5分钟)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.backup_dir = backup_dir
        self.force = bool(force)
        self.aggregate_freq = aggregate_freq
        os.makedirs(self.output_dir, exist_ok=True)
        if self.backup_dir is not None:
            os.makedirs(self.backup_dir, exist_ok=True)

    def convert_zip_to_parquet(
        self, zip_file: str, symbol: Optional[str] = None
    ) -> Optional[Dict]:
        """
        将单个 ZIP 文件转换为 Parquet 格式

        Args:
            zip_file: ZIP 文件路径
            symbol: 交易对符号（如果为 None，从文件名自动检测）

        Returns:
            转换结果字典，包含原始文件、输出文件、日期范围等信息，失败返回 None
        """
        try:
            logger.info(f"Converting {os.path.basename(zip_file)}...")

            # 从文件名自动检测交易对
            zip_basename = os.path.basename(zip_file)
            if symbol is None:
                upper_name = zip_basename.upper()
                match = re.search(r"([A-Z]+)(USDT|USD)", upper_name)
                if match:
                    base = match.group(1)
                    quote = match.group(2)
                    if quote != "USDT":
                        quote = "USDT"
                    symbol = f"{base}{quote}"
                else:
                    symbol = "UNKNOWN"
                    logger.warning(
                        "Could not detect symbol from filename %s, using %s",
                        zip_basename,
                        symbol,
                    )

            normalized_symbol = symbol.upper()
            # Fast-path skip: derive output path from filename and skip BEFORE loading ZIP.
            # This makes reruns / incremental conversion cheap.
            output_file = self._generate_output_filename(zip_file, normalized_symbol)
            if (
                (not self.force)
                and os.path.exists(output_file)
                and os.path.getsize(output_file) > 0
            ):
                logger.info(f"Skip (already converted): {output_file}")
                return {
                    "original_file": zip_file,
                    "output_file": output_file,
                    "skipped": True,
                    "symbol": symbol,
                }

            # 解压zip文件
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                file_list = zip_ref.namelist()
                if not file_list:
                    logger.warning(f"No files in zip: {zip_file}")
                    return None

                # 读取CSV文件
                csv_file = file_list[0]

                with zip_ref.open(csv_file) as f:
                    # 尝试读取第一行检查是否有header
                    first_line = f.readline().decode("utf-8", errors="ignore")

                read_params = {"low_memory": False}

                # 如果第一行是数字，说明没有header
                if first_line.strip().split(",")[0].replace(".", "").isdigit():
                    logger.info("No header detected, using default column names")
                    read_params.update(
                        {
                            "header": None,
                            "names": [
                                "agg_trade_id",
                                "price",
                                "quantity",
                                "first_trade_id",
                                "last_trade_id",
                                "transact_time",
                                "is_buyer_maker",
                            ],
                        }
                    )

                try:
                    if is_binance_um_monthly_aggtrade_zip(zip_basename):
                        logger.info(
                            "Reading aggTrades CSV from zip into memory (%s); "
                            "MONTHLY Vision archives are very large — expect high RAM/swap "
                            "and long runtime (small containers may OOM; prefer daily ZIPs).",
                            zip_basename,
                        )
                    else:
                        logger.info(
                            "Reading aggTrades CSV from zip into memory (%s); "
                            "next log line is row count.",
                            zip_basename,
                        )
                    with zip_ref.open(csv_file) as csv_handle:
                        df = pd.read_csv(csv_handle, **read_params)
                except Exception as read_error:
                    logger.warning(
                        "Primary CSV load failed for %s, retrying with python engine: %s",
                        os.path.basename(zip_file),
                        read_error,
                    )
                    fallback_params = {**read_params, "engine": "python"}
                    with zip_ref.open(csv_file) as csv_handle:
                        df = pd.read_csv(csv_handle, **fallback_params)

                logger.info(f"Loaded data: {df.shape}")

                df_ticks = self._preprocess_tick_data(df)
                if df_ticks is None or df_ticks.empty:
                    logger.warning("No tick data after preprocessing for %s", zip_file)
                    return None
                # Ensure stable output schema: timestamp, price, volume, side (+ symbol)
                expected_cols = ["timestamp", "price", "volume", "side"]
                missing_cols = [c for c in expected_cols if c not in df_ticks.columns]
                if missing_cols:
                    logger.warning(
                        "Missing expected columns %s after preprocessing for %s",
                        missing_cols,
                        zip_file,
                    )
                    return None
                df_ticks = df_ticks[expected_cols].copy()
                df_ticks["symbol"] = normalized_symbol

                df_ticks.to_parquet(output_file, compression="snappy", index=False)
                logger.info(f"Saved tick parquet: {output_file}")

                return {
                    "original_file": zip_file,
                    "output_file": output_file,
                    "start_date": df_ticks["timestamp"].min().strftime("%Y-%m-%d"),
                    "end_date": df_ticks["timestamp"].max().strftime("%Y-%m-%d"),
                    "shape": df_ticks.shape,
                    "symbol": symbol,
                    "skipped": False,
                }

        except Exception as e:
            logger.error(f"Error converting {zip_file}: {e}")
            return None

    def _preprocess_tick_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        预处理 tick 数据

        Args:
            df: 原始 DataFrame

        Returns:
            处理后的 DataFrame，失败返回 None
        """
        try:
            required_cols = {"transact_time", "price", "quantity"}
            if not all(col in df.columns for col in required_cols):
                logger.warning("Missing required columns in %s", df.columns.tolist())
                return None

            df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df["volume"] = pd.to_numeric(df["quantity"], errors="coerce")
            df = df.dropna(subset=["timestamp", "price", "volume"])
            df = df.sort_values("timestamp")

            if "is_buyer_maker" in df.columns:
                df["side"] = np.where(df["is_buyer_maker"].astype(bool), -1, 1)
            elif "side" in df.columns:
                df["side"] = (
                    df["side"].map({"buy": 1, "sell": -1}).fillna(0).astype(int)
                )
            else:
                df["side"] = np.sign(df["volume"]).replace(0, 1)

            # 追加：将原始 tick 聚合到指定频率级别（显著降低体积，适用于 1h/4h 策略）
            # 聚合逻辑：
            # - 按指定频率内按买/卖分别累加成交量
            # - 价格使用该时间段的整体 VWAP（price * volume / sum(volume)）
            # - 若某时间段只有买或只有卖，则只输出对应一条记录
            df["buy_volume"] = np.where(df["side"] == 1, df["volume"], 0.0)
            df["sell_volume"] = np.where(df["side"] == -1, df["volume"], 0.0)
            df["price_volume"] = df["price"] * df["volume"]

            agg = (
                df.set_index("timestamp")
                # 使用 self.aggregate_freq 指定的频率（默认 "1s"，也可以是 "1T" 等）
                .resample(self.aggregate_freq).agg(
                    {
                        "buy_volume": "sum",
                        "sell_volume": "sum",
                        "price_volume": "sum",
                        "volume": "sum",
                    }
                )
            )

            # 计算该时间段 VWAP
            agg["vwap"] = agg["price_volume"] / agg["volume"].replace(0, np.nan)

            rows = []
            for ts, row in agg.iterrows():
                vwap = row["vwap"]
                buy_vol = row["buy_volume"]
                sell_vol = row["sell_volume"]
                if not np.isfinite(vwap):
                    continue
                if buy_vol > 0:
                    rows.append(
                        {"timestamp": ts, "price": vwap, "volume": buy_vol, "side": 1}
                    )
                if sell_vol > 0:
                    rows.append(
                        {"timestamp": ts, "price": vwap, "volume": sell_vol, "side": -1}
                    )

            if not rows:
                return None

            agg_df = pd.DataFrame(rows)
            return agg_df[["timestamp", "price", "volume", "side"]]

        except Exception as e:
            logger.error("Error preprocessing tick data: %s", e)
            return None

    def _generate_output_filename(self, zip_file: str, symbol: str) -> str:
        """
        生成输出文件名

        Args:
            zip_file: ZIP 文件路径
            symbol: 交易对符号

        Returns:
            输出文件完整路径

        文件名规则：
            - Monthly ZIP (SYMBOL-aggTrades-2025-12.zip) → SYMBOL_2025-12.parquet
            - Daily ZIP (SYMBOL-aggTrades-2025-12-01.zip) → SYMBOL_2025-12-01.parquet
        """
        zip_basename = os.path.basename(zip_file)

        # 先尝试提取完整日期 YYYY-MM-DD（daily 格式）
        match = re.search(r"(\d{4})-(\d{2})-(\d{2})", zip_basename)
        if match:
            date_part = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        else:
            # 再尝试提取月度格式 YYYY-MM（monthly 格式）
            match = re.search(r"(\d{4})-(\d{2})", zip_basename)
            if match:
                date_part = f"{match.group(1)}-{match.group(2)}"
            else:
                date_part = datetime.now().strftime("%Y-%m")
                logger.warning(
                    "Could not extract date from filename %s, using %s",
                    zip_basename,
                    date_part,
                )

        # 生成输出文件名
        output_filename = f"{symbol}_{date_part}.parquet"
        return os.path.join(self.output_dir, output_filename)

    def convert_all_files(
        self,
        pattern: str = "*aggTrades-*.zip",
        symbols: Optional[List[str]] = None,
        *,
        exclude_binance_monthly_aggtrade_zips: bool = False,
    ) -> Dict:
        """
        转换所有匹配的 ZIP 文件（支持多币种）

        Args:
            pattern: 文件匹配模式
            symbols: 可选的币种列表过滤（如 ["BTCUSDT", "ETHUSDT"]），None 表示不过滤
            exclude_binance_monthly_aggtrade_zips: True 时跳过 UM **按月** Vision ZIP，
                防止 broad glob 误选整月 raw agg 一次性读入内存触发 OOM。

        Returns:
            转换结果字典，包含成功和失败的文件列表
        """
        logger.info(f"Converting all files matching pattern: {pattern}")
        if symbols:
            logger.info(f"Filtering by symbols: {symbols}")

        # 查找所有匹配的zip文件
        zip_files = glob.glob(os.path.join(self.input_dir, pattern))

        # 如果指定了 symbols，则过滤文件列表
        if symbols:
            symbols_upper = [s.upper() for s in symbols]
            filtered_files = []
            for zf in zip_files:
                basename = os.path.basename(zf).upper()
                # 从文件名提取币种（支持 BTCUSDT-aggTrades-... 格式）
                match = re.search(r"([A-Z]+)(USDT|USD)", basename)
                if match:
                    detected_symbol = f"{match.group(1)}{match.group(2)}"
                    # 统一为 USDT 结尾
                    if detected_symbol.endswith("USD") and not detected_symbol.endswith(
                        "USDT"
                    ):
                        detected_symbol = detected_symbol[:-3] + "USDT"
                    if detected_symbol in symbols_upper:
                        filtered_files.append(zf)
            zip_files = filtered_files
            logger.info(
                f"Filtered to {len(zip_files)} files matching symbols: {symbols}"
            )

        if exclude_binance_monthly_aggtrade_zips:
            before_ct = len(zip_files)
            zip_files = [
                z
                for z in zip_files
                if not is_binance_um_monthly_aggtrade_zip(os.path.basename(z))
            ]
            skipped_m = before_ct - len(zip_files)
            if skipped_m:
                logger.warning(
                    "Excluded %s Binance UM monthly aggTrades ZIP(s) "
                    "(exclude_binance_monthly_aggtrade_zips=True).",
                    skipped_m,
                )

        logger.info(f"Found {len(zip_files)} files to convert")

        converted_files: List[Dict] = []
        skipped_files: List[Dict] = []
        failed_files: List[str] = []

        total_files = len(zip_files)
        if total_files == 0:
            logger.warning("No matching ZIP files found for conversion.")
            return {
                "converted_files": converted_files,
                "failed_files": failed_files,
                "total_files": total_files,
            }

        for index, zip_file in enumerate(zip_files, start=1):
            file_name = os.path.basename(zip_file)
            progress_prefix = f"[{index}/{total_files}]"

            # 先检查是否已转换，再打印日志（避免误导用户）
            # 从文件名提取 symbol
            upper_name = file_name.upper()
            sym_match = re.search(r"([A-Z]+)(USDT|USD)", upper_name)
            if sym_match:
                detected_symbol = f"{sym_match.group(1)}{'USDT' if sym_match.group(2) != 'USDT' else sym_match.group(2)}"
            else:
                detected_symbol = "UNKNOWN"
            output_file = self._generate_output_filename(zip_file, detected_symbol)
            if (
                (not self.force)
                and os.path.exists(output_file)
                and os.path.getsize(output_file) > 0
            ):
                print(f"{progress_prefix} ⏩ Skip (cached): {file_name}")
                skipped_files.append(
                    {
                        "original_file": zip_file,
                        "output_file": output_file,
                        "skipped": True,
                    }
                )
                continue

            print(f"{progress_prefix} 📦 Converting {file_name} ...")
            result = self.convert_zip_to_parquet(zip_file)
            if result:
                if bool(result.get("skipped")):
                    # 不应该走到这里，但保留兼容性
                    skipped_files.append(result)
                    print(f"{progress_prefix} ⏩ Skip (already converted): {file_name}")
                else:
                    converted_files.append(result)
                    print(f"{progress_prefix} ✅ Success: {file_name}")
                    # 可选：备份原始 ZIP 到 backup_dir
                    if self.backup_dir is not None:
                        try:
                            backup_path = os.path.join(self.backup_dir, file_name)
                            shutil.copy2(zip_file, backup_path)
                        except Exception as e:
                            logger.warning(
                                "Failed to backup %s to %s: %s",
                                file_name,
                                self.backup_dir,
                                e,
                            )
            else:
                failed_files.append(zip_file)
                print(f"{progress_prefix} ❌ Failed: {file_name}")

        logger.info(
            f"Conversion complete: {len(converted_files)} successful, {len(failed_files)} failed"
        )

        return {
            "converted_files": converted_files,
            "skipped_files": skipped_files,
            "failed_files": failed_files,
            "total_files": len(zip_files),
        }


def main():
    """命令行入口点"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Binance ZIP aggTrades to Parquet (tick data)"
    )
    parser.add_argument(
        "--pattern",
        default="*aggTrades-*.zip",
        help="ZIP filename glob pattern (default: *aggTrades-*.zip). Example: BNBUSDT-aggTrades-2024-*.zip",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="ZIP input directory (default: data/agg_data)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Parquet output directory (default: data/parquet_data)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-convert even if output parquet already exists.",
    )
    parser.add_argument(
        "--aggregate-freq",
        default="1min",
        help="Aggregation frequency for tick data (default: 1min). "
        "Examples: '1s' (1 second), '1T' (1 minute), '5T' (5 minutes). "
        "Uses pandas resample frequency strings.",
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated list of symbols to convert (e.g., BTCUSDT,ETHUSDT). "
        "If not specified, all matching files will be converted.",
    )
    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("🚀 Converting ZIP files to Parquet format...")

    # 配置路径（使用绝对路径，指向仓库根目录）
    # 从模块路径推断项目根目录
    current_file = Path(__file__).resolve()
    base_dir = current_file.parents[2]  # src/data_tools -> src -> project root
    input_dir = args.input_dir or str(base_dir / "data" / "agg_data")
    output_dir = args.output_dir or str(base_dir / "data" / "parquet_data")

    print(f"📂 Base directory: {base_dir}")
    print(f"📂 Input directory: {input_dir}")
    print(f"📂 Output directory: {output_dir}")
    print()

    # 检查输入目录
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        print(f"💡 Expected data directory structure:")
        print(f"   {base_dir}/data/agg_data/")
        return

    # 创建转换器
    converter = DataConverter(
        input_dir,
        output_dir,
        force=bool(args.force),
        aggregate_freq=args.aggregate_freq,
    )

    # 转换所有文件
    symbols_list = None
    if args.symbols:
        symbols_list = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        print(f"🎯 Filtering by symbols: {', '.join(symbols_list)}")
        print()
    results = converter.convert_all_files(
        pattern=str(args.pattern), symbols=symbols_list
    )

    # 打印结果
    print(f"\n📊 Conversion Results:")
    print(f"   Total files: {results['total_files']}")
    print(f"   Converted: {len(results['converted_files'])}")
    print(f"   Skipped: {len(results.get('skipped_files', []))}")
    print(f"   Failed: {len(results['failed_files'])}")

    if results["converted_files"]:
        print(f"\n✅ Successfully converted files:")
        for file_info in results["converted_files"][:5]:  # 显示前5个
            print(
                f"   {os.path.basename(file_info['original_file'])} -> {os.path.basename(file_info['output_file'])}"
            )

        if len(results["converted_files"]) > 5:
            print(f"   ... and {len(results['converted_files']) - 5} more files")

    if results["failed_files"]:
        print(f"\n❌ Failed files:")
        for failed_file in results["failed_files"][:5]:  # 显示前5个
            print(f"   {os.path.basename(failed_file)}")

        if len(results["failed_files"]) > 5:
            print(f"   ... and {len(results['failed_files']) - 5} more files")

    print(f"\n🎉 Data conversion complete!")
    print(f"   Output directory: {output_dir}")
    print(f"   ZIP files preserved in: {input_dir}")
    print(f"   Ready for fast processing!")


if __name__ == "__main__":
    main()
