#!/usr/bin/env python3
"""
批量下载Binance历史交易数据
====================================

功能：
1. 下载BTC、ETH、SOL的aggTrades数据
2. 时间范围：默认 2021年1月 - 当前月份（可通过参数覆盖）
3. 自动跳过已存在的文件
4. 支持断点续传
5. 显示下载进度和统计
"""

import os
import sys
import requests
import zipfile
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import argparse


class BinanceMultiSymbolDownloader:
    """Binance多币种历史数据下载器

    简化设计：
    - data_dir (agg_data/) 存放下载的 ZIP（永久保留）
    - parquet_dir (parquet_data/) 存放转换后的 Parquet
    - 检查顺序：parquet -> zip，已存在则跳过下载
    """

    def __init__(
        self,
        data_dir: str = "data/agg_data",
        parquet_dir: Optional[str] = None,
        granularity: str = "monthly",  # "monthly" or "daily"
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_dir = Path(parquet_dir) if parquet_dir else None
        self.granularity = granularity

        # Binance数据基础URL
        if granularity == "daily":
            self.base_url = (
                "https://data.binance.vision/data/futures/um/daily/aggTrades"
            )
        else:
            self.base_url = (
                "https://data.binance.vision/data/futures/um/monthly/aggTrades"
            )

        # Reusable HTTP session (better resilience for large ZIP downloads)
        self.session = requests.Session()
        try:
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            retry = Retry(
                total=5,
                connect=5,
                read=5,
                backoff_factor=1.0,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset(["GET"]),
                raise_on_status=False,
            )
            adapter = HTTPAdapter(
                max_retries=retry, pool_connections=10, pool_maxsize=10
            )
            self.session.mount("https://", adapter)
            self.session.mount("http://", adapter)
        except Exception:
            # If retry tooling is unavailable, fall back to plain Session.
            pass

        # 支持的交易对
        self.symbols = {
            "BTCUSDT": "Bitcoin",
            "ETHUSDT": "Ethereum",
            "SOLUSDT": "Solana",
            "BNBUSDT": "BNB",
            "XRPUSDT": "Ripple",
            "ADAUSDT": "Cardano",
            "DOGEUSDT": "Dogecoin",
            "DOTUSDT": "Polkadot",
            "MATICUSDT": "Polygon",
            "SHIBUSDT": "Shiba Inu",
        }
        self.default_symbols = list(self.symbols.keys())
        self.symbol_aliases = {
            "BTC": "BTCUSDT",
            "ETH": "ETHUSDT",
            "SOL": "SOLUSDT",
            "BNB": "BNBUSDT",
            "XRP": "XRPUSDT",
            "ADA": "ADAUSDT",
            "DOGE": "DOGEUSDT",
            "DOT": "DOTUSDT",
            "MATIC": "MATICUSDT",
            "SHIB": "SHIBUSDT",
        }

        # 下载统计
        self.stats = {
            "total": 0,
            "downloaded": 0,
            "skipped": 0,
            "failed": 0,
            "total_size_mb": 0,
        }

    def get_month_list(
        self,
        start_year: int = 2021,
        start_month: int = 1,
        end_year: Optional[int] = None,
        end_month: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """生成月份列表"""
        if end_year is None or end_month is None:
            today = datetime.utcnow()
            end_year = int(end_year or today.year)
            end_month = int(end_month or today.month)

        months = []
        current_year = start_year
        current_month = start_month

        while (current_year < end_year) or (
            current_year == end_year and current_month <= end_month
        ):
            months.append((current_year, current_month))
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1

        return months

    def get_date_list(
        self,
        start_date: str,  # "YYYY-MM-DD"
        end_date: Optional[str] = None,  # "YYYY-MM-DD"
    ) -> List[str]:
        """生成日期列表（用于每日数据下载）

        Args:
            start_date: 开始日期 "YYYY-MM-DD"
            end_date: 结束日期 "YYYY-MM-DD"，默认为昨天（Binance每日数据有延迟）

        Returns:
            日期列表 ["YYYY-MM-DD", ...]
        """
        from datetime import datetime, timedelta

        start = datetime.strptime(start_date, "%Y-%m-%d")

        if end_date is None:
            # 默认到昨天（因为Binance每日数据次日10:00 UTC生成）
            end = datetime.utcnow() - timedelta(days=1)
        else:
            end = datetime.strptime(end_date, "%Y-%m-%d")

        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)

        return dates

    def normalize_symbol(self, symbol: str) -> Optional[str]:
        """标准化交易对名称，支持别名（如BTC -> BTCUSDT）"""
        if not symbol:
            return None

        normalized = symbol.upper().replace("-", "").replace("/", "")

        if normalized in self.symbol_aliases:
            normalized = self.symbol_aliases[normalized]
        elif not normalized.endswith("USDT"):
            normalized = f"{normalized}USDT"

        if normalized not in self.symbols:
            # 对于未预置的币种，使用符号本身作为名称
            self.symbols[normalized] = normalized

        return normalized

    def _parquet_symbol(self, symbol: str) -> str:
        """Normalize symbol for parquet filenames (e.g., BTCUSDT)."""
        normalized = symbol.upper().replace("-", "").replace("/", "")
        if not normalized.endswith("USDT"):
            normalized = f"{normalized}USDT"
        return normalized

    def check_local_file(
        self, symbol: str, year: int, month: int, day: Optional[int] = None
    ) -> Optional[str]:
        """检查本地是否已具备该月份/日期数据：优先检查 Parquet；其次检查 ZIP

        Returns:
            "parquet" if parquet exists, "zip" if zip exists, None if missing
        """
        # 1) 检查 Parquet
        if self.parquet_dir:
            parquet_symbol = self._parquet_symbol(symbol)
            if day is not None:
                # 每日数据
                parquet_name = f"{parquet_symbol}_{year}-{month:02d}-{day:02d}.parquet"
            else:
                # 月度数据
                parquet_name = f"{parquet_symbol}_{year}-{month:02d}.parquet"
            parquet_path = self.parquet_dir / parquet_name
            if parquet_path.exists() and parquet_path.stat().st_size > 0:
                print(f"✅ Parquet 已存在: {parquet_name}")
                return "parquet"

        # 2) 检查 ZIP
        if day is not None:
            filename = f"{symbol}-aggTrades-{year}-{month:02d}-{day:02d}.zip"
        else:
            filename = f"{symbol}-aggTrades-{year}-{month:02d}.zip"
        file_path = self.data_dir / filename

        if not file_path.exists():
            return None

        # 检查文件大小（至少应该有1MB，避免下载不完整的文件）
        file_size = file_path.stat().st_size
        min_size = 1 * 1024 * 1024 if day is None else 100 * 1024  # 每日文件至少100KB
        if file_size < min_size:
            print(f"   ⚠️  {filename} 文件太小 ({file_size} bytes)，将重新下载")
            file_path.unlink()  # 删除不完整的文件
            return None

        print(f"✅ ZIP 已存在: {filename}")
        return "zip"

    def download_file(
        self,
        symbol: str,
        year: int,
        month: int,
        day: Optional[int] = None,
        retry_times: int = 3,
        timeout: int = 600,
    ) -> bool:
        """下载单个文件"""
        if day is not None:
            filename = f"{symbol}-aggTrades-{year}-{month:02d}-{day:02d}.zip"
        else:
            filename = f"{symbol}-aggTrades-{year}-{month:02d}.zip"
        file_path = self.data_dir / filename
        tmp_path = file_path.with_suffix(file_path.suffix + ".part")
        url = f"{self.base_url}/{symbol}/{filename}"

        for attempt in range(retry_times):
            try:
                print(
                    f"   📥 下载 {filename} (尝试 {attempt + 1}/{retry_times})...",
                    end=" ",
                )

                # Resume support: keep partial download as .part and continue via Range.
                existing = tmp_path.stat().st_size if tmp_path.exists() else 0
                headers = {"Accept-Encoding": "identity"}
                if existing > 0:
                    headers["Range"] = f"bytes={existing}-"

                # 发送请求（use Session + retries）
                response = self.session.get(
                    url, timeout=(10, timeout), stream=True, headers=headers
                )
                response.raise_for_status()

                # 获取文件大小
                content_len = int(response.headers.get("content-length", 0))
                total_size = (
                    int(response.headers.get("content-range", "").split("/")[-1] or 0)
                    if response.headers.get("content-range")
                    else 0
                )
                if total_size <= 0:
                    total_size = content_len + existing if content_len > 0 else 0

                # If server ignored Range and returned full content, restart.
                if existing > 0 and response.status_code == 200:
                    existing = 0
                    if tmp_path.exists():
                        tmp_path.unlink()

                # 下载文件
                chunk_size = 1024 * 256  # 256KB
                mode = "ab" if existing > 0 else "wb"
                with open(tmp_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)

                # 验证下载
                final_size = tmp_path.stat().st_size
                size_mb = final_size / 1024 / 1024

                if total_size > 0 and final_size != total_size:
                    raise Exception(f"文件大小不匹配: {final_size} != {total_size}")

                # Atomically move into place
                tmp_path.replace(file_path)

                print(f"✅ ({size_mb:.1f}MB)")
                self.stats["downloaded"] += 1
                self.stats["total_size_mb"] += size_mb
                return True

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    print(f"❌ 文件不存在 (404)")
                    return False  # 不重试404错误
                else:
                    print(f"❌ HTTP错误 {e.response.status_code}")

            except Exception as e:
                print(f"❌ {str(e)}")

            # Keep .part for resume, but remove any zero-byte temp
            try:
                if tmp_path.exists() and tmp_path.stat().st_size == 0:
                    tmp_path.unlink()
            except Exception:
                pass

            # 等待后重试
            if attempt < retry_times - 1:
                wait_time = (attempt + 1) * 2  # 递增等待时间
                print(f"   ⏳ 等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)

        self.stats["failed"] += 1
        return False

    def download_symbol_data(self, symbol: str, months: List[Tuple[int, int]]) -> Dict:
        """下载单个币种的所有数据"""
        symbol_name = self.symbols.get(symbol, symbol)
        print(f"\n{'='*60}")
        print(f"📊 {symbol_name} ({symbol})")
        print(f"{'='*60}")

        symbol_stats = {
            "total": len(months),
            "downloaded": 0,
            "skipped": 0,
            "failed": 0,
        }

        for i, (year, month) in enumerate(months, 1):
            month_str = f"{year}-{month:02d}"
            print(f"[{i}/{len(months)}] {month_str}:", end=" ")

            self.stats["total"] += 1

            # 检查本地是否已有文件
            existing_type = self.check_local_file(symbol, year, month)
            if existing_type:
                if existing_type == "zip":
                    filename = f"{symbol}-aggTrades-{year}-{month:02d}.zip"
                    file_path = self.data_dir / filename
                    size_mb = file_path.stat().st_size / 1024 / 1024
                    print(f"✅ 已存在 ({size_mb:.1f}MB)")
                symbol_stats["skipped"] += 1
                self.stats["skipped"] += 1
                continue

            # 下载文件
            if self.download_file(symbol, year, month):
                symbol_stats["downloaded"] += 1
            else:
                symbol_stats["failed"] += 1

            # 避免请求过快
            time.sleep(0.5)

        return symbol_stats

    def download_symbol_daily_data(self, symbol: str, dates: List[str]) -> Dict:
        """下载单个币种的每日数据

        Args:
            symbol: 交易对符号
            dates: 日期列表 ["YYYY-MM-DD", ...]

        Returns:
            统计信息字典
        """
        symbol_name = self.symbols.get(symbol, symbol)
        print(f"\n{'='*60}")
        print(f"📊 {symbol_name} ({symbol})")
        print(f"{'='*60}")

        symbol_stats = {
            "total": len(dates),
            "downloaded": 0,
            "skipped": 0,
            "failed": 0,
        }

        for i, date_str in enumerate(dates, 1):
            print(f"[{i}/{len(dates)}] {date_str}:", end=" ")

            self.stats["total"] += 1

            # 解析日期
            year, month, day = map(int, date_str.split("-"))

            # 检查本地是否已有文件
            existing_type = self.check_local_file(symbol, year, month, day)
            if existing_type:
                if existing_type == "zip":
                    filename = f"{symbol}-aggTrades-{date_str}.zip"
                    file_path = self.data_dir / filename
                    size_kb = file_path.stat().st_size / 1024
                    print(f"✅ 已存在 ({size_kb:.1f}KB)")
                symbol_stats["skipped"] += 1
                self.stats["skipped"] += 1
                continue

            # 下载文件
            if self.download_file(symbol, year, month, day):
                symbol_stats["downloaded"] += 1
            else:
                symbol_stats["failed"] += 1

            # 避免请求过快
            time.sleep(0.5)

        return symbol_stats

    def download_all(
        self,
        start_year: int = 2021,
        start_month: int = 1,
        end_year: int = 2025,
        end_month: int = 9,
        symbols: List[str] = None,
        auto_confirm: bool = False,
    ) -> None:
        """下载所有币种的数据"""
        print("\n" + "=" * 60)
        print("🚀 Binance 历史交易数据批量下载器")
        print("=" * 60)
        print(
            f"📅 时间范围: {start_year}-{start_month:02d} 至 {end_year}-{end_month:02d}"
        )
        print(f"📂 ZIP目录: {self.data_dir.absolute()}")
        if self.parquet_dir:
            print(f"📂 Parquet目录: {self.parquet_dir.absolute()}")

        # 确定要下载的币种
        if symbols is None:
            symbols = list(self.default_symbols)
        else:
            normalized = []
            for raw_symbol in symbols:
                normalized_symbol = self.normalize_symbol(raw_symbol)
                if normalized_symbol:
                    normalized.append(normalized_symbol)
            # 去重同时保持顺序
            symbols = list(dict.fromkeys(normalized))
            if not symbols:
                print("❌ 未找到有效的交易对，请检查输入。")
                return

        print(f"💰 币种: {', '.join(symbols)}")

        # 生成月份列表
        months = self.get_month_list(start_year, start_month, end_year, end_month)
        total_files = len(months) * len(symbols)

        print(f"📦 预计文件数: {total_files} ({len(months)} 月 × {len(symbols)} 币种)")
        print()

        # 确认开始
        if auto_confirm:
            print("✅ 自动确认模式，跳过交互确认")
        else:
            try:
                response = input(
                    "⚠️  这可能需要几小时时间和大量磁盘空间。是否继续? (y/N): "
                )
                if response.lower() != "y":
                    print("❌ 已取消")
                    return
            except EOFError:
                print("✅ 非交互环境，自动继续")

        start_time = time.time()

        # 逐个币种下载
        for symbol in symbols:
            symbol_stats = self.download_symbol_data(symbol, months)
            print(f"\n{symbol} 统计:")
            print(f"  ✅ 下载: {symbol_stats['downloaded']}")
            print(f"  ⏭️  跳过: {symbol_stats['skipped']}")
            print(f"  ❌ 失败: {symbol_stats['failed']}")

        # 总体统计
        elapsed_time = time.time() - start_time
        elapsed_minutes = elapsed_time / 60

        print("\n" + "=" * 60)
        print("📊 下载统计")
        print("=" * 60)
        print(f"总文件数: {self.stats['total']}")
        print(f"✅ 新下载: {self.stats['downloaded']}")
        print(f"⏭️  已跳过: {self.stats['skipped']}")
        print(f"❌ 失败: {self.stats['failed']}")
        print(f"📦 总大小: {self.stats['total_size_mb']:.1f} MB")
        print(f"⏱️  总耗时: {elapsed_minutes:.1f} 分钟")

        if self.stats["downloaded"] > 0:
            avg_speed = (
                self.stats["total_size_mb"] / elapsed_minutes
                if elapsed_minutes > 0
                else 0
            )
            print(f"⚡ 平均速度: {avg_speed:.1f} MB/分钟")

        print("\n✅ 下载完成！")
        print(f"📂 数据保存在: {self.data_dir.absolute()}")

    def list_downloaded_files(self) -> Dict[str, List[str]]:
        """列出已下载的文件"""
        downloaded = {symbol: [] for symbol in self.symbols.keys()}

        for file_path in self.data_dir.glob("*-aggTrades-*.zip"):
            filename = file_path.name
            for symbol in self.symbols.keys():
                if filename.startswith(symbol):
                    # 提取日期
                    date_part = filename.replace(f"{symbol}-aggTrades-", "").replace(
                        ".zip", ""
                    )
                    downloaded[symbol].append(date_part)

        return {k: sorted(v) for k, v in downloaded.items()}

    def print_summary(self) -> None:
        """打印下载摘要"""
        print("\n" + "=" * 60)
        print("📁 本地数据摘要")
        print("=" * 60)

        downloaded = self.list_downloaded_files()

        for symbol, dates in downloaded.items():
            symbol_name = self.symbols[symbol]
            print(f"\n{symbol_name} ({symbol}): {len(dates)} 个月")

            if dates:
                print(f"  最早: {dates[0]}")
                print(f"  最新: {dates[-1]}")

                # 计算总大小
                total_size = 0
                for date in dates:
                    filename = f"{symbol}-aggTrades-{date}.zip"
                    file_path = self.data_dir / filename
                    if file_path.exists():
                        total_size += file_path.stat().st_size

                print(f"  大小: {total_size / 1024 / 1024:.1f} MB")

                # 检查缺失的月份
                if len(dates) > 1:
                    # 简单检查：应该有连续的月份
                    expected_count = self._count_months_between(dates[0], dates[-1])
                    if expected_count > len(dates):
                        missing = expected_count - len(dates)
                        print(f"  ⚠️  可能缺失 {missing} 个月的数据")

    def _count_months_between(self, start_date: str, end_date: str) -> int:
        """计算两个日期之间的月份数"""
        start_year, start_month = map(int, start_date.split("-"))
        end_year, end_month = map(int, end_date.split("-"))
        return (end_year - start_year) * 12 + (end_month - start_month) + 1


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="下载Binance历史交易数据 (2021-2025)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载所有币种的全部数据
  python download_training_data.py
  
  # 只下载BTC数据
  python download_training_data.py --symbols BTCUSDT
  
  # 下载BTC和ETH
  python download_training_data.py --symbols BTCUSDT ETHUSDT
  
  # 指定时间范围
  python download_training_data.py --start-year 2024 --start-month 1
  
  # 查看已下载的文件
  python download_training_data.py --summary
        """,
    )

    parser.add_argument(
        "--data-dir", default="data/agg_data", help="ZIP保存目录 (默认: data/agg_data)"
    )
    parser.add_argument(
        "--parquet-dir", default=None, help="Parquet目录 (提供后将跳过已存在月份)"
    )
    parser.add_argument(
        "--symbols", nargs="+", help="指定要下载的币种（支持 BTC、BTCUSDT 等别名）"
    )
    parser.add_argument(
        "--start-year", type=int, default=2021, help="开始年份 (默认: 2021)"
    )
    parser.add_argument("--start-month", type=int, default=1, help="开始月份 (默认: 1)")
    parser.add_argument(
        "--end-year", type=int, default=None, help="结束年份 (默认: 当前年份)"
    )
    parser.add_argument(
        "--end-month", type=int, default=None, help="结束月份 (默认: 当前月份)"
    )
    parser.add_argument("--summary", action="store_true", help="只显示已下载文件的摘要")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="非交互模式：跳过确认提示，直接执行下载。",
    )

    args = parser.parse_args()

    # 创建下载器
    downloader = BinanceMultiSymbolDownloader(args.data_dir, args.parquet_dir)

    if args.summary:
        # 只显示摘要
        downloader.print_summary()
    else:
        # Default end date: current UTC month (unless caller specifies)
        if args.end_year is None or args.end_month is None:
            today = datetime.utcnow()
            if args.end_year is None:
                args.end_year = today.year
            if args.end_month is None:
                args.end_month = today.month

        # 执行下载
        # 执行下载（支持非交互模式）
        if args.yes:
            # Monkey patch: avoid blocking input() in download_all
            import builtins

            old_input = builtins.input
            builtins.input = lambda *_args, **_kwargs: "y"
            try:
                downloader.download_all(
                    start_year=args.start_year,
                    start_month=args.start_month,
                    end_year=int(args.end_year),
                    end_month=int(args.end_month),
                    symbols=args.symbols,
                )
            finally:
                builtins.input = old_input
        else:
            downloader.download_all(
                start_year=args.start_year,
                start_month=args.start_month,
                end_year=int(args.end_year),
                end_month=int(args.end_month),
                symbols=args.symbols,
            )

        # 显示最终摘要
        downloader.print_summary()


if __name__ == "__main__":
    main()
