"""Helpers for loading Binance aggTrades (tick-level) data on demand."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd


def _parse_timeframe_to_minutes(tf: str) -> Optional[int]:
    """Utility used by callers; kept here for convenience."""
    tf = tf.strip().upper()
    if tf.endswith("T"):
        try:
            return int(tf[:-1])
        except ValueError:
            return None
    if tf.endswith("H"):
        try:
            return int(float(tf[:-1]) * 60)
        except ValueError:
            return None
    if tf.endswith("D"):
        try:
            return int(float(tf[:-1]) * 1440)
        except ValueError:
            return None
    if tf.isdigit():
        return int(tf)
    return None


def _month_range(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> List[Tuple[int, int]]:
    start_period = start_ts.to_period("M")
    end_period = end_ts.to_period("M")
    periods = pd.period_range(start_period, end_period, freq="M")
    return [(p.year, p.month) for p in periods]


def _detect_csv_name(zip_ref: zipfile.ZipFile) -> str:
    for name in zip_ref.namelist():
        if name.endswith(".csv"):
            return name
    return zip_ref.namelist()[0]


def _read_tick_csv_from_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_name = _detect_csv_name(zf)
        with zf.open(csv_name) as handle:
            first_line = handle.readline().decode("utf-8", errors="ignore")
        read_params = {"low_memory": False}
        if first_line.strip().split(",")[0].replace(".", "").isdigit():
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
        with zf.open(csv_name) as handle:
            df = pd.read_csv(handle, **read_params)
    return df


def _load_month_ticks(
    symbol: str,
    year: int,
    month: int,
    ticks_dir: Path,
    cache_dir: Optional[Path],
) -> pd.DataFrame:
    cache_file: Optional[Path] = None
    if cache_dir is not None:
        cache_file = cache_dir / symbol / f"{symbol}_{year}-{month:02d}.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)

    zip_path = ticks_dir / f"{symbol}-aggTrades-{year}-{month:02d}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Missing tick ZIP: {zip_path}. Run `make data-download` first."
        )
    raw_df = _read_tick_csv_from_zip(zip_path)
    if "transact_time" not in raw_df.columns or "price" not in raw_df.columns:
        raise ValueError(f"Unexpected schema inside {zip_path}")
    ticks_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw_df["transact_time"], unit="ms"),
            "price": pd.to_numeric(raw_df["price"], errors="coerce"),
            "volume": pd.to_numeric(
                raw_df.get("quantity", raw_df.get("volume")), errors="coerce"
            ),
        }
    ).dropna()

    if "is_buyer_maker" in raw_df.columns:
        sides = np.where(raw_df["is_buyer_maker"].astype(bool), -1, 1)
    else:
        sides = np.sign(ticks_df["volume"]).replace(0, 1)
    ticks_df["side"] = sides

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        ticks_df.to_parquet(cache_file, index=False)

    return ticks_df


def load_tick_data(
    symbol: str,
    start_ts: str,
    end_ts: str,
    ticks_dir: str = "data/parquet_data",
    lookback_minutes: int = 60,
) -> pd.DataFrame:
    """Load tick parquet range directly (used for ad-hoc analyses)."""
    tick_files = list_tick_files(
        symbol, start_ts, end_ts, ticks_dir=ticks_dir, lookback_minutes=lookback_minutes
    )
    frames = []
    for path in tick_files:
        df = pd.read_parquet(path)
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        raise ValueError("No tick data loaded; please verify tick parquet path.")

    ticks = pd.concat(frames, ignore_index=True)
    # Normalize to tz-naive timestamps (UTC) for robust comparisons.
    # concat of mixed tz-aware / tz-naive parquet files may produce object dtype,
    # so always round-trip through utc=True then strip tz.
    ticks["timestamp"] = pd.to_datetime(ticks["timestamp"], utc=True).dt.tz_convert(
        None
    )
    start = pd.to_datetime(start_ts, utc=True).tz_convert(None) - pd.Timedelta(
        minutes=lookback_minutes
    )
    end = pd.to_datetime(end_ts, utc=True).tz_convert(None) + pd.Timedelta(
        minutes=lookback_minutes
    )
    mask = (ticks["timestamp"] >= start) & (ticks["timestamp"] <= end)
    ticks = ticks.loc[mask]
    if ticks.empty:
        raise ValueError("Tick data is empty after filtering by time range.")

    ticks = ticks.sort_values("timestamp").set_index("timestamp")
    return ticks[["price", "volume", "side"]]


def serialize_tick_loader_params(params: dict) -> str:
    """Utility to store loader params inside feature compute_params."""
    return json.dumps(params)


def deserialize_tick_loader_params(payload: str) -> dict:
    """
    反序列化 tick 加载器参数

    Args:
        payload: JSON 字符串，包含 tick 加载器参数

    Returns:
        dict: 包含以下键的字典：
            - symbol: 交易对符号
            - tick_files: tick 文件路径列表
            - start_ts: 开始时间戳
            - end_ts: 结束时间戳
            - lookback_minutes: 回看分钟数

    Raises:
        ValueError: 如果 payload 格式不正确或缺少必需字段
        json.JSONDecodeError: 如果 payload 不是有效的 JSON
    """
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to deserialize ticks_loader_json: Invalid JSON format. "
            f"Error: {e}. Payload preview: {payload[:200]}..."
        ) from e

    # 验证必需字段
    required_fields = ["symbol", "tick_files", "start_ts", "end_ts"]
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(
            f"Failed to deserialize ticks_loader_json: Missing required fields: {missing_fields}. "
            f"Available fields: {list(data.keys())}"
        )

    # 验证 tick_files 不为空
    if not data.get("tick_files"):
        raise ValueError(
            f"Failed to deserialize ticks_loader_json: tick_files is empty. "
            f"Symbol: {data.get('symbol')}, Time range: {data.get('start_ts')} to {data.get('end_ts')}"
        )

    return data


def list_tick_files(
    symbol: str,
    start_ts: str,
    end_ts: str,
    ticks_dir: str,
    lookback_minutes: int = 60,
) -> List[str]:
    """
    Return existing monthly tick parquet files covering [start_ts, end_ts].

    Note: Only returns files that actually exist. If a month's file is missing,
    it will be skipped (not raise an error) to allow partial data processing.
    """
    ticks_root = Path(ticks_dir)
    # Normalize to tz-naive timestamps (UTC) for robust comparisons.
    # Upstream callers may pass tz-aware timestamps (e.g., UTC-indexed feature frames),
    # while tick parquet timestamps are tz-naive. Mixing them causes:
    # "Cannot compare tz-naive and tz-aware timestamps".
    start = pd.to_datetime(start_ts, utc=True).tz_convert(None) - pd.Timedelta(
        minutes=lookback_minutes
    )
    end = pd.to_datetime(end_ts, utc=True).tz_convert(None) + pd.Timedelta(
        minutes=lookback_minutes
    )
    months = _month_range(start, end)

    tick_files: List[str] = []
    missing_files: List[str] = []
    for year, month in months:
        file_path = ticks_root / f"{symbol}_{year}-{month:02d}.parquet"
        if file_path.exists():
            tick_files.append(str(file_path))
        else:
            missing_files.append(str(file_path))

    if not tick_files:
        raise FileNotFoundError(
            f"No tick parquet files found for {symbol} in time range [{start_ts}, {end_ts}]. "
            f"Missing files: {missing_files[:5]}... "
            "Please run 'make data-convert' or 'python -m src.data_tools.zip_to_parquet' first."
        )

    if missing_files:
        import warnings

        warnings.warn(
            f"Some tick files are missing (will use available data only): {missing_files[:5]}..."
        )

    return sorted(tick_files)


def build_tick_loader_payload(
    symbol: str,
    start_ts: str,
    end_ts: str,
    ticks_dir: str,
    lookback_minutes: int = 60,
) -> str:
    tick_files = list_tick_files(
        symbol,
        start_ts,
        end_ts,
        ticks_dir=ticks_dir,
        lookback_minutes=lookback_minutes,
    )
    payload = {
        "symbol": symbol.upper(),
        "tick_files": tick_files,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "lookback_minutes": lookback_minutes,
    }
    return serialize_tick_loader_params(payload)


def _estimate_bucket_volume_from_cache(
    cache_files: List[str],
    lookback_days: int = 7,
    quantile: float = 0.3,
) -> float:
    hourly_parts = []
    for path_str in cache_files:
        path = Path(path_str)
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["timestamp", "volume"])
        if df.empty:
            continue
        df = df.dropna(subset=["timestamp", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        hourly_parts.append(df["volume"].resample("1H").sum())

    if not hourly_parts:
        return 100.0

    hourly = pd.concat(hourly_parts).sort_index()
    lookback_hours = max(1, lookback_days * 24)
    typical = hourly.rolling(window=lookback_hours, min_periods=1).quantile(quantile)
    bucket_volume = (
        float(typical.iloc[-1]) if not typical.empty else float(hourly.mean())
    )
    return max(bucket_volume, 1e-6)


def _get_monthly_vpin_cache_key(
    file_path: str,
    bucket_volume: float,
    bucket_volume_usd: Optional[float] = None,
    prev_bucket_state: Optional[dict] = None,
) -> str:
    """
    生成按月VPIN缓存的键（不包含 start/end，按月完整计算）

    优化：按月完整计算并缓存，不同时间窗口可以复用同一月份的缓存

    如果 prev_bucket_state 不为空，将其信息加入缓存键，以支持跨月连续性的缓存
    """
    import hashlib
    import json

    path = Path(file_path)
    month_str = path.stem.split("_")[-1] if "_" in path.stem else path.stem
    if bucket_volume_usd is not None:
        key_str = f"vpin_monthly_usd_{month_str}_{bucket_volume_usd:.6f}"
    else:
        key_str = f"vpin_monthly_{month_str}_{bucket_volume:.6f}"

    # 如果 prev_bucket_state 不为空，将其信息加入缓存键
    if prev_bucket_state is not None:
        # 将 prev_bucket_state 序列化为字符串（使用固定精度避免浮点误差）
        state_str = json.dumps(
            {
                "buy": round(prev_bucket_state.get("current_buy", 0.0), 6),
                "sell": round(prev_bucket_state.get("current_sell", 0.0), 6),
                "filled": round(prev_bucket_state.get("filled_value", 0.0), 6),
            },
            sort_keys=True,
        )
        key_str = f"{key_str}_state_{state_str}"

    return hashlib.md5(key_str.encode()).hexdigest()


def _load_monthly_vpin_cache(
    cache_dir: Path, cache_key: str
) -> Optional[tuple[List[Tuple[pd.Timestamp, float]], dict]]:
    """
    从缓存加载单月的VPIN buckets和final_state

    返回: (buckets, final_state) 或 None
    支持多种缓存格式：
    1. 旧格式：只有buckets（list）
    2. 标准格式：(buckets, final_state) tuple
    3. 优化格式：只有final_state（dict），表示标准缓存只存了final_state
    """
    cache_file = cache_dir / f"{cache_key}.pkl"
    if cache_file.exists():
        try:
            import pickle

            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
                # 检查缓存格式
                if isinstance(cached_data, list):
                    # 旧格式：只有buckets，假设final_state为空
                    empty_state = {
                        "current_buy": 0.0,
                        "current_sell": 0.0,
                        "filled_value": 0.0,
                    }
                    return (cached_data, empty_state)
                elif isinstance(cached_data, dict):
                    # 优化格式：只有final_state（标准缓存）
                    # 返回 None buckets，表示需要重新计算
                    return (None, cached_data)
                elif isinstance(cached_data, tuple) and len(cached_data) == 2:
                    # 标准格式：(buckets, final_state) 或 (None, final_state)
                    return cached_data
                else:
                    print(f"      ⚠️  Invalid cache format for {cache_key}", flush=True)
                    return None
        except Exception as e:
            print(f"      ⚠️  Failed to load cache {cache_key}: {e}", flush=True)
    return None


def _save_monthly_vpin_cache(
    cache_dir: Path,
    cache_key: str,
    buckets: Optional[List[Tuple[pd.Timestamp, float]]],
    final_state: Optional[dict] = None,
    save_buckets: bool = True,
):
    """
    保存单月的VPIN buckets和final_state到缓存

    Args:
        cache_dir: 缓存目录
        cache_key: 缓存键
        buckets: VPIN buckets列表，如果为None表示不保存buckets
        final_state: final_state字典，必须提供
        save_buckets: 是否保存buckets（用于标准缓存优化：只存final_state）

    缓存格式：
    1. 标准缓存（save_buckets=False）：只保存final_state（dict）
    2. 状态缓存（save_buckets=True）：保存(buckets, final_state) tuple
    3. 向后兼容：如果final_state为空，保存为旧格式（只有buckets）
    """
    cache_file = cache_dir / f"{cache_key}.pkl"
    try:
        import pickle

        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            # 如果final_state为空，保存为旧格式（只有buckets）以保持向后兼容
            if final_state is None or all(
                v < 1e-6 for v in final_state.values() if isinstance(v, (int, float))
            ):
                if buckets is not None:
                    pickle.dump(buckets, f)
                else:
                    print(
                        f"      ⚠️  Cannot save cache: both buckets and final_state are None",
                        flush=True,
                    )
            elif not save_buckets:
                # 优化格式：标准缓存只保存final_state
                pickle.dump(final_state, f)
            else:
                # 标准格式：保存(buckets, final_state) tuple
                if buckets is not None:
                    pickle.dump((buckets, final_state), f)
                else:
                    # 如果没有buckets，只保存final_state
                    pickle.dump(final_state, f)
    except Exception as e:
        print(f"      ⚠️  Failed to save cache {cache_key}: {e}", flush=True)


def _compute_vpin_buckets_for_month(
    path: Path,
    bucket_volume: float,
    bucket_volume_usd: Optional[float] = None,
    initial_state: Optional[dict] = None,
) -> tuple[List[Tuple[pd.Timestamp, float]], dict]:
    """
    计算单月的VPIN buckets（完整月份，不裁剪）

    关键：支持从初始状态继续计算，确保跨月 bucket 连续性

    VPIN (Volume-Weighted Price Imbalance) 计算原理：
    1. 按交易量（或 USD 价值）顺序填充 bucket
    2. 每个 bucket 填满后，计算买卖不平衡度：|buy_volume - sell_volume| / bucket_volume
    3. 支持跨 tick 拆分（一个 tick 可能填充多个 bucket）
    4. 使用容差 BUCKET_COMPLETION_TOLERANCE = 1e-6 避免浮点误差

    Args:
        path (str | Path): Tick 文件路径（Parquet 格式）
        bucket_volume (float): Bucket volume (数量，如果 bucket_volume_usd 为 None 时使用)
            例如：bucket_volume=1000 表示每个 bucket 需要 1000 个币的交易量
        bucket_volume_usd (Optional[float]): Bucket volume in USD (如果提供，使用 USD 价值计算)
            例如：bucket_volume_usd=100000 表示每个 bucket 需要 10 万美元的交易量
            注意：bucket_volume 和 bucket_volume_usd 二选一，优先使用 bucket_volume_usd
        initial_state (Optional[dict]): 初始 bucket 状态（用于跨月连续性），格式：
            {
                "current_buy": float,      # 当前 bucket 的买入量
                "current_sell": float,     # 当前 bucket 的卖出量
                "filled_value": float      # 当前 bucket 已填充的总量（current_buy + current_sell）
            }
            如果为 None，从空 bucket 开始计算

    Returns:
        tuple[List[Tuple[pd.Timestamp, float]], dict]:
            - buckets: List of (timestamp, vpin_value) tuples
                - timestamp: 该 bucket 完成的时间戳
                - vpin_value: VPIN 值（0.0 到 1.0 之间，表示买卖不平衡度）
            - final_state: 未完成的 bucket 状态（用于下一批次），格式：
                {
                    "current_buy": float,
                    "current_sell": float,
                    "filled_value": float
                }

    Example:
        >>> buckets, final_state = _compute_vpin_buckets_for_month(
        ...     "BTCUSDT_2024-01.parquet",
        ...     bucket_volume=1000.0,
        ...     initial_state={"current_buy": 200.0, "current_sell": 100.0, "filled_value": 300.0}
        ... )
        >>> len(buckets)  # 该月完成的 bucket 数量
        150
        >>> final_state["filled_value"]  # 未完成的 bucket 已填充量
        450.0
    """
    buckets = []

    # 从初始状态开始（如果有），否则从0开始
    if initial_state is not None:
        current_buy = initial_state.get("current_buy", 0.0)
        current_sell = initial_state.get("current_sell", 0.0)
        filled_value = initial_state.get("filled_value", 0.0)
    else:
        current_buy = 0.0
        current_sell = 0.0
        filled_value = 0.0

    # 如果使用 USD bucket_volume，需要读取 price 列
    if bucket_volume_usd is not None:
        required_cols = ["timestamp", "volume", "side", "price"]
    else:
        required_cols = ["timestamp", "volume", "side"]

    df = pd.read_parquet(path, columns=required_cols)
    if df.empty:
        # 返回空 buckets 和当前状态（可能来自 initial_state）
        final_state = {
            "current_buy": current_buy,
            "current_sell": current_sell,
            "filled_value": filled_value,
        }
        return buckets, final_state

    df = df.dropna(subset=required_cols)
    timestamps = pd.to_datetime(df["timestamp"]).to_numpy()
    volumes = df["volume"].astype(float).to_numpy()
    sides = df["side"].astype(int).to_numpy()

    # 如果使用 USD bucket_volume，计算每个 tick 的 USD 价值
    if bucket_volume_usd is not None:
        prices = df["price"].astype(float).to_numpy()
        values_usd = prices * volumes  # 每个 tick 的 USD 价值
        target_bucket = bucket_volume_usd
    else:
        values_usd = None
        target_bucket = bucket_volume

    for i, (ts, vol, side) in enumerate(zip(timestamps, volumes, sides)):

        # 确定当前 tick 的价值（USD 或数量）
        if bucket_volume_usd is not None:
            tick_value = values_usd[i]
        else:
            tick_value = vol

        remaining = tick_value
        while remaining > 0:
            space_left = target_bucket - filled_value
            trade_value = min(remaining, space_left)
            if side == 1:
                current_buy += trade_value
            else:
                current_sell += trade_value
            filled_value += trade_value
            remaining -= trade_value

            # 使用容差判断 bucket 完成，避免浮点误差（USD 模式下尤其重要）
            BUCKET_COMPLETION_TOLERANCE = 1e-6  # 可配置的容差，适配不同币种量级
            if filled_value >= target_bucket - BUCKET_COMPLETION_TOLERANCE:
                imbalance = abs(current_buy - current_sell)
                buckets.append((pd.Timestamp(ts), imbalance / target_bucket))
                current_buy = 0.0
                current_sell = 0.0
                filled_value = 0.0

    # 返回 buckets 和未完成的 bucket 状态（用于跨月连续性）
    final_state = {
        "current_buy": current_buy,
        "current_sell": current_sell,
        "filled_value": filled_value,
    }
    return buckets, final_state


def compute_vpin_from_cached_ticks(
    cache_files: List[str],
    start_ts: str,
    end_ts: str,
    bucket_volume: Optional[float],
    n_buckets: int,
    adaptive: bool,
    quantile: float = 0.3,
    lookback_days: int = 7,
    lookback_minutes: int = 60,
    monthly_cache_dir: Optional[str] = "cache/features/monthly",
    bucket_volume_usd: Optional[float] = None,
    max_preload_months: int = 6,
) -> pd.Series:
    """
    从tick文件计算VPIN，支持按月缓存

    Args:
        cache_files: Tick parquet文件列表
        start_ts: 开始时间
        end_ts: 结束时间
        bucket_volume: Bucket volume（数量，如果为None且bucket_volume_usd为None则自适应计算）
        n_buckets: 滚动窗口大小
        adaptive: 是否自适应bucket volume（仅在bucket_volume_usd为None时有效）
        quantile: 自适应计算时的分位数
        lookback_days: 自适应计算时的回看天数
        lookback_minutes: 时间范围扩展的分钟数
        monthly_cache_dir: 按月缓存目录（如果为None则禁用缓存）
        bucket_volume_usd: Bucket volume in USD（如果提供，使用USD价值计算，所有品种使用相同值）
    """
    if not cache_files:
        raise ValueError("No cached tick files provided for VPIN computation.")

    # Normalize to tz-naive timestamps (UTC) for robust comparisons.
    # Upstream callers may pass tz-aware timestamps (e.g., UTC-indexed feature frames),
    # while tick parquet timestamps are tz-naive. Mixing them causes:
    # "Cannot compare tz-naive and tz-aware timestamps".
    start = pd.to_datetime(start_ts, utc=True).tz_convert(None) - pd.Timedelta(
        minutes=lookback_minutes
    )
    end = pd.to_datetime(end_ts, utc=True).tz_convert(None) + pd.Timedelta(
        minutes=lookback_minutes
    )

    # 优化：根据 start/end 时间过滤 cache_files，只加载需要的月份
    # 文件名格式：{symbol}_{year}-{month:02d}.parquet
    def _get_file_month(path_str: str) -> Optional[pd.Timestamp]:
        """从文件名提取月份时间戳"""
        try:
            path = Path(path_str)
            # 假设文件名格式：BTCUSDT_2025-01.parquet
            name = path.stem  # 去掉 .parquet
            parts = name.split("_")
            if len(parts) >= 2:
                date_part = parts[-1]  # 2025-01
                year, month = map(int, date_part.split("-"))
                return pd.Timestamp(year=year, month=month, day=1)
        except (ValueError, IndexError):
            pass
        return None

    # 优化：只加载当前月份和前一个月的数据（用于滚动平均）
    # 策略：
    # 1. 确定需要计算的时间范围涉及的月份（start 到 end）
    # 2. 只加载这些月份 + 前一个月（用于滚动平均的 n_buckets 前置数据）
    # 3. 最多只加载两个月的数据，大大减少内存占用
    #
    # 例如：
    # - 计算7月的VPIN：只需要7月和6月的数据
    # - 计算7月5日的VPIN：只需要7月5日之前（7月部分）和6月的数据

    # 确定需要计算的月份范围
    start_month = pd.Timestamp(year=start.year, month=start.month, day=1)
    end_month = pd.Timestamp(year=end.year, month=end.month, day=1)

    # 需要加载的月份：当前月份范围 + 前一个月（用于滚动平均）
    required_months = set()

    # 添加所有涉及的月份
    current = start_month
    while current <= end_month:
        required_months.add((current.year, current.month))
        current = current + pd.offsets.MonthBegin(1)

    # 添加前一个月（用于滚动平均的前置数据）
    prev_month = start_month - pd.offsets.MonthBegin(1)
    required_months.add((prev_month.year, prev_month.month))

    # 过滤：只保留需要的月份文件
    def _file_month_matches(path_str: str) -> bool:
        """检查文件月份是否在需要的月份列表中"""
        file_month = _get_file_month(path_str)
        if file_month is None:
            # 无法解析文件名，保守处理：包含该文件
            return True
        return (file_month.year, file_month.month) in required_months

    filtered_cache_files = [f for f in cache_files if _file_month_matches(f)]

    if len(filtered_cache_files) < len(cache_files):
        month_str = ", ".join([f"{y}-{m:02d}" for y, m in sorted(required_months)])
        print(
            f"   📅 Filtered cache files: {len(cache_files)} -> {len(filtered_cache_files)} "
            f"(only loading months: {month_str})",
            flush=True,
        )
    elif len(filtered_cache_files) > 0:
        month_str = ", ".join([f"{y}-{m:02d}" for y, m in sorted(required_months)])
        print(
            f"   📅 Loading {len(filtered_cache_files)} month(s): {month_str} "
            f"(time range: {start.date()} to {end.date()})",
            flush=True,
        )

    cache_files = sorted(filtered_cache_files)

    # 如果提供了 bucket_volume_usd，优先使用 USD 模式
    if bucket_volume_usd is not None:
        # USD 模式：所有品种使用相同的 USD bucket_volume
        if bucket_volume_usd <= 0:
            raise ValueError("bucket_volume_usd must be positive")
        print(
            f"   💰 Using USD bucket_volume mode: ${bucket_volume_usd:,.0f} USD per bucket"
        )
    else:
        # 传统模式：按数量计算
        if bucket_volume is None:
            if adaptive:
                bucket_volume = _estimate_bucket_volume_from_cache(
                    cache_files, lookback_days=lookback_days, quantile=quantile
                )
            else:
                bucket_volume = 100.0

    # 按月缓存目录
    cache_dir = Path(monthly_cache_dir) if monthly_cache_dir else None

    # 流式处理：按月分批计算，每次只加载两个月的数据（当前月+前一个月）
    # 策略：
    # 1. 确定需要计算的月份范围（start 到 end）
    # 2. 按月循环处理：
    #    - 计算5月：加载4月和5月的数据
    #    - 计算6月：加载5月和6月的数据（5月可能已经缓存）
    #    - 计算7月：加载6月和7月的数据（6月可能已经缓存）
    # 3. 每次只加载两个月的数据到内存，大大减少内存占用

    # 确定需要计算的月份范围
    start_month = pd.Timestamp(year=start.year, month=start.month, day=1)
    end_month = pd.Timestamp(year=end.year, month=end.month, day=1)

    # 按月分批处理
    all_vpin_series = []
    total_months = 0
    cached_count = 0
    computed_count = 0

    # 优化：维护固定长度的 buckets buffer（用于滚动平均），而不是整个前月的 buckets
    # 只保留最近 n_buckets 个 buckets（包含时间戳），节省内存且逻辑更清晰
    recent_buckets = []  # List[Tuple[Timestamp, float]]，最近 n_buckets 个 buckets
    # 维护未完成的 bucket 状态（用于跨月连续性）
    prev_bucket_state = None

    # 预加载阶段：确保首次计算时有足够的历史 bucket（≥ n_buckets）
    # 如果第一个月的数据不足以提供 n_buckets 个 bucket，需要向前多加载几个月
    if len(recent_buckets) < n_buckets:
        # 向前查找前几个月，直到收集到足够的 buckets
        preload_month = start_month - pd.offsets.MonthBegin(1)
        preload_attempts = 0
        max_preload_attempts = max(
            0, int(max_preload_months)
        )  # 最多向前查找 N 个月（可配置）

        while (
            len(recent_buckets) < n_buckets and preload_attempts < max_preload_attempts
        ):
            preload_file = None
            for f in cache_files:
                file_month = _get_file_month(f)
                if (
                    file_month is not None
                    and file_month.year == preload_month.year
                    and file_month.month == preload_month.month
                ):
                    preload_file = f
                    break

            if preload_file:
                print(
                    f"   📥 Preloading {preload_month.year}-{preload_month.month:02d} for rolling window initialization",
                    flush=True,
                )
                if cache_dir:
                    preload_cache_key = _get_monthly_vpin_cache_key(
                        preload_file, bucket_volume, bucket_volume_usd
                    )
                    cached_result = _load_monthly_vpin_cache(
                        cache_dir, preload_cache_key
                    )
                    if cached_result is not None:
                        cached_preload_buckets, _ = cached_result
                        if cached_preload_buckets is None:
                            # 标准缓存只存了final_state，需要重新计算buckets
                            preload_buckets, _ = _compute_vpin_buckets_for_month(
                                preload_file,
                                bucket_volume,
                                bucket_volume_usd,
                                initial_state=None,
                            )
                            cached_preload_buckets = preload_buckets
                            computed_count += 1
                        # 将预加载的 buckets 添加到 recent_buckets 前面（时间顺序）
                        recent_buckets = cached_preload_buckets + recent_buckets
                        cached_count += 1
                        print(
                            f"      -> {Path(preload_file).name} (cached, {len(cached_preload_buckets)} buckets)",
                            flush=True,
                        )
                    else:
                        preload_buckets, preload_final_state = (
                            _compute_vpin_buckets_for_month(
                                preload_file,
                                bucket_volume,
                                bucket_volume_usd,
                                initial_state=None,
                            )
                        )
                        # 将预加载的 buckets 添加到 recent_buckets 前面（时间顺序）
                        recent_buckets = preload_buckets + recent_buckets
                        computed_count += 1
                        # 预加载阶段保存标准缓存：只保存final_state（优化：不保存buckets）
                        _save_monthly_vpin_cache(
                            cache_dir,
                            preload_cache_key,
                            None,
                            preload_final_state,
                            save_buckets=False,
                        )
                        print(
                            f"      -> {Path(preload_file).name} (computed, {len(preload_buckets)} buckets)",
                            flush=True,
                        )
                else:
                    preload_buckets, preload_final_state = (
                        _compute_vpin_buckets_for_month(
                            preload_file,
                            bucket_volume,
                            bucket_volume_usd,
                            initial_state=None,
                        )
                    )
                    # 将预加载的 buckets 添加到 recent_buckets 前面（时间顺序）
                    recent_buckets = preload_buckets + recent_buckets
                    computed_count += 1
                    print(
                        f"      -> {Path(preload_file).name} (computed, no cache, {len(preload_buckets)} buckets)",
                        flush=True,
                    )

                # 只保留最近 n_buckets 个（如果已经足够）
                if len(recent_buckets) > n_buckets:
                    recent_buckets = recent_buckets[-n_buckets:]

            preload_month = preload_month - pd.offsets.MonthBegin(1)
            preload_attempts += 1

        if len(recent_buckets) < n_buckets:
            if len(recent_buckets) == 0:
                print(
                    f"   ⚠️ Warning: No historical buckets found for rolling window initialization (need {n_buckets})",
                    flush=True,
                )
                print(
                    f"      This may happen if: (1) no historical data files exist, (2) data files are not in expected location, or (3) date range doesn't match",
                    flush=True,
                )
                print(
                    f"      Rolling window will start with available data (may have reduced accuracy initially)",
                    flush=True,
                )
            else:
                print(
                    f"   ⚠️ Warning: Only collected {len(recent_buckets)} buckets for rolling window (need {n_buckets})",
                    flush=True,
                )
                print(
                    f"      Rolling window will start with {len(recent_buckets)} buckets (may have reduced accuracy initially)",
                    flush=True,
                )
        else:
            print(
                f"   ✅ Preloaded {len(recent_buckets)} buckets for rolling window initialization",
                flush=True,
            )

    current = start_month
    while current <= end_month:
        total_months += 1
        year, month = current.year, current.month

        # 当前月
        current_month_file = None
        for f in cache_files:
            file_month = _get_file_month(f)
            if (
                file_month is not None
                and file_month.year == year
                and file_month.month == month
            ):
                current_month_file = f
                break

        if current_month_file is None:
            # 跳过不存在的月份
            current = current + pd.offsets.MonthBegin(1)
            continue

        # 前一个月（用于滚动平均）
        prev_month = current - pd.offsets.MonthBegin(1)
        prev_month_file = None
        for f in cache_files:
            file_month = _get_file_month(f)
            if (
                file_month is not None
                and file_month.year == prev_month.year
                and file_month.month == prev_month.month
            ):
                prev_month_file = f
                break

        print(
            f"   📅 Processing {year}-{month:02d} (loading {prev_month.year}-{prev_month.month:02d} + {year}-{month:02d})",
            flush=True,
        )

        # 关键修复：如果 prev_bucket_state 为 None，应该尝试获取前一个月的 final_state
        # 原因：即使当前月是循环的第一个月（如计算3~6月，3月是第一个），如果前面有数据（2月），
        # 也不应该用 prev=None，而应该获取前一个月的 final_state 作为 prev
        # 只有当前面没有数据了（既没有数据文件，也没有缓存），才应该用 prev=None
        if prev_bucket_state is None:
            # 尝试获取前一个月的 final_state
            if prev_month_file is not None:
                # 前一个月的数据文件存在，从文件计算或从缓存加载
                if cache_dir is not None:
                    prev_month_cache_key = _get_monthly_vpin_cache_key(
                        prev_month_file,
                        bucket_volume,
                        bucket_volume_usd,
                        prev_bucket_state=None,
                    )
                    prev_month_cached_result = _load_monthly_vpin_cache(
                        cache_dir, prev_month_cache_key
                    )
                    if prev_month_cached_result is not None:
                        _, prev_month_final_state = prev_month_cached_result
                        if prev_month_final_state is not None:
                            prev_bucket_state = prev_month_final_state
                            print(
                                f"      📥 Loaded prev_month final_state: filled_value = {prev_bucket_state.get('filled_value', 0.0):.6f}",
                                flush=True,
                            )
                    else:
                        # 前一个月的标准缓存不存在，需要临时计算以获取 final_state
                        print(
                            f"      📥 Computing prev_month to get final_state...",
                            flush=True,
                        )
                        _, prev_month_final_state = _compute_vpin_buckets_for_month(
                            Path(prev_month_file),
                            bucket_volume,
                            bucket_volume_usd,
                            initial_state=None,
                        )
                        prev_bucket_state = prev_month_final_state
                        # 缓存前一个月的 final_state（标准缓存）
                        _save_monthly_vpin_cache(
                            cache_dir,
                            prev_month_cache_key,
                            None,
                            prev_month_final_state,
                            save_buckets=False,
                        )
                        print(
                            f"      📥 Computed prev_month final_state: filled_value = {prev_bucket_state.get('filled_value', 0.0):.6f}",
                            flush=True,
                        )
                else:
                    # 没有缓存目录，临时计算前一个月的 final_state
                    print(
                        f"      📥 Computing prev_month to get final_state (no cache)...",
                        flush=True,
                    )
                    _, prev_month_final_state = _compute_vpin_buckets_for_month(
                        Path(prev_month_file),
                        bucket_volume,
                        bucket_volume_usd,
                        initial_state=None,
                    )
                    prev_bucket_state = prev_month_final_state
            elif cache_dir is not None:
                # 前一个月的数据文件不存在，但尝试从缓存中查找（可能之前计算过）
                # 从当前月的文件名推断前一个月的文件名格式
                # 例如：BTCUSDT_2024-01.parquet -> BTCUSDT_2023-12.parquet
                try:
                    current_file_path = Path(current_month_file)
                    # 尝试从文件名中提取符号和日期
                    # 假设格式为：SYMBOL_YYYY-MM.parquet 或 SYMBOL_YYYYMM.parquet
                    file_stem = current_file_path.stem  # 不含扩展名
                    parts = file_stem.split("_")
                    if len(parts) >= 2:
                        symbol = "_".join(parts[:-1])  # 符号部分
                        date_str = parts[-1]  # 日期部分

                        # 尝试解析日期
                        if len(date_str) == 7 and date_str[4] == "-":  # YYYY-MM
                            year, month = int(date_str[:4]), int(date_str[5:7])
                        elif len(date_str) == 6:  # YYYYMM
                            year, month = int(date_str[:4]), int(date_str[4:6])
                        else:
                            # 无法解析，跳过
                            year, month = None, None

                        if year is not None and month is not None:
                            # 计算前一个月
                            if month == 1:
                                prev_year, prev_month = year - 1, 12
                            else:
                                prev_year, prev_month = year, month - 1

                            # 构建前一个月的文件名（尝试两种格式）
                            prev_month_file_candidates = [
                                f"{symbol}_{prev_year}-{prev_month:02d}.parquet",
                                f"{symbol}_{prev_year}{prev_month:02d}.parquet",
                            ]

                            # 尝试从缓存中查找前一个月的 final_state
                            for prev_file_candidate in prev_month_file_candidates:
                                prev_month_cache_key = _get_monthly_vpin_cache_key(
                                    prev_file_candidate,
                                    bucket_volume,
                                    bucket_volume_usd,
                                    prev_bucket_state=None,
                                )
                                prev_month_cached_result = _load_monthly_vpin_cache(
                                    cache_dir, prev_month_cache_key
                                )
                                if prev_month_cached_result is not None:
                                    _, prev_month_final_state = prev_month_cached_result
                                    if prev_month_final_state is not None:
                                        prev_bucket_state = prev_month_final_state
                                        print(
                                            f"      📥 Loaded prev_month final_state from cache (inferred file: {prev_file_candidate}): filled_value = {prev_bucket_state.get('filled_value', 0.0):.6f}",
                                            flush=True,
                                        )
                                        break
                except Exception as e:
                    # 无法推断文件名格式，跳过
                    pass

        # 加载当前月的数据（使用 prev_bucket_state 确保跨月连续性）
        current_buckets = []
        current_final_state = None

        # 优化缓存策略：
        # 1. 每个月的 final_state 是固定的（只取决于该月数据），不依赖于前一个月的状态
        # 2. 前一个月的状态只影响当前月的第一个 bucket，但不影响 final_state
        # 3. 因此，我们可以：
        #    - 总是缓存每个月的标准结果（从空状态开始），包含 buckets 和 final_state
        #    - 如果 prev_bucket_state 不为空，先尝试使用状态缓存（如果存在），否则重新计算并缓存
        #    - 状态缓存的 key 包含 prev_bucket_state 信息，以便下次直接使用

        # 生成缓存键
        standard_cache_key = None
        state_cache_key = None
        if cache_dir is not None:
            standard_cache_key = _get_monthly_vpin_cache_key(
                current_month_file,
                bucket_volume,
                bucket_volume_usd,
                prev_bucket_state=None,
            )
            if prev_bucket_state is not None:
                state_cache_key = _get_monthly_vpin_cache_key(
                    current_month_file,
                    bucket_volume,
                    bucket_volume_usd,
                    prev_bucket_state,
                )

        # 尝试从缓存加载
        cached_result = None
        cache_key_used = None

        if prev_bucket_state is not None and state_cache_key is not None:
            # 如果 prev_bucket_state 不为空，先尝试使用状态缓存
            cached_result = _load_monthly_vpin_cache(cache_dir, state_cache_key)
            if cached_result is not None:
                cache_key_used = state_cache_key

        if cached_result is None and standard_cache_key is not None:
            # 如果状态缓存未命中，尝试使用标准缓存
            cached_result = _load_monthly_vpin_cache(cache_dir, standard_cache_key)
            if cached_result is not None:
                cache_key_used = standard_cache_key

        if cached_result is not None:
            # 缓存命中
            cached_buckets, cached_final_state = cached_result
            cached_count += 1

            # 明确处理标准缓存只存 final_state 的情况（buckets=None）
            if cached_buckets is None:
                # 标准缓存只存了 final_state，需要重新计算 buckets
                if prev_bucket_state is None:
                    # prev_bucket_state 为空，说明前面没有数据了（prev_month_file 为 None）
                    print(
                        f"      -> {Path(current_month_file).name} (computed, final_state from cache, prev=None)",
                        flush=True,
                    )
                    current_buckets, computed_final_state = (
                        _compute_vpin_buckets_for_month(
                            current_month_file,
                            bucket_volume,
                            bucket_volume_usd,
                            initial_state=None,
                        )
                    )
                    computed_count += 1
                    # 验证 final_state 是否一致（应该一致，因为只取决于该月数据）
                    if (
                        abs(
                            computed_final_state.get("filled_value", 0.0)
                            - cached_final_state.get("filled_value", 0.0)
                        )
                        > 1e-6
                    ):
                        print(
                            f"      ⚠️  Warning: final_state mismatch (computed: {computed_final_state.get('filled_value', 0.0):.6f}, cached: {cached_final_state.get('filled_value', 0.0):.6f})",
                            flush=True,
                        )
                        current_final_state = computed_final_state
                    else:
                        current_final_state = cached_final_state
                else:
                    # prev_bucket_state 不为空，使用它重新计算 buckets
                    filled_pct = (
                        (
                            prev_bucket_state.get("filled_value", 0.0)
                            / bucket_volume
                            * 100
                        )
                        if bucket_volume > 0
                        else 0.0
                    )
                    print(
                        f"      -> {Path(current_month_file).name} (computed, prev state {filled_pct:.1f}% filled, final_state from cache)",
                        flush=True,
                    )
                    current_buckets, computed_final_state = (
                        _compute_vpin_buckets_for_month(
                            current_month_file,
                            bucket_volume,
                            bucket_volume_usd,
                            initial_state=prev_bucket_state,
                        )
                    )
                    computed_count += 1

                    # 验证 final_state 是否一致（应该一致，因为只取决于该月数据）
                    if (
                        abs(
                            computed_final_state.get("filled_value", 0.0)
                            - cached_final_state.get("filled_value", 0.0)
                        )
                        > 1e-6
                    ):
                        print(
                            f"      ⚠️  Warning: final_state mismatch (computed: {computed_final_state.get('filled_value', 0.0):.6f}, cached: {cached_final_state.get('filled_value', 0.0):.6f})",
                            flush=True,
                        )
                        current_final_state = computed_final_state
                    else:
                        current_final_state = cached_final_state

                    # 缓存结果（使用包含状态的缓存键，以便下次直接使用）
                    if (
                        state_cache_key is not None
                        and state_cache_key != standard_cache_key
                    ):  # 避免重复缓存
                        _save_monthly_vpin_cache(
                            cache_dir,
                            state_cache_key,
                            current_buckets,
                            current_final_state,
                            save_buckets=True,
                        )
            else:
                # 缓存包含完整的 buckets，可以直接使用
                current_buckets = cached_buckets
                current_final_state = cached_final_state

                if prev_bucket_state is None:
                    # prev_bucket_state 为空，但缓存有 buckets，可以直接使用
                    print(
                        f"      -> {Path(current_month_file).name} (cached, prev=None)",
                        flush=True,
                    )
                else:
                    # prev_bucket_state 不为空
                    filled_pct = (
                        (
                            prev_bucket_state.get("filled_value", 0.0)
                            / bucket_volume
                            * 100
                        )
                        if bucket_volume > 0
                        else 0.0
                    )
                    if cache_key_used == state_cache_key:
                        # 状态缓存命中，直接使用
                        print(
                            f"      -> {Path(current_month_file).name} (cached, with prev state {filled_pct:.1f}% filled)",
                            flush=True,
                        )
                    else:
                        # 使用了标准缓存的 buckets，但 prev_bucket_state 不为空
                        # 注意：标准缓存的 buckets 是从 prev=None 计算的，可能不符合实际情况
                        # 为了确保正确性，应该重新计算
                        print(
                            f"      -> {Path(current_month_file).name} (computed, prev state {filled_pct:.1f}% filled, buckets from cache may be incorrect)",
                            flush=True,
                        )
                        current_buckets, computed_final_state = (
                            _compute_vpin_buckets_for_month(
                                current_month_file,
                                bucket_volume,
                                bucket_volume_usd,
                                initial_state=prev_bucket_state,
                            )
                        )
                        computed_count += 1

                        # 验证 final_state 是否一致（应该一致，因为只取决于该月数据）
                        if (
                            abs(
                                computed_final_state.get("filled_value", 0.0)
                                - current_final_state.get("filled_value", 0.0)
                            )
                            > 1e-6
                        ):
                            print(
                                f"      ⚠️  Warning: final_state mismatch (computed: {computed_final_state.get('filled_value', 0.0):.6f}, cached: {current_final_state.get('filled_value', 0.0):.6f})",
                                flush=True,
                            )
                            current_final_state = computed_final_state

                        # 缓存结果（使用包含状态的缓存键，以便下次直接使用）
                        if (
                            state_cache_key is not None
                            and state_cache_key != standard_cache_key
                        ):  # 避免重复缓存
                            _save_monthly_vpin_cache(
                                cache_dir,
                                state_cache_key,
                                current_buckets,
                                current_final_state,
                                save_buckets=True,
                            )
        else:
            # 缓存未命中，重新计算
            if prev_bucket_state is not None:
                filled_pct = (
                    (prev_bucket_state.get("filled_value", 0.0) / bucket_volume * 100)
                    if bucket_volume > 0
                    else 0.0
                )
                print(
                    f"      -> {Path(current_month_file).name} (computed, prev state {filled_pct:.1f}% filled)",
                    flush=True,
                )
            else:
                # prev_bucket_state 为 None，说明前面没有数据了（prev_month_file 为 None）
                print(
                    f"      -> {Path(current_month_file).name} (computed, prev=None)",
                    flush=True,
                )
            current_buckets, current_final_state = _compute_vpin_buckets_for_month(
                current_month_file,
                bucket_volume,
                bucket_volume_usd,
                initial_state=prev_bucket_state,
            )
            computed_count += 1
            # 缓存结果
            # 标准缓存：只保存final_state（优化：不保存buckets，节省存储空间）
            if standard_cache_key is not None:
                _save_monthly_vpin_cache(
                    cache_dir,
                    standard_cache_key,
                    None,
                    current_final_state,
                    save_buckets=False,
                )
            # 状态缓存：保存buckets和final_state（用于加速带上下文的查询）
            if state_cache_key is not None and state_cache_key != standard_cache_key:
                _save_monthly_vpin_cache(
                    cache_dir,
                    state_cache_key,
                    current_buckets,
                    current_final_state,
                    save_buckets=True,
                )

        # 优化：加载前一个月的数据用于滚动平均（如果需要）
        # 策略：
        # 1. 如果是第一个月且 recent_buckets 不足 n_buckets，需要加载前一个月
        # 2. 否则，recent_buckets 已经包含足够的历史数据
        need_prev_for_rolling = (
            (len(recent_buckets) < n_buckets)
            and (prev_month_file is not None)
            and (prev_month_file != current_month_file)
        )

        if need_prev_for_rolling:
            # 需要加载前一个月的数据来初始化滚动窗口
            # 检查前一个月的数据是否已经加载
            need_load_prev = True
            if recent_buckets:
                first_bucket = recent_buckets[0]  # (timestamp, vpin) tuple
                first_bucket_time = first_bucket[0]  # timestamp
                if isinstance(first_bucket_time, pd.Timestamp):
                    if (
                        first_bucket_time.year == prev_month.year
                        and first_bucket_time.month == prev_month.month
                    ):
                        need_load_prev = False

            if need_load_prev:
                # 前一个月的数据还没有加载，现在加载（完整月份，不需要 prev_bucket_state）
                if cache_dir:
                    prev_cache_key = _get_monthly_vpin_cache_key(
                        prev_month_file,
                        bucket_volume,
                        bucket_volume_usd,
                        prev_bucket_state=None,
                    )
                    cached_prev_result = _load_monthly_vpin_cache(
                        cache_dir, prev_cache_key
                    )
                    if cached_prev_result is not None:
                        cached_prev_buckets, _ = cached_prev_result
                        if cached_prev_buckets is None:
                            # 标准缓存只存了final_state，需要重新计算buckets
                            prev_buckets, _ = _compute_vpin_buckets_for_month(
                                prev_month_file,
                                bucket_volume,
                                bucket_volume_usd,
                                initial_state=None,
                            )
                            cached_prev_buckets = prev_buckets
                            computed_count += 1
                        # 只保留最近 n_buckets 个 buckets
                        recent_buckets = cached_prev_buckets[-n_buckets:]
                        cached_count += 1
                        print(
                            f"      -> {Path(prev_month_file).name} (cached, kept last {len(recent_buckets)} buckets)",
                            flush=True,
                        )
                    else:
                        prev_buckets, prev_final_state = (
                            _compute_vpin_buckets_for_month(
                                prev_month_file,
                                bucket_volume,
                                bucket_volume_usd,
                                initial_state=None,
                            )
                        )
                        # 只保留最近 n_buckets 个 buckets
                        recent_buckets = prev_buckets[-n_buckets:]
                        computed_count += 1
                        # 前一个月数据保存标准缓存：只保存final_state（优化：不保存buckets）
                        _save_monthly_vpin_cache(
                            cache_dir,
                            prev_cache_key,
                            None,
                            prev_final_state,
                            save_buckets=False,
                        )
                        print(
                            f"      -> {Path(prev_month_file).name} (computed, kept last {len(recent_buckets)} buckets)",
                            flush=True,
                        )
                else:
                    prev_buckets, _ = _compute_vpin_buckets_for_month(
                        prev_month_file,
                        bucket_volume,
                        bucket_volume_usd,
                        initial_state=None,
                    )
                    # 只保留最近 n_buckets 个 buckets
                    recent_buckets = prev_buckets[-n_buckets:]
                    computed_count += 1
                    print(
                        f"      -> {Path(prev_month_file).name} (computed, no cache, kept last {len(recent_buckets)} buckets)",
                        flush=True,
                    )

        # 合并 recent_buckets 和当前月的 buckets（用于滚动平均）
        month_buckets = recent_buckets + current_buckets

        if not month_buckets:
            current = current + pd.offsets.MonthBegin(1)
            continue

        # 转换为 DataFrame 并计算滚动平均
        buckets_df = (
            pd.DataFrame(month_buckets, columns=["timestamp", "vpin"])
            .set_index("timestamp")
            .sort_index()
        )

        # 计算滚动平均（使用前一个月+当前月的数据）
        vpin_series_full = (
            buckets_df["vpin"].rolling(window=n_buckets, min_periods=1).mean()
        )

        # 只保留当前月的数据（裁剪到当前月的范围）
        month_start = current
        month_end = current + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)
        # 与实际的 start/end 取交集
        month_start = max(month_start, start)
        month_end = min(month_end, end + pd.Timedelta(days=1))

        month_vpin = vpin_series_full.loc[month_start:month_end]
        if len(month_vpin) > 0:
            all_vpin_series.append(month_vpin)

        # 更新状态，供下一轮使用
        # 1. 更新 recent_buckets：只保留最近 n_buckets 个 buckets（用于滚动平均）
        # 将当前月的 buckets 添加到 recent_buckets，然后只保留最后 n_buckets 个
        recent_buckets = (recent_buckets + current_buckets)[-n_buckets:]
        # 2. 更新未完成的 bucket 状态（用于跨月连续性）
        prev_bucket_state = current_final_state

        # 移动到下一个月
        current = current + pd.offsets.MonthBegin(1)

    if not all_vpin_series:
        raise ValueError(
            "VPIN computation produced no buckets; check tick data source."
        )

    # 合并所有月份的 VPIN 序列
    vpin_series = pd.concat(all_vpin_series).sort_index()

    # 最终裁剪到需要的范围
    vpin_series = vpin_series.loc[start:end]

    print(
        f"   ✅ VPIN computed with {total_months} month(s) processed (bucket_volume={bucket_volume:.2f})",
        flush=True,
    )
    if cache_dir and cached_count > 0:
        print(
            f"   💾 Used {cached_count} cached months, computed {computed_count} new months",
            flush=True,
        )

    return vpin_series.sort_index()


# ============================================================================
# Trade Clustering 缓存函数
# ============================================================================


def _get_monthly_trade_clustering_cache_key(
    file_path: str,
    window_size: int,
    initial_state: Optional[Dict[str, Any]] = None,
) -> str:
    """
    生成按月 Trade Clustering 缓存的键（不包含 start/end，按月完整计算）

    优化：按月完整计算并缓存，不同时间窗口可以复用同一月份的缓存

    如果 initial_state 不为空，将其信息加入缓存键，以支持跨月连续性的缓存
    """
    import hashlib
    import json

    path = Path(file_path)
    month_str = path.stem.split("_")[-1] if "_" in path.stem else path.stem
    key_str = f"trade_clustering_monthly_{month_str}_{window_size}"

    # 如果 initial_state 不为空，将其信息加入缓存键
    if initial_state is not None:
        # 将 initial_state 序列化为字符串（使用固定精度避免浮点误差）
        # 注意：需要将 numpy 类型转换为 Python 原生类型
        current_run_side = initial_state.get("current_run_side")
        if current_run_side is not None:
            # 转换为 Python int（如果是 numpy int64）
            if hasattr(current_run_side, "item"):
                current_run_side = current_run_side.item()
            else:
                current_run_side = int(current_run_side)

        state_str = json.dumps(
            {
                "current_run_side": current_run_side,
                "current_run_length": round(
                    float(initial_state.get("current_run_length", 0)), 6
                ),
                "window_total_ticks": round(
                    float(initial_state.get("window_total_ticks", 0)), 6
                ),
                # 注意：window_runs, buy_runs_in_window, sell_runs_in_window 是 deque，
                # 序列化时会被转换为 list，这里只保存关键信息
                "window_runs_count": int(len(initial_state.get("window_runs", []))),
                "buy_runs_count": int(len(initial_state.get("buy_runs_in_window", []))),
                "sell_runs_count": int(
                    len(initial_state.get("sell_runs_in_window", []))
                ),
            },
            sort_keys=True,
        )
        key_str = f"{key_str}_state_{state_str}"

    return hashlib.md5(key_str.encode()).hexdigest()


def _load_monthly_trade_clustering_cache(
    cache_dir: Path, cache_key: str
) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    从缓存加载单月的 Trade Clustering 结果和状态

    返回: (DataFrame, state) 或 None
    支持多种缓存格式：
    1. 标准缓存：只保存 state（DataFrame 为 None）
    2. 状态缓存：保存完整结果（DataFrame + state）
    """
    from collections import deque

    cache_file = cache_dir / f"{cache_key}.pkl"
    if cache_file.exists():
        try:
            import pickle

            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)

                # 处理不同的缓存格式
                if isinstance(cached_data, tuple):
                    cluster_df, state = cached_data
                    # 如果 state 中的 deque 被序列化为 list，转换回 deque
                    if state is not None:
                        if "window_runs" in state and isinstance(
                            state["window_runs"], list
                        ):
                            state["window_runs"] = deque(state["window_runs"])
                        if "buy_runs_in_window" in state and isinstance(
                            state["buy_runs_in_window"], list
                        ):
                            state["buy_runs_in_window"] = deque(
                                state["buy_runs_in_window"]
                            )
                        if "sell_runs_in_window" in state and isinstance(
                            state["sell_runs_in_window"], list
                        ):
                            state["sell_runs_in_window"] = deque(
                                state["sell_runs_in_window"]
                            )
                    return (cluster_df, state)
                elif isinstance(cached_data, dict):
                    # 旧格式：只有 state（标准缓存）
                    # 如果 state 中的 deque 被序列化为 list，转换回 deque
                    if "window_runs" in cached_data and isinstance(
                        cached_data["window_runs"], list
                    ):
                        cached_data["window_runs"] = deque(cached_data["window_runs"])
                    if "buy_runs_in_window" in cached_data and isinstance(
                        cached_data["buy_runs_in_window"], list
                    ):
                        cached_data["buy_runs_in_window"] = deque(
                            cached_data["buy_runs_in_window"]
                        )
                    if "sell_runs_in_window" in cached_data and isinstance(
                        cached_data["sell_runs_in_window"], list
                    ):
                        cached_data["sell_runs_in_window"] = deque(
                            cached_data["sell_runs_in_window"]
                        )
                    return (None, cached_data)
                else:
                    # 未知格式，返回 None
                    return None
        except Exception as e:
            print(
                f"      ⚠️  Failed to load trade clustering cache {cache_key}: {e}",
                flush=True,
            )
    return None


def _save_monthly_trade_clustering_cache(
    cache_dir: Path,
    cache_key: str,
    result: Tuple[Optional[pd.DataFrame], Dict[str, Any]],
):
    """
    保存单月的 Trade Clustering 结果和状态到缓存

    Args:
        cache_dir: 缓存目录
        cache_key: 缓存键
        result: (DataFrame, state) 元组
            - DataFrame: Trade Clustering 结果（可以为 None，标准缓存只保存 state）
            - state: final_state 字典，必须提供

    缓存格式：
    1. 标准缓存（DataFrame 为 None）：只保存 state（dict）
    2. 状态缓存（DataFrame 不为 None）：保存完整结果（tuple）
    """
    cache_file = cache_dir / f"{cache_key}.pkl"
    try:
        import pickle

        cache_dir.mkdir(parents=True, exist_ok=True)
        cluster_df, state = result

        with open(cache_file, "wb") as f:
            if cluster_df is None:
                # 标准缓存：只保存 state（dict）
                pickle.dump(state, f)
            else:
                # 状态缓存：保存完整结果（tuple）
                pickle.dump((cluster_df, state), f)
    except Exception as e:
        print(
            f"      ⚠️  Failed to save trade clustering cache {cache_key}: {e}",
            flush=True,
        )
