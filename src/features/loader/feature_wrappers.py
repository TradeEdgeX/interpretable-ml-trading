"""
特征计算包装函数

为一些需要特殊参数处理的函数创建包装函数，使其能够通过配置文件直接调用
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from src.features.registry import register_feature

from src.features.time_series.baseline_features import (
    add_poc_hal_dimensionless_features,
    compute_atr,
    calculate_sqs,
    _get_sr_boundary_definitions,
    _compute_boundary_strengths,
)
from src.features.time_series.utils_liquidity_features import (
    extract_liquidity_features,
)
from src.features.time_series.utils_volume_profile import (
    compute_unified_volume_profile_features,
    compute_unified_volume_profile_derived_features,
)
from src.features.time_series.utils_footprint import (
    compute_kline_footprint_features,
    FootprintConfig,
)


def compute_poc_hal_features(
    df: pd.DataFrame,
    poc_window: int = 160,
    price_col: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    计算 POC (Point of Control) 和 HAL (Value Area) 特征
    
    这是一个独立的基础特征，用于计算：
    - poc: Point of Control（成交量最大的价格点）
    - hal_high: Value Area 上界（70% 成交量区间的上界）
    - hal_low: Value Area 下界（70% 成交量区间的下界）
    - hal_mid: Value Area 中点
    
    Args:
        df: DataFrame with required columns: high, low, close, volume
        poc_window: POC 计算窗口大小
        price_col: 可选的价格列名（如 'wpt_price_reconstructed'）
        **kwargs: 其他参数
    
    Returns:
        DataFrame with poc, hal_high, hal_low, hal_mid columns added
    """
    result = df.copy()
    
    # 确定使用的价格序列
    if price_col is None and "wpt_price_reconstructed" in result.columns:
        price_col = "wpt_price_reconstructed"
    
    # 计算 POC 和 HAL
    result = add_poc_hal_dimensionless_features(
        result,
        required_features={"poc", "hal_high", "hal_low", "hal_mid"},
        poc_window=poc_window,
        price_col=price_col,
    )
    
    return result


def compute_sqs_hal_high(
    df: pd.DataFrame,
    window: int = 60,
    tolerance_factor: float = 0.5,
    sr_type: str = "resistance",
    **kwargs
) -> pd.DataFrame:
    """
    计算 HAL high 的 SQS（Structure Quality Score）
    
    包装函数：
    1. 一次性计算 HAL（滚动窗口，每个时间点有一个 HAL 值）
    2. 对每个时间点，使用当前时间点的 HAL 价格作为 sr_price 计算 SQS
    
    注意：HAL 是滚动计算的，每个时间点都有一个值。SQS 使用当前时间点的 HAL 价格
    作为支撑阻力价格，评估这个价格的历史质量。
    
    Args:
        df: DataFrame with required columns: high, low, close, volume, atr
        window: SQS 计算窗口（用于评估 SR 质量的历史窗口）
        tolerance_factor: ATR 容忍带系数
        sr_type: SR 类型（'resistance' for HAL high）
        **kwargs: 其他参数（如 poc_window）
    
    Returns:
        DataFrame with 'sqs_hal_high' column added
    """
    result = df.copy()
    
    # 0. 确保 ATR 存在（SQS 计算必需）
    if "atr" not in result.columns:
        result["atr"] = compute_atr(
            result["high"],
            result["low"],
            result["close"],
            period=14
        )
    
    # 1. 检查 poc 和 hal_high 是否已存在（应该来自 poc_hal_features 依赖）
    # 如果不存在，说明依赖关系有问题，需要报错或计算
    if "poc" not in result.columns or "hal_high" not in result.columns:
        # 如果列不存在，尝试计算（向后兼容，但会打印警告）
        print(f"       ⚠️  Warning: 'poc' or 'hal_high' not found, computing them (should come from poc_hal_features dependency)")
        poc_window = kwargs.get("poc_window", 160)
        price_col = kwargs.get("price_col", None)
        if price_col is None and "wpt_price_reconstructed" in result.columns:
            price_col = "wpt_price_reconstructed"
        result = add_poc_hal_dimensionless_features(
            result,
            required_features={"hal_high", "poc"},
            poc_window=poc_window,
            price_col=price_col,
        )
    
    if "hal_high" not in result.columns:
        # 如果 HAL 计算失败，返回全 0
        result["sqs_hal_high"] = 0.0
        return result
    
    # 2. 对每个时间点计算 SQS
    # 使用当前时间点的 HAL 价格作为 sr_price（支撑阻力价格）
    # 评估这个价格在历史窗口内的质量
    sqs_values = []
    hal_high_series = result["hal_high"]
    
    for i in range(len(result)):
        if i < window:
            sqs_values.append(0.0)
            continue
        
        # 获取当前时间点的 HAL high 价格（作为支撑阻力价格）
        sr_price = hal_high_series.iloc[i]
        if pd.isna(sr_price) or sr_price <= 0:
            sqs_values.append(0.0)
            continue
        
        # 获取历史数据窗口（不含未来信息）
        # 只取窗口内的数据，用于评估这个 SR 价格的历史质量
        start_idx = max(0, i - window + 1)
        hist_df = result.iloc[start_idx:i+1].copy()
        
        # 确保有足够的列
        required_cols = ["high", "low", "close", "atr", "volume"]
        if not all(col in hist_df.columns for col in required_cols):
            sqs_values.append(0.0)
            continue
        
        # 计算 SQS：评估 sr_price 在历史窗口内的质量
        try:
            sqs = calculate_sqs(
                sr_price=sr_price,  # 使用当前时间点的 HAL 价格作为 SR 价格
                df=hist_df,  # 历史数据窗口，用于评估质量
                window=min(window, len(hist_df)),
                tolerance_factor=tolerance_factor,
                sr_type=sr_type,
            )
            sqs_values.append(float(sqs) if not np.isnan(sqs) else 0.0)
        except Exception:
            sqs_values.append(0.0)
    
    result["sqs_hal_high"] = pd.Series(sqs_values, index=result.index)
    return result


def compute_sqs_hal_low(
    df: pd.DataFrame,
    window: int = 60,
    tolerance_factor: float = 0.5,
    sr_type: str = "support",
    **kwargs
) -> pd.DataFrame:
    """
    计算 HAL low 的 SQS（Structure Quality Score）
    
    包装函数：
    1. 一次性计算 HAL（滚动窗口，每个时间点有一个 HAL 值）
    2. 对每个时间点，使用当前时间点的 HAL 价格作为 sr_price 计算 SQS
    
    注意：HAL 是滚动计算的，每个时间点都有一个值。SQS 使用当前时间点的 HAL 价格
    作为支撑阻力价格，评估这个价格的历史质量。
    
    Args:
        df: DataFrame with required columns: high, low, close, volume, atr
        window: SQS 计算窗口（用于评估 SR 质量的历史窗口）
        tolerance_factor: ATR 容忍带系数
        sr_type: SR 类型（'support' for HAL low）
        **kwargs: 其他参数（如 poc_window）
    
    Returns:
        DataFrame with 'sqs_hal_low' column added
    """
    result = df.copy()
    
    # 0. 确保 ATR 存在（SQS 计算必需）
    if "atr" not in result.columns:
        result["atr"] = compute_atr(
            result["high"],
            result["low"],
            result["close"],
            period=14
        )
    
    # 1. 检查 poc 和 hal_low 是否已存在（应该来自 poc_hal_features 依赖）
    # 如果不存在，说明依赖关系有问题，需要报错或计算
    if "poc" not in result.columns or "hal_low" not in result.columns:
        # 如果列不存在，尝试计算（向后兼容，但会打印警告）
        print(f"       ⚠️  Warning: 'poc' or 'hal_low' not found, computing them (should come from poc_hal_features dependency)")
        poc_window = kwargs.get("poc_window", 160)
        price_col = kwargs.get("price_col", None)
        if price_col is None and "wpt_price_reconstructed" in result.columns:
            price_col = "wpt_price_reconstructed"
        result = add_poc_hal_dimensionless_features(
            result,
            required_features={"hal_low", "poc"},
            poc_window=poc_window,
            price_col=price_col,
        )
    
    if "hal_low" not in result.columns:
        # 如果 HAL 计算失败，返回全 0
        result["sqs_hal_low"] = 0.0
        return result
    
    # 2. 对每个时间点计算 SQS
    # 使用当前时间点的 HAL 价格作为 sr_price（支撑阻力价格）
    # 评估这个价格在历史窗口内的质量
    sqs_values = []
    hal_low_series = result["hal_low"]
    
    for i in range(len(result)):
        if i < window:
            sqs_values.append(0.0)
            continue
        
        # 获取当前时间点的 HAL low 价格（作为支撑阻力价格）
        sr_price = hal_low_series.iloc[i]
        if pd.isna(sr_price) or sr_price <= 0:
            sqs_values.append(0.0)
            continue
        
        # 获取历史数据窗口（不含未来信息）
        # 只取窗口内的数据，用于评估这个 SR 价格的历史质量
        start_idx = max(0, i - window + 1)
        hist_df = result.iloc[start_idx:i+1].copy()
        
        # 确保有足够的列
        required_cols = ["high", "low", "close", "atr", "volume"]
        if not all(col in hist_df.columns for col in required_cols):
            sqs_values.append(0.0)
            continue
        
        # 计算 SQS：评估 sr_price 在历史窗口内的质量
        try:
            sqs = calculate_sqs(
                sr_price=sr_price,  # 使用当前时间点的 HAL 价格作为 SR 价格
                df=hist_df,  # 历史数据窗口，用于评估质量
                window=min(window, len(hist_df)),
                tolerance_factor=tolerance_factor,
                sr_type=sr_type,
            )
            sqs_values.append(float(sqs) if not np.isnan(sqs) else 0.0)
        except Exception:
            sqs_values.append(0.0)
    
    result["sqs_hal_low"] = pd.Series(sqs_values, index=result.index)
    return result


def compute_sr_strength_max(
    df: pd.DataFrame,
    window: int = 60,
    tolerance_factor: float = 0.5,
    **kwargs
) -> pd.DataFrame:
    """
    计算最大 SR 强度
    
    包装函数：自动获取边界定义，然后计算强度
    
    Args:
        df: DataFrame with required columns
        window: 计算窗口
        tolerance_factor: ATR 容忍带系数
        **kwargs: 其他参数
    
    Returns:
        DataFrame with 'sr_strength_max' column added
    """
    result = df.copy()
    
    # 0. 确保必需的边界列存在（hal_high, hal_low, poc）
    # 这些列可能由 sqs_hal_high/sqs_hal_low 计算，但可能不完整
    # 如果不存在，自动计算它们
    need_compute_boundaries = False
    boundary_cols = ["hal_high", "hal_low", "poc"]
    missing_cols = [col for col in boundary_cols if col not in result.columns]
    
    if missing_cols:
        need_compute_boundaries = True
    else:
        # 检查列是否存在但全部为 NaN
        for col in boundary_cols:
            if col in result.columns and result[col].notna().sum() == 0:
                need_compute_boundaries = True
                break
    
    if need_compute_boundaries:
        # 使用与 sqs_hal_high/sqs_hal_low 相同的参数
        poc_window = kwargs.get("poc_window", 160)
        price_col = kwargs.get("price_col", None)
        if price_col is None and "wpt_price_reconstructed" in result.columns:
            price_col = "wpt_price_reconstructed"
        
        # 计算所有边界列（hal_high, hal_low, poc）
        result = add_poc_hal_dimensionless_features(
            result,
            required_features={"hal_high", "hal_low", "poc"},
            poc_window=poc_window,
            price_col=price_col,
        )
    
    # 1. 确保 ATR 存在（边界强度计算必需）
    if "atr" not in result.columns:
        result["atr"] = compute_atr(
            result["high"],
            result["low"],
            result["close"],
            period=14
        )
    
    # 2. 获取边界定义
    boundaries = _get_sr_boundary_definitions(result)
    
    if not boundaries:
        result["sr_strength_max"] = 0.0
        return result
    
    # 3. 计算边界强度
    compression_series = result.get("compression_confidence")
    boundary_strengths = _compute_boundary_strengths(
        data=result,
        boundaries=boundaries,
        window=window,
        tolerance_factor=tolerance_factor,
        compression_series=compression_series,
    )
    
    # 4. 找到最大强度
    if not boundary_strengths:
        result["sr_strength_max"] = 0.0
        return result
    
    # 合并所有强度序列，取每行的最大值
    strength_df = pd.DataFrame(boundary_strengths)
    result["sr_strength_max"] = strength_df.max(axis=1).fillna(0.0)
    
    return result


@register_feature("compute_footprint_features", category="footprint")
def compute_footprint_features(
    df: pd.DataFrame,
    ticks: Optional[pd.DataFrame] = None,
    ticks_loader_json: Optional[str] = None,
    open_col: str = "open_time",
    close_col: str = "close_time",
    price_bin_size: float = None,
    price_bin_method: str = "fd",
    price_bin_target_bins: int = 40,
    value_area_pct: float = 0.7,
    tick_size: float = None,
    monthly_cache_dir: Optional[str] = "cache/features/monthly",
    persist_monthly: bool = True,
) -> pd.DataFrame:
    """
    Compute single-bar footprint features and merge back to the kline DataFrame.

    Args:
        df: Kline DataFrame with open/close timestamp columns or DateTimeIndex.
        ticks: Tick DataFrame with columns ['price', 'volume', 'side'] and DateTimeIndex.
        ticks_loader_json: JSON string for tick loader params (optional, used if ticks is None).
        open_col/close_col: column names delimiting each bar. If columns don't exist, uses index.
        price_bin_size: explicit bin width; if None, auto.
        price_bin_method: 'fd' (Freedman–Diaconis) or 'fixed_bins'.
        price_bin_target_bins: number of bins when using fixed_bins or as fallback.
        value_area_pct: coverage for VAH/VAL (default 70%).
        tick_size: per-symbol minimum price increment; highest priority when set.

    Returns:
        DataFrame with footprint columns appended.
    
    Raises:
        ValueError: If ticks data is not provided and ticks_loader_json is not available.
    """
    def _normalize_footprint_output(out_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize footprint outputs to be cross-asset comparable (unitless)."""
        out = out_df.copy()
        eps = 1e-8
        close_s = pd.to_numeric(out.get("close"), errors="coerce").astype(float)
        high_s = pd.to_numeric(out.get("high"), errors="coerce").astype(float)
        low_s = pd.to_numeric(out.get("low"), errors="coerce").astype(float)

        # ATR (causal). Prefer existing `atr` if present; otherwise compute quickly.
        atr_s = out.get("atr")
        if atr_s is None:
            # Simple ATR proxy: rolling mean of true range (period=14), causal.
            prev_close = close_s.shift(1)
            tr = pd.concat(
                [
                    (high_s - low_s).abs(),
                    (high_s - prev_close).abs(),
                    (low_s - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr_s = tr.rolling(window=14, min_periods=1).mean()
        atr_s = pd.to_numeric(atr_s, errors="coerce").astype(float).clip(lower=eps)

        def _atr_distance(level: pd.Series) -> pd.Series:
            lv = pd.to_numeric(level, errors="coerce").astype(float)
            return ((lv - close_s) / atr_s).replace([np.inf, -np.inf], np.nan)

        def _signed_log1p(x: pd.Series) -> pd.Series:
            v = pd.to_numeric(x, errors="coerce").astype(float)
            return (np.sign(v) * np.log1p(np.abs(v))).replace([np.inf, -np.inf], np.nan)

        def _robust_rolling_z(
            x: pd.Series, window: int = 50, min_periods: int = 10
        ) -> pd.Series:
            v = pd.to_numeric(x, errors="coerce").astype(float)
            med = v.rolling(window=window, min_periods=min_periods).median()
            q25 = v.rolling(window=window, min_periods=min_periods).quantile(0.25)
            q75 = v.rolling(window=window, min_periods=min_periods).quantile(0.75)
            iqr = (q75 - q25).replace(0, np.nan)
            z = (v - med) / (iqr + eps)
            return z.replace([np.inf, -np.inf], np.nan)

        # Price-level footprint columns → ATR distance (unitless)
        price_cols = [
            "fp_poc",
            "fp_hvn",
            "fp_lvn",
            "fp_vah",
            "fp_val",
            "fp_max_imbalance_price",
            "fp_exhaustion_price",
        ]
        for c in price_cols:
            if c in out.columns:
                out[c] = _atr_distance(out[c])

        # Heavy-tailed / scale-dependent columns → log1p + robust rolling scaling
        if "fp_delta_poc" in out.columns:
            out["fp_delta_poc"] = _robust_rolling_z(_signed_log1p(out["fp_delta_poc"]))

        if "fp_max_imbalance_ratio" in out.columns:
            r = pd.to_numeric(out["fp_max_imbalance_ratio"], errors="coerce").astype(
                float
            )
            # ratio >= 1; compress to long-tail friendly domain
            r_log = np.log1p((r - 1.0).clip(lower=0.0))
            out["fp_max_imbalance_ratio"] = _robust_rolling_z(r_log)

        # Skewness metrics → tanh bound [-1,1]
        for c in ["fp_volume_skew", "fp_delta_skew"]:
            if c in out.columns:
                out[c] = np.tanh(pd.to_numeric(out[c], errors="coerce").astype(float))

        # Exhaustion z-score is already a per-bar z-score across bins; just clip to avoid rare extremes.
        if "fp_exhaustion_zscore" in out.columns:
            out["fp_exhaustion_zscore"] = (
                pd.to_numeric(out["fp_exhaustion_zscore"], errors="coerce")
                .astype(float)
                .clip(-8.0, 8.0)
            )

        # Delta divergence is binary-ish → float in [0,1]
        if "fp_delta_divergence" in out.columns:
            out["fp_delta_divergence"] = (
                pd.to_numeric(out["fp_delta_divergence"], errors="coerce")
                .fillna(0.0)
                .astype(float)
                .clip(0.0, 1.0)
            )
        return out

    # 检查 ticks 数据
    if ticks is None or len(ticks) == 0:
        if ticks_loader_json:
            # From ticks_loader_json: use monthly on-disk cache to avoid reloading ticks and re-computing
            # footprint for each seed/step.
            import hashlib
            import json
            from pathlib import Path

            from src.data_tools.tick_loader import (
                deserialize_tick_loader_params,
                load_tick_data,
            )

            loader_params = deserialize_tick_loader_params(ticks_loader_json)
            if not (isinstance(df.index, pd.DatetimeIndex) and len(df) > 0):
                raise ValueError(
                    "DataFrame must have DatetimeIndex to load ticks from ticks_loader_json"
                )

            symbol = str(loader_params["symbol"]).upper()
            lookback_minutes = int(loader_params.get("lookback_minutes", 60))
            ticks_dir = loader_params.get("ticks_dir")
            if not ticks_dir:
                tick_files = loader_params.get("tick_files", [])
                if tick_files:
                    ticks_dir = str(Path(tick_files[0]).parent)
                else:
                    ticks_dir = "data/parquet_data"

            # Ensure open_time/close_time exist for downstream per-bar slicing.
            local_df = df
            created_time_cols = False
            if open_col not in local_df.columns or close_col not in local_df.columns:
                local_df = local_df.copy()
                created_time_cols = True
                local_df[open_col] = local_df.index
                if len(local_df) > 1:
                    time_delta = local_df.index[1] - local_df.index[0]
                    local_df[close_col] = local_df.index + time_delta
                else:
                    local_df[close_col] = local_df.index + pd.Timedelta(hours=4)

            # Monthly cache key: safe across seeds/steps for the same date range.
            # We include a lightweight bar-index signature to avoid incorrect reuse if the caller
            # changes the bar set within the same month.
            cache_root = Path(monthly_cache_dir) if monthly_cache_dir else None
            cache_params = {
                "price_bin_size": price_bin_size,
                "price_bin_method": price_bin_method,
                "price_bin_target_bins": int(price_bin_target_bins),
                "value_area_pct": float(value_area_pct),
                "tick_size": tick_size,
            }
            params_sig = hashlib.md5(
                json.dumps(cache_params, sort_keys=True, default=str).encode()
            ).hexdigest()[:12]

            fp_parts: list[pd.DataFrame] = []
            # Group bars by calendar month (bar index).
            for period, df_month in local_df.groupby(pd.Grouper(freq="M")):
                if df_month.empty:
                    continue
                month_str = period.strftime("%Y-%m")
                idx_sig = hashlib.md5(
                    (str(df_month.index.min()) + str(df_month.index.max()) + str(len(df_month))).encode()
                ).hexdigest()[:12]
                cache_path = None
                if cache_root and persist_monthly:
                    cache_path = (
                        cache_root
                        / "footprint"
                        / symbol
                        / f"{symbol}_{month_str}_fp_{params_sig}_{idx_sig}.parquet"
                    )
                cached_fp = None
                if cache_path and cache_path.exists():
                    try:
                        cached_fp = pd.read_parquet(cache_path)
                        if isinstance(cached_fp.index, pd.DatetimeIndex) and len(cached_fp) > 0:
                            cached_fp = cached_fp.reindex(df_month.index)
                    except Exception:
                        cached_fp = None

                if cached_fp is None:
                    # Load ticks only for this month span (with lookback padding to be safe).
                    start_ts = df_month[open_col].min().strftime("%Y-%m-%d %H:%M:%S")
                    end_ts = df_month[close_col].max().strftime("%Y-%m-%d %H:%M:%S")
                    month_ticks = load_tick_data(
                        symbol=symbol,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        ticks_dir=ticks_dir,
                        lookback_minutes=lookback_minutes,
                    )

                    cfg = FootprintConfig(
                        price_bin_size=price_bin_size,
                        price_bin_method=price_bin_method,
                        price_bin_target_bins=price_bin_target_bins,
                        value_area_pct=value_area_pct,
                        tick_size=tick_size,
                    )
                    fp_df = compute_kline_footprint_features(
                        ticks=month_ticks,
                        klines=df_month,
                        open_col=open_col,
                        close_col=close_col,
                        cfg=cfg,
                    )
                    cached_fp = fp_df
                    if cache_path:
                        try:
                            cache_path.parent.mkdir(parents=True, exist_ok=True)
                            cached_fp.to_parquet(cache_path)
                        except Exception:
                            pass

                fp_parts.append(cached_fp)

            # Merge per-month footprint features back to original df.
            fp_all = (
                pd.concat(fp_parts, axis=0).sort_index()
                if fp_parts
                else pd.DataFrame(index=local_df.index)
            )
            out = df.copy()
            for col in fp_all.columns:
                out[col] = fp_all[col].reindex(df.index)
            # Apply the same normalization path as the direct ticks branch.
            out = _normalize_footprint_output(out)

            # Drop temporary open_time/close_time if we created them.
            if created_time_cols:
                if open_col in out.columns and open_col not in df.columns:
                    out = out.drop(columns=[open_col])
                if close_col in out.columns and close_col not in df.columns:
                    out = out.drop(columns=[close_col])
            return out
        else:
            raise ValueError(
                "Footprint calculation requires tick data. "
                "Please provide tick data via the 'ticks' parameter "
                "or configure ticks_loader_json. "
                "Footprint cannot be computed without tick data."
            )
    
    # 验证 ticks 数据格式
    required_cols = ["price", "volume", "side"]
    missing_cols = [col for col in required_cols if col not in ticks.columns]
    if missing_cols:
        raise ValueError(
            f"Tick data must contain columns: {required_cols}. "
            f"Missing columns: {missing_cols}"
        )
    # 如果 open_time/close_time 列不存在，使用索引作为时间边界
    if open_col not in df.columns or close_col not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            # 使用索引作为时间边界，创建临时列
            df = df.copy()
            # 对于每个 K 线，open_time 是当前索引，close_time 是下一个索引（或当前索引 + timeframe）
            # 但为了简化，我们假设每个索引代表一个 K 线的开始时间
            # close_time 可以通过计算下一个索引或使用时间间隔来推断
            # 这里我们使用一个简单的方法：如果索引是 DatetimeIndex，直接使用索引
            # 但 compute_kline_footprint_features 需要 open_col 和 close_col 列
            # 所以我们需要创建这些列
            df[open_col] = df.index
            # 对于 close_time，我们需要推断下一个时间点
            # 如果索引是规则的（如 4H），可以使用 shift(-1) 或计算时间间隔
            # 这里我们尝试从索引推断时间间隔
            if len(df) > 1:
                time_delta = df.index[1] - df.index[0]
                df[close_col] = df.index + time_delta
            else:
                # 如果只有一行，使用一个默认的时间间隔（如 4 小时）
                df[close_col] = df.index + pd.Timedelta(hours=4)
        else:
            raise ValueError(
                f"DataFrame must have '{open_col}' and '{close_col}' columns, "
                f"or a DatetimeIndex to infer time boundaries."
            )
    
    cfg = FootprintConfig(
        price_bin_size=price_bin_size,
        price_bin_method=price_bin_method,
        price_bin_target_bins=price_bin_target_bins,
        value_area_pct=value_area_pct,
        tick_size=tick_size,
    )
    fp_df = compute_kline_footprint_features(
        ticks=ticks,
        klines=df,
        open_col=open_col,
        close_col=close_col,
        cfg=cfg,
    )
    # align and merge; footprint rows already aligned to df.index
    out = df.copy()
    for col in fp_df.columns:
        out[col] = fp_df[col]

    out = _normalize_footprint_output(out)
    # 移除临时创建的 open_time/close_time 列（如果它们原本不存在）
    if open_col not in df.columns or close_col not in df.columns:
        if open_col in out.columns and open_col not in df.columns:
            out = out.drop(columns=[open_col])
        if close_col in out.columns and close_col not in df.columns:
            out = out.drop(columns=[close_col])
    return out


@register_feature("compute_unified_volume_profile", category="volume_profile")
def compute_unified_volume_profile(
    df: pd.DataFrame,
    window: int = 160,
    bins: int | str = "auto",
    value_area_ratio: float = 0.7,
    wavelet: str = "db4",
    level: int = 4,
    drop_high_freq: bool = True,
    use_typical_price: bool = False,
    use_wpt_price: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    统一的 Volume Profile 特征计算（合并 POC/HAL 和 VPVR）
    
    在一次计算中同时输出 POC/HAL 和 HVN/LVN 特征，避免重复计算。
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window size (default: 160)
        bins: Number of price bins. If "auto" (default), uses Freedman-Diaconis rule
        value_area_ratio: Value Area ratio (default: 0.7, i.e., 70%)
        wavelet: Wavelet function (default: "db4")
        level: WPT decomposition level (default: 4)
        drop_high_freq: Whether to drop highest frequency subband (default: True)
        use_typical_price: If True, use (H+L+C)/3; else use close or WPT price
        use_wpt_price: If True and wpt_price_reconstructed exists, use it (default: True)
        **kwargs: Other parameters (price_col, volume_col, etc.)
    
    Returns:
        DataFrame with unified volume profile features:
        - vp_poc, vp_poc_volume_ratio, vp_hal_high, vp_hal_low, vp_hal_mid
        - vp_hvn_count, vp_lvn_count, vp_lvn_distance, vp_volume_density, vp_price_in_lvn
    """
    from src.features.time_series.utils_volume_profile import (
        compute_unified_volume_profile_features,
        compute_unified_volume_profile_derived_features,
    )
    
    result = df.copy()
    
    # 确定价格序列
    price_series = None
    if use_wpt_price and "wpt_price_reconstructed" in result.columns:
        price_series = result["wpt_price_reconstructed"]
    
    # 计算基础特征
    result = compute_unified_volume_profile_features(
        result,
        price_col=kwargs.get("price_col", "close"),
        volume_col=kwargs.get("volume_col", "volume"),
        high_col=kwargs.get("high_col", "high"),
        low_col=kwargs.get("low_col", "low"),
        window=window,
        bins=bins,
        value_area_ratio=value_area_ratio,
        wavelet=wavelet,
        level=level,
        drop_high_freq=drop_high_freq,
        use_typical_price=use_typical_price,
        price_series=price_series,
    )
    
    # 计算衍生特征
    result = compute_unified_volume_profile_derived_features(
        result,
        price_col=kwargs.get("price_col", "close"),
    )
    
    return result


def compute_wpt_vpvr(
    df: pd.DataFrame,
    wavelet: str = "db4",
    level: int = 4,
    vpvr_window: int = 100,
    bins: int | str = "auto",
    feature_type: str = "vpvr",
    **kwargs
) -> pd.DataFrame:
    """
    计算 WPT 降噪的 VPVR 特征（向后兼容包装函数）
    
    现在使用统一的 Volume Profile 实现，同时输出 POC/HAL 和 VPVR 特征。
    
    Args:
        df: DataFrame with OHLCV data
        wavelet: Wavelet function
        level: WPT decomposition level
        vpvr_window: VPVR 计算窗口（现在作为 window 参数）
        bins: 价格分箱数。如果为 "auto"（默认），则使用 Freedman-Diaconis rule 自动计算
        feature_type: 特征类型（'vpvr'，向后兼容）
        **kwargs: 其他参数
    
    Returns:
        DataFrame with unified volume profile features
    """
    # 使用统一的实现
    return compute_unified_volume_profile(
        df,
        window=vpvr_window,  # 使用 vpvr_window 作为 window
        bins=bins,
        wavelet=wavelet,
        level=level,
        use_typical_price=True,  # VPVR 使用典型价格
        **kwargs
    )
