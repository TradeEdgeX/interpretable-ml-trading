"""
TA-Lib 特征包装函数

为单个 TA-Lib 指标创建包装函数，支持按需计算
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import talib

from src.features.registry import register_feature


def _prepare_inputs(
    df: Optional[pd.DataFrame], kwargs: Dict[str, object]
) -> tuple[Dict[str, object], pd.Index]:
    """Convert pandas inputs to numpy arrays and infer index."""

    processed: Dict[str, object] = {}
    index: Optional[pd.Index] = None

    for key, value in kwargs.items():
        if isinstance(value, pd.Series):
            # TA-Lib expects float64 ("double") inputs for most indicators.
            processed[key] = (
                pd.to_numeric(value, errors="coerce").astype("float64").values
            )
            if index is None:
                index = value.index
        elif isinstance(value, pd.DataFrame):
            processed[key] = value.apply(pd.to_numeric, errors="coerce").astype(
                "float64"
            ).values
            if index is None:
                index = value.index
        else:
            processed[key] = value

    if index is None:
        if df is None:
            raise ValueError(
                "Unable to infer index for TA-Lib inputs: no pandas Series/DataFrame provided."
            )
        index = df.index

    return processed, index


@register_feature("compute_talib_indicator", category="talib")
def compute_talib_indicator(
    df: pd.DataFrame,
    indicator_name: str,
    *,
    output_column: Optional[str] = None,
    output_columns: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    通用 TA-Lib 指标计算函数，可通过 column_mappings 传入任意参数。

    YAML 中可通过 compute_params 指定 indicator_name/timeperiod，
    column_mappings 映射 DataFrame 列到 talib 函数参数（real/high/low/close 等）。
    """

    result = df.copy()

    talib_func = getattr(talib, indicator_name, None)
    if talib_func is None:
        raise ValueError(f"Unknown TA-Lib indicator: {indicator_name}")

    processed_kwargs, index = _prepare_inputs(df, kwargs)

    try:
        talib_result = talib_func(**processed_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Error computing TA-Lib indicator '{indicator_name}': {exc}"
        ) from exc

    if isinstance(talib_result, tuple):
        cols = output_columns or [
            f"{indicator_name.lower()}_{i}" for i in range(len(talib_result))
        ]
        if len(cols) != len(talib_result):
            raise ValueError(
                f"Output columns mismatch for {indicator_name}: "
                f"expected {len(talib_result)}, got {len(cols)}"
            )
        for col_name, series in zip(cols, talib_result):
            result[col_name] = pd.Series(series, index=index)
    else:
        col = output_column or indicator_name.lower()
        result[col] = pd.Series(talib_result, index=index)

    return result


@register_feature("compute_talib_indicator_from_series", category="talib")
def compute_talib_indicator_from_series(
    *,
    indicator_name: str,
    output_column: Optional[str] = None,
    output_columns: Optional[List[str]] = None,
    normalize_mode: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Narrow-IO entrypoint for TA-Lib indicators.

    Feature pipeline can call this with `pass_full_df: false` + `column_mappings`
    so only required Series are passed (no wide DataFrame).
    
    Args:
        normalize_mode: Optional normalization mode:
            - "position": (close - indicator) / close, for MA indicators (SMA/EMA/TEMA/KAMA)
            - "atr": indicator / ATR, for price-based indicators (requires high/low/close)
            - "change_ratio": pct_change / rolling_std, for cumulative indicators (OBV/AD)
            - "relative_close": indicator / close, for price-level indicators (MACDext/MACDfix/MOM)
            - "return_pct": indicator / close.shift(1), for raw difference-style momentum
            - None: no normalization (default, for backward compatibility)
    """
    # Extract close series for normalization (if needed)
    close_series = kwargs.get('real')
    if close_series is None:
        close_series = kwargs.get('close')
    
    # Infer index from provided Series/DataFrame args, then call the canonical implementation.
    _, index = _prepare_inputs(None, kwargs)
    df = pd.DataFrame(index=index)
    result = compute_talib_indicator(
        df,
        indicator_name,
        output_column=output_column,
        output_columns=output_columns,
        **kwargs,
    )
    
    # Apply normalization if specified (column-wise, supports multi-output indicators)
    if normalize_mode is not None and close_series is not None:
        close_safe = pd.to_numeric(close_series, errors='coerce').replace(0, pd.NA)
        target_cols = output_columns or result.columns

        def _normalize_series(series: pd.Series) -> pd.Series:
            if normalize_mode == "position":
                return (close_safe - series) / close_safe
            if normalize_mode == "atr":
                high = kwargs.get('high')
                low = kwargs.get('low')
                close = kwargs.get('close') or close_series
                if high is None or low is None or close is None:
                    return series
                import talib
                atr = pd.Series(
                    talib.ATR(
                        pd.to_numeric(high, errors='coerce').values,
                        pd.to_numeric(low, errors='coerce').values,
                        pd.to_numeric(close, errors='coerce').values,
                        timeperiod=14,
                    ),
                    index=index,
                ).replace(0, pd.NA)
                return series / atr
            if normalize_mode == "change_ratio":
                change = series.diff()
                rolling_std = change.rolling(window=20, min_periods=5).std().replace(0, pd.NA)
                return change / rolling_std
            if normalize_mode == "relative_close":
                return series / close_safe
            if normalize_mode == "return_pct":
                return series / close_safe.shift(1)
            return series

        for col in target_cols:
            if col in result.columns:
                result[col] = _normalize_series(result[col])
    
    return result


@register_feature("compute_talib_sma", category="talib")
def compute_talib_sma(
    df: pd.DataFrame,
    period: int = 20,
    series: Optional[pd.Series] = None,
    output_column: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    if series is None:
        series = df["close"]
    if output_column is None:
        output_column = f"sma_{period}"

    return compute_talib_indicator(
        df,
        "SMA",
        real=series,
        timeperiod=period,
        output_column=output_column,
        **kwargs,
    )


@register_feature("compute_talib_ema", category="talib")
def compute_talib_ema(
    df: pd.DataFrame,
    period: int = 20,
    series: Optional[pd.Series] = None,
    output_column: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    if series is None:
        series = df["close"]
    if output_column is None:
        output_column = f"ema_{period}"

    return compute_talib_indicator(
        df,
        "EMA",
        real=series,
        timeperiod=period,
        output_column=output_column,
        **kwargs,
    )


@register_feature("compute_talib_rsi", category="talib")
def compute_talib_rsi(
    df: pd.DataFrame,
    period: int = 14,
    series: Optional[pd.Series] = None,
    output_column: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    if series is None:
        series = df["close"]
    if output_column is None:
        output_column = f"rsi_{period}"

    return compute_talib_indicator(
        df,
        "RSI",
        real=series,
        timeperiod=period,
        output_column=output_column,
        **kwargs,
    )


@register_feature("compute_talib_macd", category="talib")
def compute_talib_macd(
    df: pd.DataFrame,
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9,
    series: Optional[pd.Series] = None,
    output_columns: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    if series is None:
        series = df["close"]
    if output_columns is None:
        output_columns = ["macd", "macd_signal", "macd_histogram"]

    return compute_talib_indicator(
        df,
        "MACD",
        real=series,
        fastperiod=fastperiod,
        slowperiod=slowperiod,
        signalperiod=signalperiod,
        output_columns=output_columns,
        **kwargs,
    )
