"""Signed realized-R label under a fixed execution profile (simulate_rr_execution)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scripts.backtest_execution_layer import simulate_rr_execution
from src.time_series_model.strategies.labels.execution_config_utils import (
    load_exec_profile,
    normalize_execution_config,
)
from src.time_series_model.strategies.labels.forward_rr_signed_label import (
    compute_raw_signed_forward_rr,
)


def _resolve_direction_col(df: pd.DataFrame) -> str:
    for col in ("entry_direction", "direction", "signal"):
        if col in df.columns:
            return col
    return "entry_direction"


def _simulate_side_rr(
    df: pd.DataFrame,
    *,
    exec_config: dict[str, Any],
    side: int,
    direction_col: str,
    atr_col: str,
    use_1min: bool = False,
    bars_1min_dict: dict[str, pd.DataFrame] | None = None,
    bar_minutes: int = 120,
) -> pd.Series:
    work = df.copy()
    work[direction_col] = float(side)
    if bars_1min_dict is not None:
        work["_bar_minutes"] = int(bar_minutes)
    kwargs: dict[str, Any] = {
        "direction_col": direction_col,
        "atr_col": atr_col,
        "silent": True,
    }
    if bars_1min_dict is not None:
        kwargs["bars_1min_dict"] = bars_1min_dict
    rr, _details = simulate_rr_execution(work, exec_config, **kwargs)
    return rr


def compute_realized_r_under_execution(
    df: pd.DataFrame,
    *,
    exec_profile: str | None = None,
    execution_yaml: str | None = None,
    strategies_root: str = "config/strategies/tree_strategies",
    strategy: str = "fast_scalp",
    exec_config: dict[str, Any] | None = None,
    atr_col: str = "atr",
    rr_floor: float = 0.0,
    direction_col: str | None = None,
    use_1min: bool = False,
    bars_1min_dict: dict[str, pd.DataFrame] | None = None,
    bar_minutes: int = 120,
    include_forward_rr: bool = False,
    forward_horizon: int = 3,
    price_col: str = "close",
) -> pd.DataFrame:
    """Return label = r_long - r_short under fixed execution semantics."""
    if exec_config is None:
        cfg = load_exec_profile(
            exec_profile=exec_profile,
            execution_yaml=execution_yaml,
            strategies_root=strategies_root,
            strategy=strategy,
        )
    else:
        cfg = normalize_execution_config(exec_config)

    dir_col = direction_col or _resolve_direction_col(df)
    if dir_col not in df.columns:
        df = df.copy()
        df[dir_col] = 0.0

    r_long = _simulate_side_rr(
        df,
        exec_config=cfg,
        side=1,
        direction_col=dir_col,
        atr_col=atr_col,
        use_1min=use_1min,
        bars_1min_dict=bars_1min_dict,
        bar_minutes=bar_minutes,
    )
    r_short = _simulate_side_rr(
        df,
        exec_config=cfg,
        side=-1,
        direction_col=dir_col,
        atr_col=atr_col,
        use_1min=use_1min,
        bars_1min_dict=bars_1min_dict,
        bar_minutes=bar_minutes,
    )
    label = r_long - r_short
    if rr_floor > 0:
        label = label.where(label.abs() >= rr_floor)

    out = pd.DataFrame(index=df.index)
    out["realized_r_long"] = r_long
    out["realized_r_short"] = r_short
    out["label"] = label
    if include_forward_rr and forward_horizon > 0:
        out["forward_rr"] = compute_raw_signed_forward_rr(
            df,
            horizon=forward_horizon,
            price_col=price_col,
            atr_col=atr_col,
        )
    return out


def compute_execution_realized_r_label(
    df: pd.DataFrame,
    *,
    exec_profile: str | None = None,
    execution_yaml: str | None = None,
    strategies_root: str = "config/strategies/tree_strategies",
    strategy: str = "fast_scalp",
    atr_col: str = "atr",
    rr_floor: float = 0.0,
    direction_col: str | None = None,
    include_forward_rr: bool = False,
    forward_horizon: int = 3,
    price_col: str = "close",
    bar_minutes: int = 120,
    holdout_embargo_minutes: int | None = None,
    **_ignored: Any,
) -> pd.Series:
    """Train pipeline entry: returns signed realized-R label Series.

    ``holdout_embargo_minutes`` is consumed by the train pipeline holdout split
    (not the label math); accepted here so it can live in labels.yaml params.
    """
    _ = holdout_embargo_minutes
    block = compute_realized_r_under_execution(
        df,
        exec_profile=exec_profile,
        execution_yaml=execution_yaml,
        strategies_root=strategies_root,
        strategy=strategy,
        atr_col=atr_col,
        rr_floor=rr_floor,
        direction_col=direction_col,
        include_forward_rr=include_forward_rr,
        forward_horizon=forward_horizon,
        price_col=price_col,
        bar_minutes=bar_minutes,
    )
    for col in ("realized_r_long", "realized_r_short", "forward_rr"):
        if col in block.columns:
            df[col] = block[col].values
    return block["label"]


def load_bars_1min_dict(
    symbols: list[str],
    *,
    data_path: str = "data/parquet_data",
    start_date: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    """Load 1min OHLCV per symbol for simulate_rr use_1min=True."""
    from src.data_tools.data_handler import DataHandler

    dh = DataHandler(data_path)
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        sym_key = str(sym).upper()
        bars = dh.load_ohlcv(
            symbol=sym_key, timeframe="1T", start_date=start_date, end_date=end_date
        )
        if bars.empty:
            continue
        bars = bars.copy()
        bars.index = pd.to_datetime(bars.index, utc=True)
        out[sym_key] = bars
    return out


def compute_realized_r_1min_ablation(
    df: pd.DataFrame,
    *,
    symbols: list[str] | None = None,
    data_path: str = "data/parquet_data",
    start_date: str | None = None,
    end_date: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Holdout ablation: signed realized-R via 1min path inside simulate_rr."""
    sym_col = "_symbol" if "_symbol" in df.columns else "symbol"
    if symbols is None:
        if sym_col in df.columns:
            symbols = sorted(df[sym_col].dropna().astype(str).str.upper().unique())
        else:
            symbols = []
    if start_date is None or end_date is None:
        idx = df.index if isinstance(df.index, pd.DatetimeIndex) else None
        if idx is not None and len(idx):
            start_date = start_date or str(idx.min().date())
            end_date = end_date or str(idx.max().date())
        else:
            raise ValueError(
                "start_date/end_date required when df has no DatetimeIndex"
            )
    bars_1min_dict = load_bars_1min_dict(
        list(symbols),
        data_path=data_path,
        start_date=str(start_date),
        end_date=str(end_date),
    )
    return compute_realized_r_under_execution(
        df,
        use_1min=True,
        bars_1min_dict=bars_1min_dict,
        **kwargs,
    )
