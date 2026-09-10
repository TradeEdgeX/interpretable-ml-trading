from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.features.registry import register_feature


# ─────────────────────────────────────────────────────────────
# IO helper
# ─────────────────────────────────────────────────────────────

def _load_funding_rate_parquet(symbol: str, funding_rate_dir: str) -> pd.Series:
    """
    Load Binance funding-rate parquet files produced by `mlbot data download-funding-rate`.

    Expected file pattern:
      <funding_rate_dir>/<SYMBOL>_YYYY-MM_funding_rate.parquet

    Expected schema:
      - DatetimeIndex named 'datetime' (UTC)
      - column: funding_rate
    """
    sym = str(symbol).strip().upper()
    root = Path(funding_rate_dir)
    paths = sorted(root.glob(f"{sym}_*_funding_rate.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No funding-rate parquet files found for {sym} under {root}. "
            f"Run: mlbot data download-funding-rate --symbols {sym} --start-year <Y> --start-month <M>"
        )

    parts: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_parquet(p)
        parts.append(df)

    df_all = pd.concat(parts, axis=0, ignore_index=False)
    if not isinstance(df_all.index, pd.DatetimeIndex):
        if "datetime" in df_all.columns:
            df_all.index = pd.to_datetime(df_all["datetime"], utc=True)
        else:
            raise ValueError(f"Funding-rate parquet schema invalid (no DatetimeIndex): {paths[0]}")

    idx = df_all.index
    idx_utc = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    s = pd.to_numeric(df_all.get("funding_rate"), errors="coerce")
    s.index = idx_utc
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


# ─────────────────────────────────────────────────────────────
# Rolling helpers
# ─────────────────────────────────────────────────────────────

def _rolling_robust_zscore(x: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Robust rolling z-score using median / MAD.

    Resistant to fat-tail spikes typical in funding-rate data.
    For normally distributed data: std \u2248 1.4826 * MAD.

    Uses ``r.apply`` for correct within-window MAD computation.
    Performance is fine for low-frequency data (funding \u2248 3 obs/day).
    """
    r = x.rolling(window=window, min_periods=min_periods)
    med = r.median()
    mad = r.apply(
        lambda w: np.median(np.abs(w - np.median(w))), raw=True
    )
    scale = (mad * 1.4826).replace(0.0, np.nan)
    return (x - med) / scale


def _sigmoid01(x: pd.Series) -> pd.Series:
    # stable-ish sigmoid into (0, 1)
    return 1.0 / (1.0 + np.exp(-x.astype(float)))


# ─────────────────────────────────────────────────────────────
# Raw funding-rate features
# ─────────────────────────────────────────────────────────────

@register_feature(
    "compute_funding_rate_features_from_df",
    category="order_flow",
    description=(
        "Attach funding rate (Binance futures) and derived features aligned "
        "to bar timestamps (no look-ahead via merge_asof backward). "
        "All rolling statistics are computed on **native funding frequency** "
        "(~8 h) before merging to bars, eliminating repeated-value bias."
    ),
    outputs=[
        "funding_rate",
        "funding_rate_abs",
        "funding_rate_change_1",
        "funding_rate_zscore_50",
        "funding_rate_abs_zscore_50",
    ],
)
def compute_funding_rate_features_from_df(
    df: pd.DataFrame,
    *,
    funding_rate_dir: str = "data/funding_rate/parquet",
    on_missing: Literal["nan", "zero", "raise"] = "nan",
    node_cache_version: str | None = None,
    cache_version: str | None = None,
    z_window: int = 50,
    z_min_periods: int = 20,
) -> pd.DataFrame:
    """Join funding rate to kline bars by ``merge_asof`` (<= bar timestamp).

    **Architecture (v2)**
    ---------------------
    All derived features (change, zscore) are computed on the **native
    funding-rate series** (~8 h cadence) *before* the merge-asof join.
    This avoids the repeated-value bias caused by expanding a low-frequency
    series onto high-frequency bars and then computing rolling statistics.

    ``z_window=50`` therefore means the last **50 funding observations**
    (≈ 400 h ≈ 16.7 days), not 50 bars.

    Robust z-score (median / MAD) is used instead of mean / std to resist
    fat-tail spikes inherent to funding-rate data.

    Requirements
    ------------
    * ``df.index`` must be DatetimeIndex
    * ``df`` must contain ``_symbol`` (preferred) or ``symbol``
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df index must be a DatetimeIndex")

    sym_col = (
        "_symbol"
        if "_symbol" in df.columns
        else ("symbol" if "symbol" in df.columns else None)
    )
    if sym_col is None:
        raise KeyError(
            "df must contain '_symbol' (preferred) or 'symbol' for funding-rate join"
        )

    idx_utc = (
        df.index.tz_localize("UTC")
        if df.index.tz is None
        else df.index.tz_convert("UTC")
    )

    # Pre-allocate output (all NaN)
    out_cols = [
        "funding_rate",
        "funding_rate_abs",
        "funding_rate_change_1",
        "funding_rate_zscore_50",
        "funding_rate_abs_zscore_50",
    ]
    out = pd.DataFrame(np.nan, index=df.index, columns=out_cols)

    left = pd.DataFrame({"_ts": idx_utc})
    left["_i"] = np.arange(len(left), dtype=int)

    for sym in pd.Series(df[sym_col]).astype(str).fillna("").unique():
        if not sym:
            continue
        mask = pd.Series(df[sym_col]).astype(str) == sym
        if not bool(mask.any()):
            continue
        mask_np = mask.to_numpy()

        try:
            fr_native = _load_funding_rate_parquet(
                sym, funding_rate_dir=funding_rate_dir
            )
        except Exception:
            if str(on_missing).lower() == "raise":
                raise
            continue

        # ── Compute ALL derived features on native frequency ──
        fr_abs = fr_native.abs()
        fr_change = fr_native.diff()                            # real interval change
        fr_z = _rolling_robust_zscore(
            fr_native, window=int(z_window), min_periods=int(z_min_periods)
        )
        fr_abs_z = _rolling_robust_zscore(
            fr_abs, window=int(z_window), min_periods=int(z_min_periods)
        )

        # Build right-side DataFrame with all features for single merge_asof
        right = pd.DataFrame({
            "_ts": fr_native.index,
            "funding_rate": fr_native.values,
            "funding_rate_abs": fr_abs.values,
            "funding_rate_change_1": fr_change.values,
            "funding_rate_zscore_50": fr_z.values,
            "funding_rate_abs_zscore_50": fr_abs_z.values,
        }).sort_values("_ts")

        left_sym = left.loc[mask_np, ["_ts", "_i"]].sort_values("_ts")
        merged = pd.merge_asof(
            left_sym, right, on="_ts",
            direction="backward", allow_exact_matches=True,
        )

        # Map back to original row order
        i_vals = left.loc[mask_np, "_i"].to_numpy()
        merged_indexed = merged.set_index("_i")
        for col in out_cols:
            out.loc[mask_np, col] = (
                merged_indexed[col].reindex(i_vals).to_numpy()
            )

    if str(on_missing).lower() == "zero":
        out["funding_rate"] = out["funding_rate"].fillna(0.0)
    elif str(on_missing).lower() == "raise" and out["funding_rate"].isna().any():
        raise ValueError(
            "Missing funding_rate after join; check downloaded data coverage"
        )

    return out


@register_feature(
    "compute_funding_scene_semantic_scores_from_df",
    category="interaction",
    description="Funding-rate scene semantic scores (0..1): compression/ignition/absorption/exhaustion, gated by trend+compression regime.",
    outputs=[
        "funding_compression_score",
        "funding_ignition_score",
        "funding_absorption_score",
        "funding_exhaustion_scene_score",
    ],
)
def compute_funding_scene_semantic_scores_from_df(
    df: pd.DataFrame,
    *,
    funding_z_col: str = "funding_rate_abs_zscore_50",
    compression_col: str = "compression_score",
    trend_col: str = "trend_r2_20",
    z_shift: float = 1.0,
    z_scale: float = 1.0,
) -> pd.DataFrame:
    """
    A simple, causal, strategy-agnostic mapping:
    - funding_stress := sigmoid((abs_z - z_shift) / z_scale)
    - compression_score := stress * compression * (1 - trend)
    - ignition_score := stress * (1 - compression) * trend
    - absorption_score := stress * compression * trend_low
    - exhaustion_score := stress * (1 - compression) * (1 - trend)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df index must be a DatetimeIndex")

    abs_z = pd.to_numeric(df[funding_z_col], errors="coerce").fillna(0.0)
    comp = pd.to_numeric(df[compression_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    trend = pd.to_numeric(df[trend_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    stress = _sigmoid01((abs_z - float(z_shift)) / float(z_scale))

    out = pd.DataFrame(index=df.index)
    out["funding_compression_score"] = (stress * comp * (1.0 - trend)).clip(0.0, 1.0)
    out["funding_ignition_score"] = (stress * (1.0 - comp) * trend).clip(0.0, 1.0)
    # "Absorption" here is defined as: funding stress present, but trend is also strong
    # (crowding while price keeps moving) — useful as a regime marker for "crowded continuation".
    out["funding_absorption_score"] = (stress * comp * trend).clip(0.0, 1.0)
    out["funding_exhaustion_scene_score"] = (stress * (1.0 - comp) * (1.0 - trend)).clip(0.0, 1.0)
    return out


