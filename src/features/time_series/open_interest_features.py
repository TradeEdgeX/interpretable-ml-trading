"""
Open Interest (OI) feature engineering for Binance futures.

Parquet schema produced by ``download_open_interest.py``::

    DatetimeIndex('datetime', UTC)
    columns: _symbol, oi_contracts, oi_usd

Two-layer OI feature taxonomy
-----------------------------

**Inventory layer** (OI level – crowding / positioning extremes):

* ``oi_usd``             – raw USD-denominated OI (forward-filled onto bars)
* ``oi_zscore``          – rolling z-score of OI *level*
* ``oi_delta_price_sign``– sign agreement between OI change and price change
                           (+1 = both up or both down, -1 = divergence)

**Flow layer** (ΔOI – participation momentum / capital flow rate):

* ``oi_change_pct``      – bar-over-bar % change
* ``oi_flow_zscore``     – rolling z-score of *ΔOI* (is the rate of change abnormal?)

Scene semantic scores (see ``compute_oi_scene_semantic_scores_from_df``):

* ``oi_compression_score``      – OI building up while price compresses
* ``oi_ignition_score``         – OI + price move together (trend confirmation)
* ``oi_absorption_score``       – OI rises but price stalls (hidden accumulation)
* ``oi_exhaustion_score``       – OI drops while volatility stays high (unwinding)
* ``oi_trend_divergence_score`` – trend strong but OI flow stalls (ME → FER transition)

All features are **causal** (no look-ahead): OI values are ``merge_asof``
backward-joined so each bar only sees OI known *at or before* that bar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.features.registry import register_feature


# ─────────────────────────────────────────────────────────────
# IO helper
# ─────────────────────────────────────────────────────────────

def _load_oi_parquet(
    symbol: str,
    oi_dir: str,
    period: str = "5m",
) -> pd.DataFrame:
    """
    Load OI parquet files produced by ``download_open_interest.py``.

    File pattern: ``<oi_dir>/<SYMBOL>_YYYY-MM_oi_<period>.parquet``
    Schema: DatetimeIndex('datetime', UTC), columns: oi_contracts, oi_usd
    """
    sym = str(symbol).strip().upper()
    root = Path(oi_dir)
    paths = sorted(root.glob(f"{sym}_*_oi_{period}.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No OI parquet files found for {sym} (period={period}) under {root}. "
            f"Run: mlbot data download-open-interest --symbols {sym} --start-year <Y> --start-month <M> --period {period}"
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
            raise ValueError(
                f"OI parquet schema invalid (no DatetimeIndex): {paths[0]}"
            )

    idx = df_all.index
    idx_utc = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df_all.index = idx_utc
    df_all = df_all.sort_index()
    df_all = df_all[~df_all.index.duplicated(keep="last")]
    return df_all


# ─────────────────────────────────────────────────────────────
# Rolling helpers (stateless / streaming-safe)
# ─────────────────────────────────────────────────────────────

def _rolling_zscore(x: pd.Series, window: int, min_periods: int) -> pd.Series:
    r = x.rolling(window=window, min_periods=min_periods)
    mean = r.mean()
    std = r.std(ddof=0).replace(0.0, np.nan)
    return (x - mean) / std


def _sigmoid01(x: pd.Series) -> pd.Series:
    return 1.0 / (1.0 + np.exp(-x.astype(float)))


# ─────────────────────────────────────────────────────────────
# Raw OI features  (pass_full_df = True, like funding_rate)
# ─────────────────────────────────────────────────────────────

@register_feature(
    "compute_oi_features_from_df",
    category="order_flow",
    description=(
        "Attach Open Interest (Binance futures) and derived features "
        "aligned to bar timestamps (no look-ahead via merge_asof backward)."
    ),
    outputs=[
        "oi_usd",
        "oi_change_pct",
        "oi_zscore",
        "oi_delta_price_sign",
        "oi_flow_zscore",
    ],
)
def compute_oi_features_from_df(
    df: pd.DataFrame,
    *,
    oi_dir: str = "data/open_interest/parquet",
    oi_period: str = "5m",
    on_missing: Literal["nan", "zero", "raise"] = "nan",
    node_cache_version: str | None = None,
    z_window: int = 50,
    z_min_periods: int = 20,
) -> pd.DataFrame:
    """
    Join OI to kline bars by ``merge_asof`` (<= bar timestamp).

    Requirements
    ------------
    * ``df.index`` must be DatetimeIndex
    * ``df`` must contain ``_symbol`` (preferred) or ``symbol``
    * ``df`` must contain ``close`` for delta-price-sign calculation
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
            "df must contain '_symbol' or 'symbol' for OI join"
        )

    idx_utc = (
        df.index.tz_localize("UTC")
        if df.index.tz is None
        else df.index.tz_convert("UTC")
    )

    out = pd.DataFrame(index=df.index)
    out["oi_usd"] = np.nan

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
            oi_df = _load_oi_parquet(sym, oi_dir=oi_dir, period=oi_period)
        except Exception:
            if str(on_missing).lower() == "raise":
                raise
            continue

        oi_usd = pd.to_numeric(oi_df.get("oi_usd"), errors="coerce")
        right = pd.DataFrame(
            {"_ts": oi_usd.index, "oi_usd": oi_usd.values}
        ).sort_values("_ts")

        left_sym = left.loc[mask_np, ["_ts", "_i"]].sort_values("_ts")
        merged = pd.merge_asof(
            left_sym, right, on="_ts",
            direction="backward", allow_exact_matches=True,
        )
        i_vals = left.loc[mask_np, "_i"].to_numpy()
        out.loc[mask_np, "oi_usd"] = (
            merged.set_index("_i")["oi_usd"].reindex(i_vals).to_numpy()
        )

    if str(on_missing).lower() == "zero":
        out["oi_usd"] = out["oi_usd"].fillna(0.0)
    elif str(on_missing).lower() == "raise" and out["oi_usd"].isna().any():
        raise ValueError(
            "Missing OI data after join; check downloaded data coverage"
        )

    oi = pd.to_numeric(out["oi_usd"], errors="coerce")

    # % change (bar-over-bar)
    oi_prev = oi.shift(1)
    out["oi_change_pct"] = ((oi - oi_prev) / oi_prev.replace(0.0, np.nan)).clip(-1.0, 1.0)

    # rolling z-score
    out["oi_zscore"] = _rolling_zscore(
        oi, window=int(z_window), min_periods=int(z_min_periods)
    )

    # delta-price-sign: +1 if OI and price move in same direction, -1 otherwise
    if "close" in df.columns:
        price_chg = pd.to_numeric(df["close"], errors="coerce").diff()
        oi_chg = oi.diff()
        out["oi_delta_price_sign"] = np.sign(price_chg * oi_chg).fillna(0.0)
    else:
        out["oi_delta_price_sign"] = 0.0

    # ── Flow layer: z-score of ΔOI (participation momentum) ──
    # Unlike oi_zscore (inventory level), this captures whether the *rate*
    # of OI change is abnormally high/low right now.
    oi_change = oi.diff()  # absolute ΔOI
    out["oi_flow_zscore"] = _rolling_zscore(
        oi_change, window=int(z_window), min_periods=int(z_min_periods)
    )

    return out


# ─────────────────────────────────────────────────────────────
# OI Scene Semantic Scores  (like funding scene semantic)
# ─────────────────────────────────────────────────────────────

@register_feature(
    "compute_oi_scene_semantic_scores_from_df",
    category="interaction",
    description=(
        "OI scene semantic scores (0..1): compression/ignition/absorption/exhaustion, "
        "derived from OI z-score, compression regime and trend strength."
    ),
    outputs=[
        "oi_compression_score",
        "oi_ignition_score",
        "oi_absorption_score",
        "oi_exhaustion_score",
        "oi_trend_divergence_score",
    ],
)
def compute_oi_scene_semantic_scores_from_df(
    df: pd.DataFrame,
    *,
    oi_z_col: str = "oi_zscore",
    oi_flow_z_col: str = "oi_flow_zscore",
    compression_col: str = "compression_score",
    trend_col: str = "trend_r2_20",
    volatility_col: str = "atr_percentile",
    z_shift: float = 0.5,
    z_scale: float = 1.0,
    divergence_flow_scale: float = 1.5,
) -> pd.DataFrame:
    """
    Map OI activity into five archetype-agnostic scene semantic scores.

    OI semantics
    -------------
    * **Compression**: OI building (high z-score) + price compressed + no trend
      → Position accumulation without directional move = structural pressure
    * **Ignition**: OI building + trend present + no compression
      → New positions entering WITH price direction = momentum confirmation
    * **Absorption**: OI building + compressed + trend present
      → Hidden accumulation during consolidation-in-trend
    * **Exhaustion**: OI z-score negative (unwinding) + no compression + no trend
      → Leveraged positions closing out = trend decay / reversal risk
    * **Trend Divergence**: trend strong but OI *flow* stalling/negative + high volatility
      → Price trending on existing positions, no new capital, volatility elevated
      = "last leg" exhaustion signal (ME → FER transition)

    All scores in [0, 1], causal (uses only past data).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df index must be a DatetimeIndex")

    oi_z = pd.to_numeric(df.get(oi_z_col, pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    comp = pd.to_numeric(df.get(compression_col, pd.Series(dtype=float)), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    trend = pd.to_numeric(df.get(trend_col, pd.Series(dtype=float)), errors="coerce").fillna(0.0).clip(0.0, 1.0)

    # OI activity: sigmoid of z-score  (high z → more OI buildup)
    oi_activity = _sigmoid01((oi_z - float(z_shift)) / float(z_scale))

    out = pd.DataFrame(index=df.index)

    # Compression: OI building × compression × no-trend
    out["oi_compression_score"] = (oi_activity * comp * (1.0 - trend)).clip(0.0, 1.0)

    # Ignition: OI building × no-compression × trend
    out["oi_ignition_score"] = (oi_activity * (1.0 - comp) * trend).clip(0.0, 1.0)

    # Absorption: OI building × compression × trend
    out["oi_absorption_score"] = (oi_activity * comp * trend).clip(0.0, 1.0)

    # Exhaustion: OI unwinding (1 - activity) × no-compression × no-trend
    oi_unwinding = (1.0 - oi_activity)
    out["oi_exhaustion_score"] = (oi_unwinding * (1.0 - comp) * (1.0 - trend)).clip(0.0, 1.0)

    # ── Trend Divergence: trend strong × sigmoid(-flow_z) × volatility ──
    # Upgrade v2: uses sigmoid(-flow_z) instead of (1 - sigmoid(flow_z))
    # so that flow < 0 triggers strongly, flow = 0 is near-neutral.
    # Also gated by volatility (atr_percentile): true exhaustion has
    # high trend + negative flow + HIGH volatility (“last leg”).
    oi_flow_z = pd.to_numeric(
        df.get(oi_flow_z_col, pd.Series(dtype=float)), errors="coerce"
    ).fillna(0.0)
    # sigmoid(-flow_z / scale): flow_z=-2 → sigmoid(1.33) ≈ 0.79, flow_z=0 → 0.5, flow_z=2 → 0.21
    flow_penalty = _sigmoid01(-oi_flow_z / float(divergence_flow_scale))
    vol = pd.to_numeric(
        df.get(volatility_col, pd.Series(dtype=float)), errors="coerce"
    ).reindex(df.index).fillna(0.5).clip(0.0, 1.0)
    out["oi_trend_divergence_score"] = (trend * flow_penalty * vol).clip(0.0, 1.0)

    return out
