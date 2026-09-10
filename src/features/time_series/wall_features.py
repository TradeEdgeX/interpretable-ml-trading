"""
T5α orderbook wall features — join depth snapshots to bars (merge_asof backward).

Primary history: Vision ``bookDepth`` daily parquet (``download-book-depth``).
Optional live/incremental: REST poll (``download-depth-snapshots``).

WS-only columns (``wall_persist_sec``, ``wall_cancel_rate_5m``, ``wall_eaten_ratio_1h``)
are emitted as NaN until a WS pipeline exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.features.registry import register_feature


def _resolve_depth_dirs(depth_dir: str | None) -> list[Path]:
    if depth_dir:
        return [Path(depth_dir)]
    return [
        Path("data/book_depth/parquet"),
        Path("data/orderbook/parquet"),
    ]


def _load_depth_snapshots(symbol: str, depth_dir: str | None) -> pd.DataFrame:
    sym = str(symbol).strip().upper()
    paths: list[Path] = []
    for root in _resolve_depth_dirs(depth_dir):
        if not root.exists():
            continue
        paths.extend(sorted(root.glob(f"{sym}_*_book_depth.parquet")))
        paths.extend(sorted(root.glob(f"{sym}_*_depth_snap.parquet")))
    if not paths:
        roots = ", ".join(str(p) for p in _resolve_depth_dirs(depth_dir))
        raise FileNotFoundError(
            f"No wall parquet for {sym} under [{roots}]. "
            f"Run: mlbot data download-book-depth --symbols {sym} --start-date YYYY-MM-DD"
        )
    parts: list[pd.DataFrame] = []
    for p in paths:
        parts.append(pd.read_parquet(p))
    df = pd.concat(parts, axis=0, ignore_index=False)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df.index = pd.to_datetime(df["datetime"], utc=True)
        else:
            raise ValueError(f"Invalid depth snapshot schema: {paths[0]}")
    idx = df.index
    df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return df.sort_index()[~df.index.duplicated(keep="last")]


@register_feature(
    "compute_wall_features_from_df",
    category="order_flow",
    description=(
        "Join Binance depth wall snapshots to bars (merge_asof backward). "
        "REST Phase 1B: persist/cancel/eaten columns are NaN until WS pipeline."
    ),
    outputs=[
        "wall_bid_notional_usd_max",
        "wall_ask_notional_usd_max",
        "wall_bid_price",
        "wall_ask_price",
        "wall_nearest_dist_atr",
        "wall_persist_sec",
        "wall_cancel_rate_5m",
        "wall_eaten_ratio_1h",
    ],
)
def compute_wall_features_from_df(
    df: pd.DataFrame,
    *,
    depth_dir: str | None = None,
    on_missing: Literal["nan", "zero", "raise"] = "nan",
    atr_col: str = "atr",
) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df index must be a DatetimeIndex")

    sym_col = (
        "_symbol"
        if "_symbol" in df.columns
        else ("symbol" if "symbol" in df.columns else None)
    )
    if sym_col is None:
        raise KeyError("df must contain '_symbol' or 'symbol'")

    idx_utc = (
        df.index.tz_localize("UTC")
        if df.index.tz is None
        else df.index.tz_convert("UTC")
    )

    out_cols = [
        "wall_bid_notional_usd_max",
        "wall_ask_notional_usd_max",
        "wall_bid_price",
        "wall_ask_price",
        "wall_nearest_dist_atr",
        "wall_persist_sec",
        "wall_cancel_rate_5m",
        "wall_eaten_ratio_1h",
    ]
    out = pd.DataFrame(np.nan, index=df.index, columns=out_cols)

    left = pd.DataFrame({"_ts": idx_utc})
    left["_i"] = np.arange(len(left), dtype=int)

    snap_cols = [
        "wall_bid_notional_usd_max",
        "wall_ask_notional_usd_max",
        "wall_bid_price",
        "wall_ask_price",
        "wall_bid_pct_band",
        "wall_ask_pct_band",
        "mid",
    ]

    for sym in pd.Series(df[sym_col]).astype(str).fillna("").unique():
        if not sym:
            continue
        mask = pd.Series(df[sym_col]).astype(str) == sym
        if not bool(mask.any()):
            continue
        mask_np = mask.to_numpy()

        try:
            snaps = _load_depth_snapshots(sym, depth_dir=depth_dir)
        except Exception:
            if str(on_missing).lower() == "raise":
                raise
            continue

        use_cols = [c for c in snap_cols if c in snaps.columns]
        right = snaps[use_cols].copy()
        right = right.reset_index()
        ts_col = "datetime" if "datetime" in right.columns else right.columns[0]
        right = right.rename(columns={ts_col: "_ts"}).sort_values("_ts")

        left_sym = left.loc[mask_np, ["_ts", "_i"]].sort_values("_ts")
        merged = pd.merge_asof(
            left_sym,
            right,
            on="_ts",
            direction="backward",
            allow_exact_matches=True,
        )
        i_vals = left.loc[mask_np, "_i"].to_numpy()
        mi = merged.set_index("_i")

        for col in [
            "wall_bid_notional_usd_max",
            "wall_ask_notional_usd_max",
            "wall_bid_price",
            "wall_ask_price",
        ]:
            if col in mi.columns:
                out.loc[mask_np, col] = mi[col].reindex(i_vals).to_numpy()

        if "wall_bid_pct_band" in mi.columns and "close" in df.columns:
            close_arr = pd.to_numeric(
                df.loc[mask_np, "close"], errors="coerce"
            ).to_numpy()
            bid_px = out.loc[mask_np, "wall_bid_price"].to_numpy()
            ask_px = out.loc[mask_np, "wall_ask_price"].to_numpy()
            bid_pct = mi["wall_bid_pct_band"].reindex(i_vals).to_numpy()
            ask_pct = mi["wall_ask_pct_band"].reindex(i_vals).to_numpy()
            out.loc[mask_np, "wall_bid_price"] = np.where(
                np.isfinite(bid_px), bid_px, close_arr * (1.0 + bid_pct / 100.0)
            )
            out.loc[mask_np, "wall_ask_price"] = np.where(
                np.isfinite(ask_px), ask_px, close_arr * (1.0 + ask_pct / 100.0)
            )

        if atr_col in df.columns:
            atr = pd.to_numeric(df.loc[mask_np, atr_col], errors="coerce").to_numpy()
            close = (
                pd.to_numeric(df.loc[mask_np, "close"], errors="coerce").to_numpy()
                if "close" in df.columns
                else mi["mid"].reindex(i_vals).to_numpy()
            )
            bid_px = (
                mi["wall_bid_price"].reindex(i_vals).to_numpy()
                if "wall_bid_price" in mi.columns
                else np.full(len(i_vals), np.nan)
            )
            ask_px = (
                mi["wall_ask_price"].reindex(i_vals).to_numpy()
                if "wall_ask_price" in mi.columns
                else np.full(len(i_vals), np.nan)
            )
            if "wall_bid_pct_band" in mi.columns:
                bid_pct = mi["wall_bid_pct_band"].reindex(i_vals).to_numpy()
                ask_pct = mi["wall_ask_pct_band"].reindex(i_vals).to_numpy()
                bid_px = np.where(
                    np.isfinite(bid_px),
                    bid_px,
                    close * (1.0 + bid_pct / 100.0),
                )
                ask_px = np.where(
                    np.isfinite(ask_px),
                    ask_px,
                    close * (1.0 + ask_pct / 100.0),
                )
            dist_bid = np.abs(close - bid_px)
            dist_ask = np.abs(ask_px - close)
            dist = np.minimum(dist_bid, dist_ask)
            with np.errstate(divide="ignore", invalid="ignore"):
                out.loc[mask_np, "wall_nearest_dist_atr"] = dist / np.where(
                    atr > 0, atr, np.nan
                )

    if str(on_missing).lower() == "zero":
        for col in out_cols:
            out[col] = out[col].fillna(0.0)
    elif str(on_missing).lower() == "raise" and out["wall_bid_notional_usd_max"].isna().any():
        raise ValueError("Missing wall data after join; run depth snapshot download")

    return out
