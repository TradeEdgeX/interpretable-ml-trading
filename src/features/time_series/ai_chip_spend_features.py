"""Epoch AI chip-sales quarterly spend, asof-joined onto bars (closed quarter)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.registry import register_feature

DEFAULT_QUARTERLY = "config/research/ai_chip_sales_quarterly.csv"


def load_ai_chip_sales_quarterly(
    path: str | Path = DEFAULT_QUARTERLY,
    *,
    drop_incomplete: bool = False,
) -> pd.DataFrame:
    """Pinned quarterly chip-cost tape.

    ``available_at`` is the first UTC day after ``end_date`` so the quarter
    is closed before any bar can read it.
    """
    raw = pd.read_csv(path, comment="#")
    out = pd.DataFrame(
        {
            "quarter": raw["quarter"].astype(str),
            "start_date": pd.to_datetime(raw["start_date"], utc=True),
            "end_date": pd.to_datetime(raw["end_date"], utc=True),
            "cost_usd": pd.to_numeric(raw["cost_usd"], errors="coerce"),
            "incomplete": pd.to_numeric(raw["incomplete"], errors="coerce")
            .fillna(0)
            .astype(int),
        }
    )
    if drop_incomplete:
        out = out.loc[out["incomplete"] == 0].copy()
    out["available_at"] = out["end_date"] + pd.Timedelta(days=1)
    out["qoq"] = out["cost_usd"].pct_change()
    expanding = out["qoq"].expanding(min_periods=4).median()
    out["qoq_high"] = (out["qoq"] > expanding).astype(float)
    out.loc[out["qoq"].isna() | expanding.isna(), "qoq_high"] = np.nan
    return out.sort_values("available_at").reset_index(drop=True)


@register_feature(
    "compute_ai_chip_spend_from_df",
    category="calendar",
    description=(
        "Last completed Epoch AI chip-sales quarter (cost USD and QoQ), "
        "asof-joined backward onto host bars."
    ),
    outputs=[
        "ai_chip_spend_usd",
        "ai_chip_spend_qoq",
        "ai_chip_spend_qoq_high",
    ],
)
def compute_ai_chip_spend_from_df(
    df: pd.DataFrame,
    *,
    quarterly_path: str = DEFAULT_QUARTERLY,
    drop_incomplete: bool = False,
    node_cache_version: str | None = None,
) -> pd.DataFrame:
    del node_cache_version
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df index must be DatetimeIndex")
    idx = (
        df.index.tz_localize("UTC")
        if df.index.tz is None
        else df.index.tz_convert("UTC")
    )
    tape = load_ai_chip_sales_quarterly(
        quarterly_path, drop_incomplete=bool(drop_incomplete)
    )
    left = pd.DataFrame({"_ts": idx, "_i": np.arange(len(idx), dtype=int)})
    right = pd.DataFrame(
        {
            "_ts": tape["available_at"],
            "ai_chip_spend_usd": tape["cost_usd"],
            "ai_chip_spend_qoq": tape["qoq"],
            "ai_chip_spend_qoq_high": tape["qoq_high"],
        }
    ).dropna(subset=["_ts"])
    merged = pd.merge_asof(
        left.sort_values("_ts"),
        right.sort_values("_ts"),
        on="_ts",
        direction="backward",
        allow_exact_matches=True,
    )
    out = pd.DataFrame(index=df.index)
    for col in (
        "ai_chip_spend_usd",
        "ai_chip_spend_qoq",
        "ai_chip_spend_qoq_high",
    ):
        vals = np.full(len(idx), np.nan, dtype=float)
        vals[merged["_i"].to_numpy()] = merged[col].to_numpy()
        out[col] = vals
    return out
