"""Smoke tests for quick_layer_scan modes and --bucket-by."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.quick_layer_scan import (
    _bucketed_report,
    _parse_bucket_by,
    _resolve_bucket_masks,
    mode_condition_set,
    mode_ic_decay,
)


def _sample_df(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dt = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    ema = rng.uniform(-0.2, 0.2, size=n)
    vol = rng.uniform(0, 1, size=n)
    success = (ema > 0).astype(int)
    return pd.DataFrame(
        {
            "datetime": dt,
            "ema_1200_position": ema,
            "vol_persistence": vol,
            "success_no_rr_extreme": success,
            "forward_rr": rng.normal(0, 0.5, size=n),
        }
    )


def test_parse_bucket_by_ema() -> None:
    buckets = _parse_bucket_by("ema:ema_1200_position@0.10")
    assert len(buckets) == 2
    df = _sample_df()
    ge = buckets[0][1](df)
    lt = buckets[1][1](df)
    assert (ge | lt).all()
    assert not (ge & lt).any()


def test_parse_bucket_by_calendar() -> None:
    buckets = _parse_bucket_by("calendar:2024-01-01,2024-01-10;2024-01-10,2024-01-20")
    assert len(buckets) == 2
    df = _sample_df()
    m0 = buckets[0][1](df)
    m1 = buckets[1][1](df)
    assert m0.sum() > 0
    assert m1.sum() > 0


def test_resolve_feature_quantile_buckets() -> None:
    df = _sample_df()
    buckets = _resolve_bucket_masks(df, "feature_quantile:vol_persistence@4")
    assert len(buckets) >= 2
    covered = pd.Series(False, index=df.index)
    for _, fn in buckets:
        covered = covered | fn(df)
    assert covered.sum() > len(df) * 0.9


def test_bucketed_condition_set_report(tmp_path: Path) -> None:
    df = _sample_df()
    label = df["success_no_rr_extreme"].astype(bool)
    base = pd.Series(True, index=df.index)
    args = type(
        "Args",
        (),
        {
            "mode": "condition-set",
            "condition": [
                "H: abs(ema_1200_position)>0.10",
                "L: abs(ema_1200_position)<=0.10",
            ],
        },
    )()
    md = _bucketed_report(args, df, label, base, "ema:ema_1200_position@0.10")
    assert "Bucket:" in md
    assert "condition_set" in md or "condition-set" in md


def test_ic_decay_smoke() -> None:
    df = _sample_df()
    args = type(
        "Args",
        (),
        {
            "features": "ema_1200_position,vol_persistence",
            "horizons": "1",
            "target": "forward_rr",
            "baseline_json": None,
        },
    )()
    md = mode_ic_decay(args, df, pd.Series(True, index=df.index))
    assert "IC decay" in md
    assert "ema_1200_position" in md


def test_condition_set_smoke() -> None:
    df = _sample_df()
    label = df["success_no_rr_extreme"].astype(bool)
    args = type(
        "Args",
        (),
        {
            "condition": ["H: abs(ema_1200_position)>0.10"],
        },
    )()
    md = mode_condition_set(args, df, label, pd.Series(True, index=df.index))
    assert "|z|" in md
