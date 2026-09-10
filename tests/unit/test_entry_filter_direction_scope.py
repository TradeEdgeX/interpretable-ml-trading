"""Entry filter direction scoping (long-only / short-only filters)."""

from __future__ import annotations

import pandas as pd

from src.time_series_model.execution.entry_filter import (
    apply_entry_filters_or,
    check_entry_filters_or_single,
    entry_filter_applies_to_direction,
)


def _anti_chase_cfg() -> dict:
    return {
        "combination_mode": "and",
        "filters": [
            {
                "id": "base",
                "enabled": True,
                "conditions": [{"feature": "x", "operator": ">=", "value": 0.5}],
            },
            {
                "id": "long_only",
                "enabled": True,
                "direction": "long",
                "conditions": [
                    {
                        "feature": "bars_since_local_high",
                        "operator": ">=",
                        "value": 0.10,
                    }
                ],
            },
            {
                "id": "short_only",
                "enabled": True,
                "direction": "short",
                "conditions": [
                    {"feature": "bars_since_local_low", "operator": ">=", "value": 0.10}
                ],
            },
        ],
    }


def test_entry_filter_applies_to_direction():
    long_f = {"direction": "long"}
    short_f = {"direction": "short"}
    assert entry_filter_applies_to_direction(long_f, 1) is True
    assert entry_filter_applies_to_direction(long_f, -1) is False
    assert entry_filter_applies_to_direction(short_f, -1) is True
    assert entry_filter_applies_to_direction(short_f, 1) is False
    assert entry_filter_applies_to_direction({}, 1) is True


def test_check_entry_filters_or_single_direction_scoped_and_mode():
    cfg = _anti_chase_cfg()
    base = {"x": 1.0, "bars_since_local_high": 0.05, "bars_since_local_low": 0.20}
    assert check_entry_filters_or_single(base, cfg, direction=1) is False
    base["bars_since_local_high"] = 0.15
    assert check_entry_filters_or_single(base, cfg, direction=1) is True
    short_at_low = {
        "x": 1.0,
        "bars_since_local_high": 0.99,
        "bars_since_local_low": 0.05,
    }
    assert check_entry_filters_or_single(short_at_low, cfg, direction=-1) is False
    short_at_low["bars_since_local_low"] = 0.15
    assert check_entry_filters_or_single(short_at_low, cfg, direction=-1) is True


def test_or_bundle_ids_vol_or_delta_and_anti_chase():
    cfg = {
        "or_bundle_ids": ["vol", "delta"],
        "filters": [
            {
                "id": "vol",
                "enabled": True,
                "conditions": [{"feature": "x", "operator": ">=", "value": 1.0}],
            },
            {
                "id": "delta",
                "enabled": True,
                "conditions": [{"feature": "y", "operator": ">=", "value": 1.0}],
            },
            {
                "id": "anti",
                "enabled": True,
                "direction": "long",
                "conditions": [
                    {
                        "feature": "bars_since_local_high",
                        "operator": ">=",
                        "value": 0.10,
                    }
                ],
            },
        ],
    }
    assert (
        check_entry_filters_or_single(
            {"x": 0.0, "y": 1.0, "bars_since_local_high": 0.05}, cfg, direction=1
        )
        is False
    )
    assert (
        check_entry_filters_or_single(
            {"x": 0.0, "y": 1.0, "bars_since_local_high": 0.15}, cfg, direction=1
        )
        is True
    )
    assert (
        check_entry_filters_or_single(
            {"x": 1.0, "y": 0.0, "bars_since_local_high": 0.05}, cfg, direction=1
        )
        is False
    )


def test_apply_entry_filters_or_direction_scoped_and_mode():
    cfg = _anti_chase_cfg()
    df = pd.DataFrame(
        {
            "entry_direction": [1.0, 1.0, -1.0, -1.0],
            "x": [1.0, 1.0, 1.0, 1.0],
            "bars_since_local_high": [0.05, 0.15, 0.99, 0.99],
            "bars_since_local_low": [0.99, 0.99, 0.05, 0.15],
        }
    )
    n = apply_entry_filters_or(df, cfg, silent=True)
    assert n == 2
    assert df["entry_direction"].tolist() == [0.0, 1.0, 0.0, -1.0]
