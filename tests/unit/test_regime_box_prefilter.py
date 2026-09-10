from __future__ import annotations

from src.time_series_model.live.regime_box_prefilter import (
    is_stable_box_bar,
    regime_box_window,
    stable_box_blocks_chop_entry,
    stable_box_blocks_trend_entry,
)


def test_regime_box_window_from_rules_box_pos_60() -> None:
    rules = [
        {
            "all_of": [
                {"feature": "box_pos_60", "operator": ">=", "value": 0.35},
            ]
        }
    ]
    assert regime_box_window({}, rules) == 60


def test_is_stable_box_from_atomic_columns() -> None:
    box = {
        "stability_min": 0.85,
        "width_min": 0.04,
        "width_max": 0.30,
        "touches_min": 5,
    }
    feats = {
        "box_stability_60": 0.90,
        "box_width_pct_60": 0.06,
        "box_touches_hi_60": 7,
        "box_touches_lo_60": 6,
    }
    assert is_stable_box_bar(feats, box, box_window=60)


def test_chop_blocks_when_exclude_box_false_and_box_ok() -> None:
    regime = {
        "exclude_box_prefilter": False,
        "box_prefilter": {
            "stability_min": 0.85,
            "width_min": 0.04,
            "width_max": 0.30,
            "touches_min": 5,
        },
    }
    feats = {
        "bpc_semantic_chop": 0.8,
        "box_stability_60": 0.90,
        "box_width_pct_60": 0.06,
        "box_touches_hi_60": 7,
        "box_touches_lo_60": 6,
    }
    rules = [{"all_of": [{"feature": "box_pos_60", "operator": ">=", "value": 0.35}]}]
    assert stable_box_blocks_chop_entry(feats, regime, rules=rules)


def test_chop_does_not_block_when_exclude_box_true() -> None:
    regime = {
        "exclude_box_prefilter": True,
        "box_prefilter": {"stability_min": 0.85, "touches_min": 5},
    }
    feats = {
        "box_stability_60": 0.99,
        "box_width_pct_60": 0.05,
        "box_touches_hi_60": 10,
        "box_touches_lo_60": 10,
    }
    assert not stable_box_blocks_chop_entry(feats, regime)


def test_trend_blocks_when_exclude_box_and_stable() -> None:
    regime = {
        "exclude_box_prefilter": True,
        "box_prefilter": {
            "stability_min": 0.85,
            "width_min": 0.04,
            "width_max": 0.30,
            "touches_min": 5,
        },
    }
    rules = [{"all_of": [{"feature": "box_pos_60", "operator": ">=", "value": 0.35}]}]
    feats = {
        "box_stability_60": 0.90,
        "box_width_pct_60": 0.06,
        "box_touches_hi_60": 7,
        "box_touches_lo_60": 6,
    }
    assert stable_box_blocks_trend_entry(feats, regime, rules=rules)
