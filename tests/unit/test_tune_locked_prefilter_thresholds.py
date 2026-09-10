#!/usr/bin/env python3

from scripts.tune_locked_prefilter_thresholds import aggregate_case, parse_float_list
from scripts.locked_prefilter_utils import apply_locked_thresholds


def test_parse_float_list():
    assert parse_float_list("0.1, 0.2,0.3") == [0.1, 0.2, 0.3]


def test_apply_locked_thresholds_updates_expected_rules():
    raw = {
        "rules": [
            {
                "feature": "fer_signed_efficiency_pct",
                "operator": ">=",
                "value": 0.0,
                "locked": True,
            },
            {
                "feature": "fer_signed_efficiency_pct",
                "operator": "<=",
                "value": 0.35,
                "locked": True,
            },
            {
                "feature": "sr_strength_max",
                "operator": ">=",
                "value": 0.55,
                "locked": True,
            },
            {
                "feature": "dist_to_nearest_sr",
                "operator": ">=",
                "value": -1.2,
                "locked": True,
            },
            {
                "feature": "dist_to_nearest_sr",
                "operator": "<=",
                "value": 1.2,
                "locked": True,
            },
        ]
    }
    out = apply_locked_thresholds(
        raw,
        template="bindings",
        params={
            "fer_signed_efficiency_pct_min": 0.05,
            "fer_signed_efficiency_pct_max": 0.4,
            "sr_strength_max_min": 0.6,
            "dist_to_nearest_sr_min": -1.0,
            "dist_to_nearest_sr_max": 1.0,
        },
    )
    rules = out["rules"]
    assert rules[0]["value"] == 0.05
    assert rules[1]["value"] == 0.4
    assert rules[2]["value"] == 0.6
    assert rules[3]["value"] == -1.0
    assert rules[4]["value"] == 1.0


def test_aggregate_case_with_trade_penalty():
    case_results = [
        {"metrics": {"sharpe_per_trade": 0.4, "total_trades": 30}},
        {"metrics": {"sharpe_per_trade": 0.2, "total_trades": 50}},
        {"metrics": {"sharpe_per_trade": -0.1, "total_trades": 70}},
    ]
    agg = aggregate_case(
        case_results,
        target_trades_min=60,
        target_trades_max=10_000,
        trade_penalty_low=0.01,
        trade_penalty_high=0.0,
        stability_penalty=0.0,
    )
    assert round(agg["median_sharpe"], 6) == 0.2
    assert round(agg["positive_ratio"], 6) == round(2 / 3, 6)
    assert agg["median_trades"] == 50.0
    # score = 0.2 - 0.01 * low_gap, low_gap = max(0, 60 - 50)
    assert round(agg["score"], 6) == 0.1


def test_apply_locked_thresholds_updates_me_rules():
    raw = {
        "rules": [
            {
                "feature": "atr_percentile",
                "operator": ">=",
                "value": 0.2,
                "locked": True,
            },
            {
                "feature": "atr_percentile",
                "operator": "<=",
                "value": 0.88,
                "locked": True,
            },
            {
                "feature": "compression_duration",
                "operator": ">=",
                "value": 0.03,
                "locked": True,
            },
            {
                "feature": "recent_compression_decay",
                "operator": "<=",
                "value": 0.35,
                "locked": True,
            },
            {
                "feature": "oi_compression_score",
                "operator": ">=",
                "value": 0.35,
                "locked": True,
            },
        ]
    }
    out = apply_locked_thresholds(
        raw,
        atr_lower=0.25,
        atr_upper=0.9,
        compression_min=0.05,
        decay_upper=0.2,
        oi_min=0.4,
        template="me",
    )
    rules = out["rules"]
    assert rules[0]["value"] == 0.25
    assert rules[1]["value"] == 0.9
    assert rules[2]["value"] == 0.05
    assert rules[3]["value"] == 0.2
    assert rules[4]["value"] == 0.4


def test_apply_locked_thresholds_updates_bpc_rules():
    raw = {
        "rules": [
            {
                "feature": "bpc_recent_breakout_strength",
                "operator": ">=",
                "value": 0.45,
                "locked": True,
            },
            {
                "feature": "bpc_pullback_depth",
                "operator": "<=",
                "value": 0.7,
                "locked": True,
            },
            {
                "feature": "bpc_recovery_strength",
                "operator": ">=",
                "value": 0.55,
                "locked": True,
            },
        ]
    }
    out = apply_locked_thresholds(
        raw,
        template="bpc",
        params={
            "bpc_recent_breakout_strength_min": 0.5,
            "bpc_pullback_depth_max": 0.6,
            "bpc_recovery_strength_min": 0.6,
        },
    )
    rules = out["rules"]
    assert rules[0]["value"] == 0.5
    assert rules[1]["value"] == 0.6
    assert rules[2]["value"] == 0.6
