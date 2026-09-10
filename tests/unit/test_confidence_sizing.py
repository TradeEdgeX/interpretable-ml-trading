"""confidence_sizing: feature thresholds → size scale (entry set unchanged)."""

from __future__ import annotations

from src.time_series_model.execution.confidence_sizing import (
    resolve_confidence_size_scale,
)
from src.time_series_model.live.generic_live_strategy import ExecutionParamGenerator


def test_disabled_returns_none():
    assert resolve_confidence_size_scale({"enabled": False}, {"x": 1}) is None
    assert resolve_confidence_size_scale(None, {}) is None


def test_all_mode_high_vs_low():
    cfg = {
        "enabled": True,
        "high_size_multiplier": 1.0,
        "low_size_multiplier": 0.5,
        "high_when": {
            "mode": "all",
            "conditions": [
                {
                    "feature": "volume_participation_score",
                    "operator": ">=",
                    "value": 0.45,
                },
                {
                    "feature": "ema_1200_position",
                    "abs": True,
                    "operator": ">=",
                    "value": 0.18,
                },
            ],
        },
    }
    assert (
        resolve_confidence_size_scale(
            cfg, {"volume_participation_score": 0.5, "ema_1200_position": -0.2}
        )
        == 1.0
    )
    assert (
        resolve_confidence_size_scale(
            cfg, {"volume_participation_score": 0.5, "ema_1200_position": -0.1}
        )
        == 0.5
    )
    assert (
        resolve_confidence_size_scale(
            cfg, {"volume_participation_score": 0.2, "ema_1200_position": -0.2}
        )
        == 0.5
    )


def test_any_mode_and_missing_feature_fail_closed_for_all():
    any_cfg = {
        "enabled": True,
        "high_when": {
            "mode": "any",
            "conditions": [
                {
                    "feature": "volume_participation_score",
                    "operator": ">=",
                    "value": 0.45,
                },
                {
                    "feature": "ema_1200_position",
                    "abs": True,
                    "operator": ">=",
                    "value": 0.18,
                },
            ],
        },
    }
    assert (
        resolve_confidence_size_scale(
            any_cfg, {"volume_participation_score": 0.5, "ema_1200_position": 0.0}
        )
        == 1.0
    )
    all_cfg = {**any_cfg, "high_when": {**any_cfg["high_when"], "mode": "all"}}
    # missing vol → not high under all
    assert resolve_confidence_size_scale(all_cfg, {"ema_1200_position": 0.3}) == 0.5


def test_execution_param_generator_applies_scale():
    raw = {
        "stop_loss": {
            "initial_r": 2.0,
            "type": "trailing",
            "trailing": {"enabled": False},
        },
        "holding": {"time_stop_bars": 0},
        "confidence_sizing": {
            "enabled": True,
            "high_size_multiplier": 1.0,
            "low_size_multiplier": 0.5,
            "high_when": {
                "mode": "all",
                "conditions": [
                    {
                        "feature": "volume_participation_score",
                        "operator": ">=",
                        "value": 0.45,
                    }
                ],
            },
        },
    }
    gen = ExecutionParamGenerator(raw)
    hi = gen.generate_params(0.5, features={"volume_participation_score": 0.9})
    lo = gen.generate_params(0.5, features={"volume_participation_score": 0.1})
    assert hi["size_multiplier"] == 1.0
    assert lo["size_multiplier"] == 0.5
