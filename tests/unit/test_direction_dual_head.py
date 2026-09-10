"""dual_head direction.yaml runtime (score_long / score_short)."""

from __future__ import annotations

from src.time_series_model.live.generic_live_strategy import DirectionEvaluator


def test_dual_head_long_only_fires() -> None:
    ev = DirectionEvaluator(
        {
            "dual_head": {
                "enabled": True,
                "long": {"score_column": "score_long", "entry_threshold": 0.6},
                "short": {"score_column": "score_short", "entry_threshold": 0.7},
                "reject_if_both_high": True,
            },
            "thresholds": {"entry_mode": "level"},
        }
    )
    direction, rule = ev.evaluate(
        {"score_long": 0.65, "score_short": 0.2}, symbol="BTCUSDT"
    )
    assert direction == 1
    assert rule == "dual_head_long"


def test_dual_head_rejects_both_high() -> None:
    ev = DirectionEvaluator(
        {
            "dual_head": {
                "enabled": True,
                "long": {"score_column": "score_long", "entry_threshold": 0.5},
                "short": {"score_column": "score_short", "entry_threshold": 0.5},
                "reject_if_both_high": True,
            },
            "thresholds": {"entry_mode": "level"},
        }
    )
    direction, rule = ev.evaluate(
        {"score_long": 0.8, "score_short": 0.8}, symbol="BTCUSDT"
    )
    assert direction == 0
    assert rule == "dual_head_both_high"
