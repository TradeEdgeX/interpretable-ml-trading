"""DirectionEvaluator: signal_match_position_band compound rule."""

from src.time_series_model.live.generic_live_strategy import DirectionEvaluator


def test_signal_match_position_band_long_aligned():
    ev = DirectionEvaluator(
        {
            "direction_rules": [
                {
                    "id": "combo",
                    "method": "signal_match_position_band",
                    "signal_rules": [
                        {"feature": "bpc_breakout_direction", "transform": "raw"},
                        {"feature": "macd_atr", "transform": "sign"},
                    ],
                    "position_band": {
                        "feature": "macro_tp_vwap_1200_position",
                        "inner_abs": 0.005,
                        "outer_abs": 0.95,
                    },
                }
            ]
        }
    )
    d, rid = ev.evaluate(
        {
            "bpc_breakout_direction": 1.0,
            "macd_atr": 0.0,
            "macro_tp_vwap_1200_position": 0.02,
        }
    )
    assert d == 1
    assert rid == "combo"


def test_signal_match_position_band_rejects_mismatch_then_fallback():
    ev = DirectionEvaluator(
        {
            "direction_rules": [
                {
                    "id": "combo",
                    "method": "signal_match_position_band",
                    "signal_rules": [
                        {"feature": "bpc_breakout_direction", "transform": "raw"},
                    ],
                    "position_band": {
                        "feature": "macro_tp_vwap_1200_position",
                        "inner_abs": 0.005,
                        "outer_abs": 0.95,
                    },
                },
                {"id": "fallback", "feature": "always_long", "transform": "raw"},
            ]
        }
    )
    d, rid = ev.evaluate(
        {
            "bpc_breakout_direction": 1.0,
            "macro_tp_vwap_1200_position": -0.02,
            "always_long": 1.0,
        }
    )
    assert d == 1
    assert rid == "fallback"


def test_signal_match_position_band_short_aligned():
    ev = DirectionEvaluator(
        {
            "direction_rules": [
                {
                    "id": "combo",
                    "method": "signal_match_position_band",
                    "signal_rules": [
                        {"feature": "bpc_breakout_direction", "transform": "raw"},
                    ],
                    "position_band": {
                        "feature": "macro_tp_vwap_1200_position",
                        "inner_abs": 0.005,
                        "outer_abs": 0.95,
                    },
                }
            ]
        }
    )
    d, rid = ev.evaluate(
        {
            "bpc_breakout_direction": -1.0,
            "macro_tp_vwap_1200_position": -0.03,
        }
    )
    assert d == -1
    assert rid == "combo"


def test_invert_side_flips_compound_direction():
    cfg = {
        "invert_side": True,
        "direction_rules": [
            {
                "id": "combo",
                "method": "signal_match_position_band",
                "signal_rules": [
                    {"feature": "bpc_breakout_direction", "transform": "raw"},
                ],
                "position_band": {
                    "feature": "macro_tp_vwap_1200_position",
                    "inner_abs": 0.005,
                    "outer_abs": 0.95,
                },
            }
        ],
    }
    ev = DirectionEvaluator(cfg)
    d, rid = ev.evaluate(
        {
            "bpc_breakout_direction": 1.0,
            "macro_tp_vwap_1200_position": 0.02,
        }
    )
    assert d == -1
    assert rid == "combo|invert_side"
