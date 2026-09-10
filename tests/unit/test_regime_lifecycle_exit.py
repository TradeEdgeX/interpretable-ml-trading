from src.time_series_model.live.position_logic import (
    regime_lifecycle_risk_off_exit,
    update_regime_lifecycle_state,
)


def _pos(cfg=None):
    return {
        "structural_exit": "abc_macro_regime_lifecycle",
        "regime_lifecycle_exit": cfg
        or {
            "bull_min_score": 4.0,
            "risk_off_drop_min": 1.0,
            "risk_off_floor_score": 3.0,
        },
        "_regime_saw_bull": False,
        "_regime_peak_score": 0.0,
    }


def test_no_exit_before_bull():
    pos = _pos()
    update_regime_lifecycle_state(pos, macro_regime_score=3.0)
    assert regime_lifecycle_risk_off_exit(pos, macro_regime_score=1.0) is None


def test_exit_on_drop_from_bull_peak():
    pos = _pos()
    for sc in [1.0, 2.0, 4.0, 5.0]:
        update_regime_lifecycle_state(pos, macro_regime_score=sc)
    reason = regime_lifecycle_risk_off_exit(pos, macro_regime_score=3.0)
    assert reason == "structural_exit_abc_macro_regime_risk_off"


def test_arm_risk_off_requires_peak_before_exit():
    pos = _pos(
        {
            "bull_min_score": 4.0,
            "risk_off_drop_min": 1.0,
            "risk_off_floor_score": 3.0,
            "arm_risk_off_min_peak": 5.0,
        }
    )
    update_regime_lifecycle_state(pos, macro_regime_score=4.0)
    assert pos["_regime_saw_bull"]
    assert regime_lifecycle_risk_off_exit(pos, macro_regime_score=2.0) is None
    update_regime_lifecycle_state(pos, macro_regime_score=5.0)
    assert regime_lifecycle_risk_off_exit(pos, macro_regime_score=3.0) is not None


def test_2024_style_crash_peak4_no_static5():
    pos = _pos()
    update_regime_lifecycle_state(pos, macro_regime_score=4.0)
    assert pos["_regime_saw_bull"]
    reason = regime_lifecycle_risk_off_exit(pos, macro_regime_score=2.0)
    assert reason == "structural_exit_abc_macro_regime_risk_off"


def test_regime_risk_off_can_be_disabled_for_spot_accum_a_layer():
    pos = _pos(
        {
            "bull_min_score": 4.0,
            "risk_off_drop_min": 1.0,
            "risk_off_floor_score": 3.0,
            "allow_regime_risk_off": False,
        }
    )
    for sc in [1.0, 2.0, 4.0, 5.0]:
        update_regime_lifecycle_state(pos, macro_regime_score=sc)
    assert regime_lifecycle_risk_off_exit(pos, macro_regime_score=2.0) is None
