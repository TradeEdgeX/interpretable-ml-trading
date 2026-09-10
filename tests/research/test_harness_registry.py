"""Public extract: only ma_cross is a registered event_backtest family."""

from src.research.experiment_index import validate_meta
from src.research.harness_registry import (
    event_backtest_reject_reason,
    harness_mismatch_issue,
    required_harness,
    spec_for,
)


def test_ma_cross_uses_event_backtest() -> None:
    assert required_harness("ma_cross") == "event_backtest"
    assert event_backtest_reject_reason(["ma_cross"]) is None
    assert spec_for("ma") is spec_for("ma_cross")


def test_private_families_are_unregistered() -> None:
    for name in ("srb", "tpc", "bpc", "rolling_trend", "chop_grid", "ashare_oversold"):
        assert spec_for(name) is None
        assert event_backtest_reject_reason([name]) is None


def test_unknown_family_is_not_blocked() -> None:
    assert spec_for("lottery100") is None
    assert event_backtest_reject_reason(["lottery100"]) is None


def test_front_matter_matching_harness_ok() -> None:
    assert harness_mismatch_issue("ma_cross", "event_backtest") is None


def test_harness_cli_check_ma_cross() -> None:
    from scripts.research.harness import main

    assert main(["ma_cross", "--check-event-backtest"]) == 0


def test_validate_meta_unknown_family_has_no_mismatch() -> None:
    meta = validate_meta(
        {
            "verdict": "promote",
            "strategy": "rolling_trend",
            "harness": "event_backtest",
            "segments": ["bear_2022"],
            "kill_switch": False,
        },
        known_segments={"bear_2022"},
    )
    assert not any("does not match family" in i for i in meta.issues)
