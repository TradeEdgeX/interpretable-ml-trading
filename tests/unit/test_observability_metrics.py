from src.time_series_model.live.observability_metrics import (
    compute_evidence_true_rate,
    compute_feature_missing_rate,
    compute_router_mode_entropy,
    compute_tick_gap_seconds,
)


def test_tick_gap_seconds_none_when_no_tick() -> None:
    assert compute_tick_gap_seconds(now_ns=10, last_tick_ts_ns=None) is None


def test_tick_gap_seconds_positive() -> None:
    assert (
        compute_tick_gap_seconds(now_ns=2_000_000_000, last_tick_ts_ns=1_000_000_000)
        == 1.0
    )


def test_feature_missing_rate() -> None:
    r = compute_feature_missing_rate(required_keys=["a", "b", "c"], features={"a": 1})
    assert r == 2 / 3


def test_evidence_true_rate() -> None:
    r = compute_evidence_true_rate({"x": True, "y": False, "z": True})
    assert r == 2 / 3


def test_router_mode_entropy_nonzero() -> None:
    e = compute_router_mode_entropy(["NO_TRADE", "TREND"])
    assert e is not None
    assert e > 0
