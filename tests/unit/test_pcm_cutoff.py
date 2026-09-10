"""PCM cutoff resolution (ADR §14.3 walk-forward vs static holdout)."""

from scripts.pipeline.calibration_window import calib_and_test_windows
from scripts.pipeline.config import iter_month_tokens
from scripts.pipeline.pcm_cutoff import resolve_pcm_cutoff_date


def test_calib_and_test_windows_june_2024_k3():
    w = calib_and_test_windows(month_token="2024-06", calibration_months=3)
    assert w["test_start"] == "2024-06-01"
    assert w["test_end"] == "2024-06-30"
    assert w["calib_end"] == "2024-05-31"
    assert w["calib_start"] == "2024-03-01"


def test_calib_and_test_windows_respects_step_months():
    w = calib_and_test_windows(
        month_token="2024-06", calibration_months=3, step_months=2
    )
    assert w["test_start"] == "2024-06-01"
    assert w["test_end"] == "2024-07-31"
    assert w["calib_end"] == "2024-05-31"
    assert w["calib_start"] == "2024-03-01"


def test_iter_month_tokens_respects_step_months():
    assert iter_month_tokens("2024-01-01", "2024-06-30", step_months=2) == [
        "2024-01",
        "2024-03",
        "2024-05",
    ]


def test_resolve_static_holdout_with_val_split():
    c = resolve_pcm_cutoff_date(
        "static_holdout",
        month_token="2024-06",
        calibration_months=3,
        holdout_start="2024-01-01",
        test_start="2024-04-01",
    )
    assert c == "2024-04-01"


def test_resolve_static_holdout_no_split_returns_none():
    assert (
        resolve_pcm_cutoff_date(
            "static_holdout",
            month_token="2024-06",
            calibration_months=3,
            holdout_start="2024-01-01",
            test_start="2024-01-01",
        )
        is None
    )


def test_resolve_walk_forward_uses_calib_end_not_global_test_start():
    c = resolve_pcm_cutoff_date(
        "walk_forward_monthly",
        month_token="2024-06",
        calibration_months=3,
        holdout_start="2024-01-01",
        test_start="2024-04-01",
    )
    assert c == "2024-05-31"


def test_resolve_walk_forward_without_month_falls_back_to_static():
    c = resolve_pcm_cutoff_date(
        "walk_forward_monthly",
        month_token=None,
        calibration_months=3,
        holdout_start="2024-01-01",
        test_start="2024-04-01",
    )
    assert c == "2024-04-01"
