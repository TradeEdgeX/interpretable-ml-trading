"""Time-abort overlay on SlotPolicy (20260819 research)."""

from auxiliary.research.ashare_pool_detector_uplift import SlotPolicy


def test_from_kwargs_keeps_abort_fields() -> None:
    policy = SlotPolicy.from_kwargs(abort_hold_days=5, abort_max_ret=0.02)
    assert policy.abort_hold_days == 5
    assert policy.abort_max_ret == 0.02


def test_default_has_no_abort() -> None:
    policy = SlotPolicy.from_kwargs()
    assert policy.abort_hold_days is None
    assert policy.abort_max_ret == 0.0
