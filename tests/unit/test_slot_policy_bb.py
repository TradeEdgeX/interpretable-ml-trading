"""Bollinger overlay fields on SlotPolicy (20260819 research)."""

from auxiliary.research.ashare_pool_detector_uplift import SlotPolicy


def test_from_kwargs_keeps_bb_fields() -> None:
    policy = SlotPolicy.from_kwargs(
        bb_exit="upper", bb_reenter="mid", bb_partial=True, bb_k=2.0
    )
    assert policy.bb_exit == "upper"
    assert policy.bb_reenter == "mid"
    assert policy.bb_partial is True
    assert policy.bb_window == 20


def test_default_has_no_bb_exit() -> None:
    policy = SlotPolicy.from_kwargs()
    assert policy.bb_exit is None
    assert policy.bb_reenter is None
    assert policy.bb_partial is False
