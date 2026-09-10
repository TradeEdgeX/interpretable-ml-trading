from auxiliary.research.ashare_pool_detector_uplift import SlotPolicy


def test_px_turn_exit_fields() -> None:
    p = SlotPolicy.from_kwargs(px_turn_exit="high_wet", px_turn_high_ret=0.15)
    assert p.px_turn_exit == "high_wet"
    assert p.px_turn_high_ret == 0.15
    assert SlotPolicy.from_kwargs().px_turn_exit is None
