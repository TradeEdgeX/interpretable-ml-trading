import pytest

from src.time_series_model.portfolio.slot_sizing import (
    compute_slot_size_from_risk,
    estimate_stop_return_frac,
    risk_only_down,
)


@pytest.mark.unit
def test_estimate_stop_return_frac():
    # price=100, atr=2 => atr_pct=0.02; stop_atr=1.5 => 0.03
    assert estimate_stop_return_frac(
        price=100.0, atr=2.0, stop_atr=1.5
    ) == pytest.approx(0.03)


@pytest.mark.unit
def test_compute_slot_size_risk_limited_and_leverage_capped():
    # equity=10k, risk=1%, risk_usd=100
    # price=100, atr=2, stop_atr=1 => stop_ret=0.02
    # risk-limited notional=100/0.02=5000 => qty=50
    out = compute_slot_size_from_risk(
        equity_usd=10_000,
        risk_frac=0.01,
        price=100.0,
        atr=2.0,
        stop_atr=1.0,
        max_leverage=10.0,
    )
    assert out.stop_return_frac == pytest.approx(0.02)
    assert out.notional_usd == pytest.approx(5000.0)
    assert out.qty == pytest.approx(50.0)

    # Now cap leverage hard: max_leverage=0.2 => notional cap=2000 => qty=20
    out2 = compute_slot_size_from_risk(
        equity_usd=10_000,
        risk_frac=0.01,
        price=100.0,
        atr=2.0,
        stop_atr=1.0,
        max_leverage=0.2,
    )
    assert out2.notional_usd == pytest.approx(2000.0)
    assert out2.qty == pytest.approx(20.0)


@pytest.mark.unit
def test_compute_slot_size_bad_inputs_return_zero():
    out = compute_slot_size_from_risk(
        equity_usd=10_000,
        risk_frac=0.01,
        price=0.0,
        atr=2.0,
        stop_atr=1.0,
    )
    assert out.qty == 0.0
    assert out.notional_usd == 0.0


@pytest.mark.unit
def test_risk_only_down():
    assert risk_only_down(
        prev_risk_frac=None, proposed_risk_frac=0.02
    ) == pytest.approx(0.02)
    assert risk_only_down(
        prev_risk_frac=0.015, proposed_risk_frac=0.02
    ) == pytest.approx(0.015)
    assert risk_only_down(
        prev_risk_frac=0.015, proposed_risk_frac=0.01
    ) == pytest.approx(0.01)


@pytest.mark.unit
def test_compute_slot_size_without_reflexivity_features():
    """Test backward compatibility: behavior unchanged when reflexivity_features is None"""
    out = compute_slot_size_from_risk(
        equity_usd=10_000,
        risk_frac=0.01,
        price=100.0,
        atr=2.0,
        stop_atr=1.0,
        max_leverage=10.0,
        reflexivity_features=None,
    )
    assert out.notional_usd == pytest.approx(5000.0)
    assert out.qty == pytest.approx(50.0)


@pytest.mark.unit
def test_compute_slot_size_with_reflexivity_soft_veto():
    """Test OFCI soft veto: position should be reduced by multiplier"""
    # Base case: no reflexivity risk
    out_base = compute_slot_size_from_risk(
        equity_usd=10_000,
        risk_frac=0.01,
        price=100.0,
        atr=2.0,
        stop_atr=1.0,
        max_leverage=10.0,
        reflexivity_features={"ofci_pct": 0.5, "shd_pct": 0.5, "lfi_pct": 0.5},
    )
    assert out_base.notional_usd == pytest.approx(5000.0)
    assert out_base.qty == pytest.approx(50.0)

    # OFCI soft veto: ofci_pct > 0.9 should apply 0.6 multiplier
    out_ofci = compute_slot_size_from_risk(
        equity_usd=10_000,
        risk_frac=0.01,
        price=100.0,
        atr=2.0,
        stop_atr=1.0,
        max_leverage=10.0,
        reflexivity_features={"ofci_pct": 0.95, "shd_pct": 0.5, "lfi_pct": 0.5},
    )
    # Expected: 5000 * 0.6 = 3000
    assert out_ofci.notional_usd == pytest.approx(3000.0)
    assert out_ofci.qty == pytest.approx(30.0)


@pytest.mark.unit
def test_compute_slot_size_with_reflexivity_hard_veto():
    """Test SHD hard veto: should return 0 position"""
    # SHD hard veto: shd_pct > 0.9 should return 0
    out_shd = compute_slot_size_from_risk(
        equity_usd=10_000,
        risk_frac=0.01,
        price=100.0,
        atr=2.0,
        stop_atr=1.0,
        max_leverage=10.0,
        reflexivity_features={"ofci_pct": 0.5, "shd_pct": 0.95, "lfi_pct": 0.5},
    )
    assert out_shd.qty == 0.0
    assert out_shd.notional_usd == 0.0


@pytest.mark.unit
def test_compute_slot_size_with_reflexivity_lfi_soft_veto():
    """Test LFI soft veto: position should be reduced by 0.3 multiplier"""
    # LFI soft veto: lfi_pct > 0.9 should apply 0.3 multiplier
    out_lfi = compute_slot_size_from_risk(
        equity_usd=10_000,
        risk_frac=0.01,
        price=100.0,
        atr=2.0,
        stop_atr=1.0,
        max_leverage=10.0,
        reflexivity_features={"ofci_pct": 0.5, "shd_pct": 0.5, "lfi_pct": 0.95},
    )
    # Expected: 5000 * 0.3 = 1500
    assert out_lfi.notional_usd == pytest.approx(1500.0)
    assert out_lfi.qty == pytest.approx(15.0)


@pytest.mark.unit
def test_compute_slot_size_backward_compatible():
    """Test that not providing reflexivity_features maintains backward compatibility"""
    out = compute_slot_size_from_risk(
        equity_usd=10_000,
        risk_frac=0.01,
        price=100.0,
        atr=2.0,
        stop_atr=1.0,
        max_leverage=10.0,
    )
    # Should behave exactly as before
    assert out.notional_usd == pytest.approx(5000.0)
    assert out.qty == pytest.approx(50.0)


@pytest.mark.unit
def test_compute_slot_size_ofci_soft_veto():
    """Test that OFCI soft veto applies 0.6 multiplier"""
    # Base case: no reflexivity risk
    out_base = compute_slot_size_from_risk(
        equity_usd=10_000,
        risk_frac=0.01,
        price=100.0,
        atr=2.0,
        stop_atr=1.0,
        max_leverage=10.0,
        reflexivity_features={"ofci_pct": 0.5, "shd_pct": 0.5, "lfi_pct": 0.5},
    )
    assert out_base.notional_usd == pytest.approx(5000.0)
    assert out_base.qty == pytest.approx(50.0)

    # OFCI soft veto: ofci_pct > 0.9 should apply 0.6 multiplier
    out_ofci = compute_slot_size_from_risk(
        equity_usd=10_000,
        risk_frac=0.01,
        price=100.0,
        atr=2.0,
        stop_atr=1.0,
        max_leverage=10.0,
        reflexivity_features={"ofci_pct": 0.95, "shd_pct": 0.5, "lfi_pct": 0.5},
    )
    # 5000 * 0.6 = 3000
    assert out_ofci.notional_usd == pytest.approx(3000.0)
    assert out_ofci.qty == pytest.approx(30.0)


@pytest.mark.unit
def test_compute_slot_size_shd_hard_veto():
    """Test that SHD hard veto returns 0 position"""
    out = compute_slot_size_from_risk(
        equity_usd=10_000,
        risk_frac=0.01,
        price=100.0,
        atr=2.0,
        stop_atr=1.0,
        max_leverage=10.0,
        reflexivity_features={"ofci_pct": 0.5, "shd_pct": 0.95, "lfi_pct": 0.5},
    )
    # Hard veto: should return 0
    assert out.qty == 0.0
    assert out.notional_usd == 0.0


@pytest.mark.unit
def test_compute_slot_size_lfi_soft_veto():
    """Test that LFI soft veto applies 0.3 multiplier"""
    # LFI soft veto: lfi_pct > 0.9 should apply 0.3 multiplier
    out_lfi = compute_slot_size_from_risk(
        equity_usd=10_000,
        risk_frac=0.01,
        price=100.0,
        atr=2.0,
        stop_atr=1.0,
        max_leverage=10.0,
        reflexivity_features={"ofci_pct": 0.5, "shd_pct": 0.5, "lfi_pct": 0.95},
    )
    # 5000 * 0.3 = 1500
    assert out_lfi.notional_usd == pytest.approx(1500.0)
    assert out_lfi.qty == pytest.approx(15.0)
