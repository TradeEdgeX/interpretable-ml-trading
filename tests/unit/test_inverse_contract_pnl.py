"""Inverse-contract formula — decoupled from the public court."""

from src.research.inverse_contract_pnl import (
    CoinMarginState,
    coin_m_contracts_from_risk_coin,
    coin_m_fee_collateral,
    coin_m_gross_pnl_collateral,
    coin_m_gross_pnl_usd,
    grid_gross_pnl_pct,
    is_long_side,
)


def test_is_long_side() -> None:
    assert is_long_side("LONG")
    assert is_long_side("buy")
    assert not is_long_side("SHORT")


def test_long_inverse_pnl_matches_binance_identity() -> None:
    # 1 / 100 − 1 / 110 = 0.000909...; × 10 contracts × 100 = 0.0909 coin
    coin = coin_m_gross_pnl_collateral(
        contracts=10.0,
        entry_price=100.0,
        exit_price=110.0,
        side="LONG",
        contract_multiplier=100.0,
    )
    assert abs(coin - 10.0 * 100.0 * (1.0 / 100.0 - 1.0 / 110.0)) < 1e-12
    usd = coin_m_gross_pnl_usd(
        contracts=10.0,
        entry_price=100.0,
        exit_price=110.0,
        side="LONG",
        contract_multiplier=100.0,
    )
    assert abs(usd - coin * 110.0) < 1e-12


def test_short_inverse_pnl_is_opposite_sign() -> None:
    long_coin = coin_m_gross_pnl_collateral(
        contracts=5.0, entry_price=200.0, exit_price=180.0, side="LONG"
    )
    short_coin = coin_m_gross_pnl_collateral(
        contracts=5.0, entry_price=200.0, exit_price=180.0, side="SHORT"
    )
    assert long_coin < 0
    assert abs(short_coin + long_coin) < 1e-12


def test_fee_and_sizing_helpers() -> None:
    fee = coin_m_fee_collateral(
        contracts=10.0, fill_price=100.0, fee_rate=0.0005, contract_multiplier=100.0
    )
    assert abs(fee - 0.005) < 1e-12
    n = coin_m_contracts_from_risk_coin(
        risk_coin=0.01, entry_price=100.0, stop_pct=0.05, contract_multiplier=100.0
    )
    assert abs(n - 0.2) < 1e-12


def test_wallet_marks_to_usdt() -> None:
    st = CoinMarginState(wallet_coin=2.0, contract_multiplier=100.0)
    assert abs(st.equity_usdt(50_000.0) - 100_000.0) < 1e-9
    st.apply_realized(-0.25)
    assert abs(st.wallet_coin - 1.75) < 1e-12


def test_grid_pct_linear_vs_inverse() -> None:
    lin = grid_gross_pnl_pct(entry=100.0, exit_px=110.0, side="LONG", margin_mode="usd_m")
    inv = grid_gross_pnl_pct(entry=100.0, exit_px=110.0, side="LONG", margin_mode="coin_m")
    assert abs(lin - 0.10) < 1e-12
    assert abs(inv - (10.0 / 110.0)) < 1e-12
