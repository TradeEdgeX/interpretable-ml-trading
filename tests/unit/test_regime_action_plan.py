"""Regime action plan + coin_sleeve ledger scope."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mlbot_console.services.exchange_balances import (
    _fetch_coin_sleeve_equity,
    fetch_scope_exchange_balance,
)
from pathlib import Path

from mlbot_console.services.rebalance_advisor import (
    build_allocation,
    build_regime_action_plan,
    build_suggestions,
    load_rebalance_config,
)


def test_transfer_plan_caps_into_spot_band_max() -> None:
    """Independent 减仓 sums must not imply dumping past Spot max band."""
    cfg = load_rebalance_config(Path(__file__).resolve().parents[2])
    alloc = build_allocation(
        ledger={
            "accounts": [
                {"scope": "spot", "ok": True, "equity_usdt": 6968.0},
                {"scope": "rolling", "ok": True, "equity_usdt": 3008.0},
                {"scope": "trend", "ok": True, "equity_usdt": 12192.0},
            ]
        },
        composite_label="risk_off",
        config=cfg,
    )
    active = [
        s for s in alloc["scopes"] if s.get("configured") and not s.get("retired")
    ]
    assert abs(sum(float(s["target_pct"]) for s in active) - 1.0) < 1e-6
    into_spot = sum(
        float(t["amount_usdt"])
        for t in alloc["transfers"]
        if t.get("to_scope") == "spot"
    )
    spot = next(s for s in active if s["scope"] == "spot")
    spot_after = float(spot["equity_usdt"]) + into_spot
    spot_max = float(spot["band"]["max"]) * float(alloc["total_nav_usdt"])
    assert spot_after <= spot_max + 1.0
    assert not any(t.get("to_scope") is None for t in alloc["transfers"])
    sug = build_suggestions(
        allocation=alloc, composite={"label": "risk_off"}, layers={}
    )
    assert sug == []


def test_build_regime_action_plan_merges_execution_first():
    plan = build_regime_action_plan(
        allocation={
            "alert": "OK",
            "scopes": [],
        },
        composite={"label": "risk_on"},
        layers={"feature_bus": {"stale": False}},
        rolling_margin={
            "alert": "REBALANCE_SUGGEST",
            "mismatch": True,
            "block_auto_switch": True,
            "suggestions": ["建议切换 preset: usdt_r5 → usdt_btc_tuned"],
        },
        bc_coin_sleeve={
            "alert": "OK",
            "mismatch": False,
            "suggestions": ["B/C margin 与 composite 一致"],
        },
    )
    assert plan["alert"] == "REBALANCE_SUGGEST"
    assert plan["suggestions"][0].startswith("[Execution]")
    assert "先平仓" in plan["suggestions"][0]


def test_fetch_coin_sleeve_equity_sums_dapi_assets():
    raw = {
        "assets": [
            {"asset": "BTC", "marginBalance": "0.1"},
            {"asset": "BNB", "marginBalance": "2"},
            {"asset": "SOL", "marginBalance": "10"},
            {"asset": "USDT", "marginBalance": "999"},
        ]
    }
    marks = {"BTCUSDT": 100000.0, "BNBUSDT": 500.0, "SOLUSDT": 150.0}
    with patch(
        "mlbot_console.services.exchange_balances._fetch_dapi_account_raw",
        return_value=raw,
    ):
        parsed = _fetch_coin_sleeve_equity(
            api_key="k",
            api_secret="s",
            mark_prices=marks,
        )
    expected = 0.1 * 100000 + 2 * 500 + 10 * 150
    assert parsed["equity_usdt"] == pytest.approx(expected, rel=1e-4)


def test_coin_sleeve_scope_not_configured_without_keys(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BINANCE_COIN_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_COIN_API_SECRET", raising=False)
    row = fetch_scope_exchange_balance("coin_sleeve", mark_prices={"BTCUSDT": 1.0})
    assert row["scope"] == "coin_sleeve"
    assert row["configured"] is False
    assert row["ok"] is False
