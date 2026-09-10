"""B equity anchors must survive restart so daily_loss cannot re-arm after a loss day."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from order_management.trend_account_safety import (
    BLayerEquityTracker,
    account_update_from_balance,
    account_update_has_equity_authority,
    default_b_equity_state_path,
    g0_equity_from_account_update,
)


def test_default_b_equity_state_path() -> None:
    p = default_b_equity_state_path("/data/position_tracker")
    assert p == Path("/data/position_tracker/_b_equity_anchors.json")


def test_account_update_from_balance_fapi_and_coin_m_shapes() -> None:
    fapi = account_update_from_balance(
        {"USDT": {"total": 1000.0, "free": 900.0}, "info": {}}
    )
    assert fapi["wallet_balance"] == 1000.0
    assert fapi["available_balance"] == 900.0
    coin = account_update_from_balance(
        {
            "info": {"totalMarginBalance": 1739.5, "availableBalance": 1700.0},
            "totalWalletBalance": 1739.5,
        }
    )
    assert coin["wallet_balance"] == 1739.5
    assert coin["available_balance"] == 1700.0


def test_account_update_from_balance_float_zero_wallet_is_wipe() -> None:
    """Numeric 0.0 must not fall through ``or`` to margin residue."""
    wipe = account_update_from_balance(
        {
            "info": {
                "totalWalletBalance": 0.0,
                "totalMarginBalance": 500.0,
                "availableBalance": 0.0,
            }
        }
    )
    assert wipe["wallet_balance"] == 0.0
    assert wipe["margin_balance"] == 500.0
    assert wipe["available_balance"] == 0.0


def test_g0_numeraire_is_wallet_balance_on_both_rest_and_ws() -> None:
    """同一口径 = 钱包余额: REST totalWalletBalance == WS wb (排除部分 upnl)."""
    rest = account_update_from_balance(
        {
            "info": {
                "totalMarginBalance": 9_000.0,
                "totalWalletBalance": 10_000.0,
                "availableBalance": 8_500.0,
            }
        }
    )
    # Wallet path for REST↔WS; margin is a separate REST overlay field.
    assert rest["wallet_balance"] == 10_000.0
    assert rest["margin_balance"] == 9_000.0
    assert g0_equity_from_account_update(rest) == 10_000.0

    # WS event with only one symbol's (partial) upnl must NOT be added.
    ws = {
        "wallet_balance": 10_000.0,
        "available_balance": 8_500.0,
        "unrealized_pnl_total": -1_000.0,
        "balances": [
            {
                "asset": "USDT",
                "wallet_balance": 10_000.0,
                "cross_wallet_balance": 8_500.0,
            }
        ],
    }
    assert account_update_has_equity_authority(ws) is True
    assert g0_equity_from_account_update(ws) == 10_000.0


def test_period_roll_clears_stale_margin_observation(tmp_path: Path) -> None:
    """Day roll clears mark observation; G0 never used margin for daily_loss."""
    t = BLayerEquityTracker(state_path=tmp_path / "e.json")
    day0 = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
    day1 = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
    t.update(10_000.0, now=day0)
    t.update_margin_from_rest(9_000.0)
    assert t.metrics_snapshot()["daily_loss"] == 0.0
    assert t.metrics_snapshot()["equity"] == 10_000.0
    m = t.update(10_000.0, now=day1)
    assert t.last_margin_equity is None
    assert m["daily_loss"] == 0.0
    assert m["equity"] == 10_000.0


def test_rest_margin_drop_does_not_halt_g0_while_wallet_unchanged(
    tmp_path: Path,
) -> None:
    """G0 wallet/realized-only: margin MTM drop must not trip daily_loss/max_dd."""
    path = tmp_path / "_b_equity_anchors.json"
    t = BLayerEquityTracker(state_path=path)
    now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
    t.update(10_000.0, now=now)
    m = t.update_margin_from_rest(7_000.0)  # deep mark loss, wallet flat
    assert m["equity"] == 10_000.0
    assert m["equity_wallet"] == 10_000.0
    assert m["equity_margin"] == 7_000.0
    assert m["daily_loss"] == 0.0
    assert m["drawdown"] == 0.0

    # WS wallet tick unchanged — still no G0 loss.
    t.update(10_000.0, now=now)
    snap = t.metrics_snapshot()
    assert snap["equity"] == 10_000.0
    assert snap["daily_loss"] == 0.0
    assert snap["equity_margin"] == 7_000.0

    # Realized wallet drop still trips G0.
    m2 = t.update(9_000.0, now=now)
    assert m2["equity"] == 9_000.0
    assert m2["daily_loss"] == pytest.approx(0.1)


def test_non_settlement_balances_have_no_equity_authority() -> None:
    upd = {
        "wallet_balance": 0.0,
        "available_balance": 0.0,
        "balances": [
            {"asset": "BNB", "wallet_balance": 1.0, "cross_wallet_balance": 1.0}
        ],
    }
    assert account_update_has_equity_authority(upd) is False
    assert g0_equity_from_account_update(upd) is None


def test_anchors_persist_across_reload(tmp_path: Path) -> None:
    path = tmp_path / "_b_equity_anchors.json"
    t0 = BLayerEquityTracker(state_path=path)
    now = datetime(2026, 7, 16, 10, 0, tzinfo=timezone.utc)
    m1 = t0.update(10_000.0, now=now)
    assert m1["daily_loss"] == 0.0
    m2 = t0.update(9_000.0, now=now)
    assert m2["daily_loss"] == 0.1
    assert path.is_file()

    # Simulate process restart: fresh tracker loads anchors, same day.
    t1 = BLayerEquityTracker(state_path=path)
    assert t1.load()
    assert t1.day_start_equity == 10_000.0
    assert t1.peak_equity == 10_000.0
    snap = t1.metrics_snapshot()
    assert snap["daily_loss"] == 0.1
    assert snap["drawdown"] == 0.1

    # Next wallet tick at same equity must keep the loss (not reset day_start).
    m3 = t1.update(9_000.0, now=now)
    assert m3["daily_loss"] == 0.1


def test_fresh_tracker_without_file_still_works(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"
    t = BLayerEquityTracker(state_path=path)
    assert t.load() is False
    m = t.update(5_000.0, now=datetime(2026, 7, 16, tzinfo=timezone.utc))
    assert m["equity"] == 5_000.0
    assert path.is_file()


def test_um_wipe_to_zero_is_full_loss(tmp_path: Path) -> None:
    """current<=0 must be 100% loss (not fail-open 0)."""
    path = tmp_path / "_b_equity_anchors.json"
    t = BLayerEquityTracker(state_path=path)
    now = datetime(2026, 7, 16, 10, 0, tzinfo=timezone.utc)
    t.update(10_000.0, now=now)
    m = t.update(0.0, now=now)
    assert m["daily_loss"] == 1.0
    assert m["drawdown"] == 1.0
    assert m["equity"] == 0.0


def test_rebind_after_constitution_persists_halt(tmp_path: Path) -> None:
    """Executor wired after __init__ must still rewrite safety_state from anchors."""
    from unittest.mock import MagicMock

    from live_data_stream.order_flow_listener import OrderFlowListener
    from time_series_model.core.constitution.safety_runtime import load_safety_state

    path = tmp_path / "_b_equity_anchors.json"
    safety_db = tmp_path / "orders.db"
    t = BLayerEquityTracker(state_path=path)
    now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
    t.update(10_000.0, now=now)
    t.update(9_000.0, now=now)  # 10% daily loss

    listener = OrderFlowListener.__new__(OrderFlowListener)
    listener.symbol = "BTCUSDT"
    listener._equity_tracker = BLayerEquityTracker(state_path=path)
    assert listener._equity_tracker.load()
    listener.constitution_executor = None
    listener._safety_halt_notified = False
    listener._coin_asset_book = None  # U-m path; coin_m uses bind_coin_m_asset_book

    listener._maybe_notify_safety_halt = MagicMock()  # type: ignore[method-assign]

    # No executor yet → no-op
    listener.rebind_b_safety_after_constitution()
    assert not safety_db.exists()

    listener.constitution_executor = SimpleNamespace(
        cfg=SimpleNamespace(
            cooldown_minutes=720,
            daily_reset_timezone="UTC",
            max_dd=0.2,
            daily_loss_limit=0.06,
            weekly_loss_limit=1.0,
            monthly_loss_limit=1.0,
            max_turnover_mean=1.0,
            max_cost_mean=1.0,
            kill_on_any_hard_violation=True,
        ),
        resolve_safety_db_path=lambda: safety_db,
    )
    listener.rebind_b_safety_after_constitution()
    state = load_safety_state(db_path=str(safety_db))
    assert state.halted is True
    assert state.halt_reason == ["daily_loss_limit"]


def test_u_m_rest_stale_rise_does_not_overwrite_ws_loss(tmp_path: Path) -> None:
    """U-m: concurrent WS loss must win over a stale higher REST snapshot."""
    path = tmp_path / "_b_equity_anchors.json"
    t = BLayerEquityTracker(state_path=path)
    now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
    t.update(10_000.0, now=now)
    seq = t.mutation_seq
    t.update(9_000.0, now=now)  # WS loss while REST in flight
    # Stale REST still says 10k — must keep 9k.
    metrics = t.update(10_000.0, now=now, stale_rise_seq=seq)
    assert metrics["equity"] == 9_000.0
    assert t.last_equity == 9_000.0
    # Same-seq REST rise (no concurrent mutation) still applies (deposit).
    t2 = BLayerEquityTracker(state_path=tmp_path / "t2.json")
    t2.update(10_000.0, now=now)
    seq2 = t2.mutation_seq
    metrics2 = t2.update(11_000.0, now=now, stale_rise_seq=seq2)
    assert metrics2["equity"] == 11_000.0


def test_listeners_share_manager_equity_tracker(tmp_path, monkeypatch) -> None:
    """MultiSymbolManager must bind one U-m tracker to every listener."""
    monkeypatch.setenv("MLBOT_MARGIN_KIND", "usd_m")
    monkeypatch.setenv("MLBOT_COIN_M_ASSET_G0", "0")
    monkeypatch.setenv("MLBOT_POSITION_TRACKER_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("MLBOT_B_EQUITY_STATE_PATH", raising=False)

    from unittest.mock import MagicMock

    from src.live_data_stream.multi_symbol_manager import MultiSymbolManager

    storage = MagicMock()
    mgr = MultiSymbolManager(
        symbols=["BTCUSDT", "ETHUSDT"],
        storage_manager=storage,
        order_manager=MagicMock(binance_api=None),
    )
    assert mgr._b_equity_tracker is not None
    assert mgr._coin_asset_book is None
    t_btc = mgr.listeners["BTCUSDT"]._equity_tracker
    t_eth = mgr.listeners["ETHUSDT"]._equity_tracker
    assert t_btc is mgr._b_equity_tracker
    assert t_eth is mgr._b_equity_tracker
    assert t_btc is t_eth


def test_u_m_ignores_coin_m_asset_state_dir(tmp_path, monkeypatch) -> None:
    """U-m must not load/write anchors under MLBOT_COIN_M_ASSET_STATE_DIR."""
    monkeypatch.setenv("MLBOT_MARGIN_KIND", "usd_m")
    monkeypatch.setenv("MLBOT_COIN_M_ASSET_G0", "0")
    coin_dir = tmp_path / "coin_lane"
    um_dir = tmp_path / "um_lane"
    coin_dir.mkdir()
    um_dir.mkdir()
    monkeypatch.setenv("MLBOT_COIN_M_ASSET_STATE_DIR", str(coin_dir))
    monkeypatch.setenv("MLBOT_POSITION_TRACKER_STATE_DIR", str(um_dir))
    monkeypatch.delenv("MLBOT_B_EQUITY_STATE_PATH", raising=False)

    from unittest.mock import MagicMock

    from src.live_data_stream.multi_symbol_manager import MultiSymbolManager

    mgr = MultiSymbolManager(
        symbols=["BTCUSDT"],
        storage_manager=MagicMock(),
        order_manager=MagicMock(binance_api=None),
    )
    path = mgr._b_equity_tracker.state_path
    assert path is not None
    assert path.parent == um_dir
    assert "coin_lane" not in str(path)


def test_inject_features_uses_tracker_equity_not_stale_wallet(tmp_path: Path) -> None:
    """Sizing equity must follow tracker after stale REST clamp."""
    from live_data_stream.order_flow_listener import OrderFlowListener

    path = tmp_path / "_b_equity_anchors.json"
    t = BLayerEquityTracker(state_path=path)
    now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
    t.update(10_000.0, now=now)
    seq = t.mutation_seq
    t.update(9_000.0, now=now)
    t.update(10_000.0, now=now, stale_rise_seq=seq)
    assert t.last_equity == 9_000.0

    listener = OrderFlowListener.__new__(OrderFlowListener)
    listener._coin_asset_book = None
    listener._equity_tracker = t
    listener._latest_account_update = {
        "wallet_balance": 10_000.0,  # stale REST still in snapshot
        "available_balance": 9_500.0,
        "unrealized_pnl_total": 0.0,
    }
    listener.constitution_executor = None
    features: dict = {}
    listener._inject_account_features(features)
    assert features["equity"] == 9_000.0
    assert features["daily_loss"] == pytest.approx(0.1)


def test_inject_features_respects_tracker_wipe(tmp_path: Path) -> None:
    from live_data_stream.order_flow_listener import OrderFlowListener

    t = BLayerEquityTracker(state_path=tmp_path / "e.json")
    now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
    t.update(10_000.0, now=now)
    t.update(0.0, now=now)
    listener = OrderFlowListener.__new__(OrderFlowListener)
    listener._coin_asset_book = None
    listener._equity_tracker = t
    listener._latest_account_update = {
        "wallet_balance": 10_000.0,  # stale positive must not re-arm
        "available_balance": 0.0,
        "unrealized_pnl_total": 0.0,
    }
    listener.constitution_executor = None
    features: dict = {}
    listener._inject_account_features(features)
    assert features["equity"] == 0.0
    assert features["daily_loss"] == 1.0


def test_listener_keeps_ws_available_when_rest_omits_it(tmp_path: Path) -> None:
    from live_data_stream.order_flow_listener import OrderFlowListener

    listener = OrderFlowListener.__new__(OrderFlowListener)
    listener._coin_asset_book = None
    listener._equity_tracker = BLayerEquityTracker(state_path=tmp_path / "e.json")
    listener.constitution_executor = None
    listener._latest_account_update = {
        "wallet_balance": 9_000.0,
        "available_balance": 8_000.0,
    }
    listener.on_account_update(
        {
            "event_type": "REST_BALANCE",
            "wallet_balance": 9_000.0,
            # available omitted on purpose (stale REST clamp)
        },
        skip_equity_update=True,
    )
    assert listener._latest_account_update["available_balance"] == 8_000.0
