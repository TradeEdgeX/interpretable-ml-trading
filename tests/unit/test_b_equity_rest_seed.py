"""B equity REST seed fan-out to listeners."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.live_data_stream.multi_symbol_manager import MultiSymbolManager
from src.order_management.coin_m_asset_equity import (
    CoinMAssetEquityBook,
    live_asset_g0_place_block_reason,
    set_live_asset_g0_place_block,
)


def test_seed_account_equity_from_rest_fans_out() -> None:
    mgr = MultiSymbolManager.__new__(MultiSymbolManager)
    mgr._coin_asset_book = None
    mgr._b_equity_tracker = None
    mgr.order_manager = MagicMock()
    mgr.order_manager.binance_api.get_account_balance.return_value = {
        "info": {"totalMarginBalance": 2000.0, "availableBalance": 1900.0},
        "USDT": {"total": 2000.0, "free": 1900.0},
    }
    l1 = MagicMock()
    l2 = MagicMock()
    mgr.listeners = {"BTCUSDT": l1, "ETHUSDT": l2}

    assert mgr.seed_account_equity_from_rest() is True
    assert l1.on_account_update.call_count == 1
    assert l2.on_account_update.call_count == 1
    assert l1.on_account_update.call_args.kwargs.get("skip_equity_update") is True
    upd = l1.on_account_update.call_args[0][0]
    assert upd["wallet_balance"] == 2000.0
    assert upd["available_balance"] == 1900.0


def test_seed_skips_when_equity_zero() -> None:
    mgr = MultiSymbolManager.__new__(MultiSymbolManager)
    mgr._coin_asset_book = None
    mgr._b_equity_tracker = None
    mgr.order_manager = MagicMock()
    mgr.order_manager.binance_api.get_account_balance.return_value = {"info": {}}
    mgr.listeners = {"BTCUSDT": MagicMock()}
    assert mgr.seed_account_equity_from_rest() is False
    mgr.listeners["BTCUSDT"].on_account_update.assert_not_called()


def test_u_m_shared_tracker_updated_once_on_rest(tmp_path: Path) -> None:
    """One account-level tracker; N listeners only merge snapshots."""
    from src.order_management.trend_account_safety import BLayerEquityTracker

    tracker = BLayerEquityTracker(state_path=tmp_path / "_b_equity_anchors.json")
    mgr = MultiSymbolManager.__new__(MultiSymbolManager)
    mgr._coin_asset_book = None
    mgr._b_equity_tracker = tracker
    mgr.order_manager = MagicMock()
    mgr.order_manager.binance_api.get_account_balance.return_value = {
        "info": {"totalMarginBalance": 5000.0, "availableBalance": 4800.0},
    }
    l1 = MagicMock()
    l2 = MagicMock()
    l1.constitution_executor = None
    l2.constitution_executor = None
    mgr.listeners = {"BTCUSDT": l1, "ETHUSDT": l2}

    assert mgr.seed_account_equity_from_rest() is True
    assert tracker.last_equity == 5000.0
    assert tracker.mutation_seq == 1
    assert l1.on_account_update.call_args.kwargs.get("skip_equity_update") is True
    assert l2.on_account_update.call_args.kwargs.get("skip_equity_update") is True


def test_u_m_ws_account_update_applies_shared_tracker_once(tmp_path: Path) -> None:
    from src.order_management.trend_account_safety import BLayerEquityTracker

    tracker = BLayerEquityTracker(state_path=tmp_path / "_b_equity_anchors.json")
    mgr = MultiSymbolManager.__new__(MultiSymbolManager)
    mgr._coin_asset_book = None
    mgr._b_equity_tracker = tracker
    l1 = MagicMock()
    l2 = MagicMock()
    l1.constitution_executor = None
    l2.constitution_executor = None
    mgr.listeners = {"BTCUSDT": l1, "ETHUSDT": l2}

    mgr._on_account_update(
        {
            "wallet_balance": 10_000.0,
            "available_balance": 9_000.0,
            "balances": [
                {
                    "asset": "USDT",
                    "wallet_balance": 10_000.0,
                    "cross_wallet_balance": 10_000.0,
                }
            ],
        }
    )
    assert tracker.last_equity == 10_000.0
    assert tracker.mutation_seq == 1
    assert l1.on_account_update.call_args.kwargs.get("skip_equity_update") is True
    assert l2.on_account_update.call_args.kwargs.get("skip_equity_update") is True


def test_u_m_rest_manager_clamps_stale_equity_fanout(tmp_path: Path) -> None:
    """REST seed must fan-out tracker-clamped equity (not stale higher REST)."""
    from datetime import datetime, timezone

    from src.order_management.trend_account_safety import BLayerEquityTracker

    tracker = BLayerEquityTracker(state_path=tmp_path / "_b_equity_anchors.json")
    now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
    tracker.update(10_000.0, now=now)

    class _RaceBal:
        def get_account_balance(self):
            # Concurrent WS loss after manager snapped mutation_seq.
            tracker.update(9_000.0, now=now)
            return {
                "info": {
                    "totalMarginBalance": 10_000.0,
                    "availableBalance": 9_500.0,
                }
            }

    mgr = MultiSymbolManager.__new__(MultiSymbolManager)
    mgr._coin_asset_book = None
    mgr._b_equity_tracker = tracker
    mgr.order_manager = MagicMock()
    mgr.order_manager.binance_api = _RaceBal()
    l1 = MagicMock()
    l1.constitution_executor = None
    mgr.listeners = {"BTCUSDT": l1}

    assert mgr.seed_account_equity_from_rest() is True
    assert tracker.last_equity == 9_000.0
    fan = l1.on_account_update.call_args[0][0]
    assert fan["wallet_balance"] == 9_000.0
    # Stale REST available must not be forced over fresher WS available.
    assert "available_balance" not in fan
    assert l1.on_account_update.call_args.kwargs.get("skip_equity_update") is True


def test_u_m_ws_explicit_wipe_updates_tracker(tmp_path: Path) -> None:
    """USDT/USDC balances[] with wallet=0 must register 100% loss (not ignored)."""
    from datetime import datetime, timezone

    from src.order_management.trend_account_safety import BLayerEquityTracker

    tracker = BLayerEquityTracker(state_path=tmp_path / "_b_equity_anchors.json")
    now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
    tracker.update(10_000.0, now=now)

    mgr = MultiSymbolManager.__new__(MultiSymbolManager)
    mgr._coin_asset_book = None
    mgr._b_equity_tracker = tracker
    l1 = MagicMock()
    l1.constitution_executor = None
    mgr.listeners = {"BTCUSDT": l1}

    mgr._on_account_update(
        {
            "wallet_balance": 0.0,
            "available_balance": 0.0,
            "unrealized_pnl_total": 0.0,
            "balances": [
                {"asset": "USDT", "wallet_balance": 0.0, "cross_wallet_balance": 0.0}
            ],
        }
    )
    assert tracker.last_equity == 0.0
    assert tracker.metrics_snapshot()["daily_loss"] == 1.0


def test_u_m_rest_margin_overlay_via_manager(tmp_path: Path) -> None:
    """Manager REST seed: G0 stays on wallet; margin is observation only."""
    from datetime import datetime, timezone

    from src.order_management.trend_account_safety import BLayerEquityTracker

    tracker = BLayerEquityTracker(state_path=tmp_path / "_b_equity_anchors.json")
    now = datetime.now(timezone.utc)
    tracker.update(10_000.0, now=now)

    mgr = MultiSymbolManager.__new__(MultiSymbolManager)
    mgr._coin_asset_book = None
    mgr._b_equity_tracker = tracker
    mgr.order_manager = MagicMock()
    mgr.order_manager.binance_api.get_account_balance.return_value = {
        "info": {
            "totalWalletBalance": 10_000.0,
            "totalMarginBalance": 9_000.0,
            "availableBalance": 8_500.0,
        }
    }
    mgr.listeners = {"BTCUSDT": MagicMock(constitution_executor=None)}

    assert mgr.seed_account_equity_from_rest() is True
    snap = tracker.metrics_snapshot()
    assert snap["equity_wallet"] == 10_000.0
    assert snap["equity_margin"] == 9_000.0
    assert snap["equity"] == 10_000.0
    assert snap["daily_loss"] == 0.0


def test_u_m_ws_uses_wallet_not_partial_upnl(tmp_path: Path) -> None:
    """WS G0 = wallet only. Partial per-event upnl must not move equity."""
    from datetime import datetime, timezone

    from src.order_management.trend_account_safety import BLayerEquityTracker

    tracker = BLayerEquityTracker(state_path=tmp_path / "_b_equity_anchors.json")
    # Manager update() uses wall-clock now — seed on the same UTC day.
    now = datetime.now(timezone.utc)
    tracker.update(10_000.0, now=now)

    mgr = MultiSymbolManager.__new__(MultiSymbolManager)
    mgr._coin_asset_book = None
    mgr._b_equity_tracker = tracker
    mgr.listeners = {"BTCUSDT": MagicMock(constitution_executor=None)}

    mgr._on_account_update(
        {
            "wallet_balance": 10_000.0,
            "available_balance": 9_000.0,
            # Only one changed symbol's upnl — must be ignored for G0 equity.
            "unrealized_pnl_total": -1_000.0,
            "balances": [
                {
                    "asset": "USDT",
                    "wallet_balance": 10_000.0,
                    "cross_wallet_balance": 9_000.0,
                }
            ],
        }
    )
    assert tracker.last_equity == 10_000.0
    assert tracker.metrics_snapshot()["daily_loss"] == 0.0

    # A realized wallet drop still trips daily_loss.
    mgr._on_account_update(
        {
            "wallet_balance": 9_000.0,
            "available_balance": 9_000.0,
            "unrealized_pnl_total": 0.0,
            "balances": [
                {
                    "asset": "USDT",
                    "wallet_balance": 9_000.0,
                    "cross_wallet_balance": 9_000.0,
                }
            ],
        }
    )
    assert tracker.last_equity == 9_000.0
    assert tracker.metrics_snapshot()["daily_loss"] == pytest.approx(0.1)


def test_u_m_non_settlement_b_array_does_not_wipe(tmp_path: Path) -> None:
    """Partial B[] with only BNB must not treat missing USDT as equity wipe."""
    from datetime import datetime, timezone

    from src.order_management.trend_account_safety import BLayerEquityTracker

    tracker = BLayerEquityTracker(state_path=tmp_path / "_b_equity_anchors.json")
    now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
    tracker.update(10_000.0, now=now)

    mgr = MultiSymbolManager.__new__(MultiSymbolManager)
    mgr._coin_asset_book = None
    mgr._b_equity_tracker = tracker
    mgr.listeners = {"BTCUSDT": MagicMock(constitution_executor=None)}

    mgr._on_account_update(
        {
            "wallet_balance": 0.0,
            "available_balance": 0.0,
            "balances": [
                {"asset": "BNB", "wallet_balance": 2.0, "cross_wallet_balance": 2.0}
            ],
        }
    )
    assert tracker.last_equity == 10_000.0


def test_u_m_position_only_update_does_not_wipe(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    from src.order_management.trend_account_safety import BLayerEquityTracker

    tracker = BLayerEquityTracker(state_path=tmp_path / "_b_equity_anchors.json")
    now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
    tracker.update(10_000.0, now=now)

    mgr = MultiSymbolManager.__new__(MultiSymbolManager)
    mgr._coin_asset_book = None
    mgr._b_equity_tracker = tracker
    mgr.listeners = {"BTCUSDT": MagicMock(constitution_executor=None)}

    mgr._on_account_update(
        {
            "wallet_balance": 0.0,
            "available_balance": 0.0,
            "balances": [],
            "positions": [{"symbol": "BTCUSDT", "position_amt": 1.0}],
        }
    )
    assert tracker.last_equity == 10_000.0


def test_coin_m_rest_transfer_fail_sets_place_block(tmp_path: Path) -> None:
    set_live_asset_g0_place_block(None)
    book = CoinMAssetEquityBook(state_path=tmp_path / "book.json")
    now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
    book.update_from_quantities({"BTC": 1.0}, now=now)

    mgr = MultiSymbolManager.__new__(MultiSymbolManager)
    mgr._coin_asset_book = book
    api = MagicMock()
    api.get_account_balance.return_value = {
        "assets": [{"asset": "BTC", "marginBalance": "1.0"}]
    }
    api.get_income.side_effect = RuntimeError("income down")
    mgr.order_manager = MagicMock()
    mgr.order_manager.binance_api = api
    listener = MagicMock()
    mgr.listeners = {"BTCUSDT": listener}

    assert mgr.refresh_account_equity_from_rest(context="periodic") is False
    assert live_asset_g0_place_block_reason()
    assert book.by_asset["BTC"].last_equity == 1.0
    listener.apply_coin_m_asset_book.assert_not_called()
    set_live_asset_g0_place_block(None)


def test_coin_m_rest_ok_clears_place_block(tmp_path: Path) -> None:
    set_live_asset_g0_place_block("asset_g0_ws_qty_increase_pending_rest")
    book = CoinMAssetEquityBook(state_path=tmp_path / "book.json")
    now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
    book.update_from_quantities({"BTC": 1.0}, now=now)

    mgr = MultiSymbolManager.__new__(MultiSymbolManager)
    mgr._coin_asset_book = book
    api = MagicMock()
    api.get_account_balance.return_value = {
        "assets": [{"asset": "BTC", "marginBalance": "1.05"}]
    }
    api.get_income.return_value = []
    mgr.order_manager = MagicMock()
    mgr.order_manager.binance_api = api
    listener = MagicMock()
    mgr.listeners = {"BTCUSDT": listener}

    assert mgr.refresh_account_equity_from_rest(context="periodic") is True
    assert live_asset_g0_place_block_reason() is None
    assert book.by_asset["BTC"].last_equity == 1.05
    listener.apply_coin_m_asset_book.assert_called()
    set_live_asset_g0_place_block(None)


def test_coin_m_rest_clears_sticky_stale_vs_ws_on_clean_refresh(
    tmp_path: Path,
) -> None:
    """Prior TOCTOU stale block must clear when a later REST has no race."""
    set_live_asset_g0_place_block(None)
    book = CoinMAssetEquityBook(state_path=tmp_path / "book.json")
    now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
    book.update_from_quantities({"BTC": 1.0}, now=now)
    set_live_asset_g0_place_block("asset_g0_rest_stale_vs_ws", asset="BTC")

    mgr = MultiSymbolManager.__new__(MultiSymbolManager)
    mgr._coin_asset_book = book
    api = MagicMock()
    api.get_account_balance.return_value = {
        "assets": [{"asset": "BTC", "marginBalance": "0.95"}]
    }
    api.get_income.return_value = []
    mgr.order_manager = MagicMock()
    mgr.order_manager.binance_api = api
    mgr.listeners = {"BTCUSDT": MagicMock()}

    assert mgr.refresh_account_equity_from_rest(context="periodic") is True
    assert live_asset_g0_place_block_reason(asset="BTC") is None
    assert abs(book.by_asset["BTC"].last_equity - 0.95) < 1e-9
    set_live_asset_g0_place_block(None)


def test_coin_m_rest_keeps_pending_rest_when_qty_unchanged(tmp_path: Path) -> None:
    """WS rise during REST must not be wiped by a successful same-qty refresh."""
    from src.order_management.coin_m_asset_equity import (
        apply_ws_asset_quantities_fail_closed,
    )

    set_live_asset_g0_place_block(None)
    book = CoinMAssetEquityBook(state_path=tmp_path / "book.json")
    now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
    book.update_from_quantities({"BTC": 1.0}, now=now)

    mgr = MultiSymbolManager.__new__(MultiSymbolManager)
    mgr._coin_asset_book = book

    class _Api:
        def get_income(self, **kwargs):
            # Concurrent WS deposit mid-refresh: hold + place-block, no seq bump.
            apply_ws_asset_quantities_fail_closed(book, {"BTC": 1.6}, now=now)
            return []

        def get_account_balance(self):
            return {"assets": [{"asset": "BTC", "marginBalance": "1.0"}]}

    mgr.order_manager = MagicMock()
    mgr.order_manager.binance_api = _Api()
    mgr.listeners = {"BTCUSDT": MagicMock()}

    assert mgr.refresh_account_equity_from_rest(context="periodic") is True
    assert book.by_asset["BTC"].last_equity == 1.0
    assert live_asset_g0_place_block_reason(asset="BTC") == (
        "asset_g0_ws_qty_increase_pending_rest"
    )
    set_live_asset_g0_place_block(None)


def test_coin_m_rest_clears_pending_rest_when_qty_rises(tmp_path: Path) -> None:
    set_live_asset_g0_place_block("asset_g0_ws_qty_increase_pending_rest", asset="BTC")
    book = CoinMAssetEquityBook(state_path=tmp_path / "book.json")
    now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
    book.update_from_quantities({"BTC": 1.0}, now=now)

    mgr = MultiSymbolManager.__new__(MultiSymbolManager)
    mgr._coin_asset_book = book
    api = MagicMock()
    api.get_account_balance.return_value = {
        "assets": [{"asset": "BTC", "marginBalance": "1.6"}]
    }
    api.get_income.return_value = [
        {
            "incomeType": "TRANSFER",
            "asset": "BTC",
            "income": "0.6",
            "time": 1,
            "tranId": "d1",
        }
    ]
    mgr.order_manager = MagicMock()
    mgr.order_manager.binance_api = api
    mgr.listeners = {"BTCUSDT": MagicMock()}

    assert mgr.refresh_account_equity_from_rest(context="periodic") is True
    assert abs(book.by_asset["BTC"].last_equity - 1.6) < 1e-9
    assert live_asset_g0_place_block_reason(asset="BTC") is None
    set_live_asset_g0_place_block(None)


def test_u_m_rest_wipe_updates_tracker(tmp_path: Path) -> None:
    from src.order_management.trend_account_safety import BLayerEquityTracker

    tracker = BLayerEquityTracker(state_path=tmp_path / "_b_equity_anchors.json")
    now = datetime.now(timezone.utc)
    tracker.update(10_000.0, now=now)

    mgr = MultiSymbolManager.__new__(MultiSymbolManager)
    mgr._coin_asset_book = None
    mgr._b_equity_tracker = tracker
    mgr.order_manager = MagicMock()
    mgr.order_manager.binance_api.get_account_balance.return_value = {
        "info": {
            "totalWalletBalance": 0.0,
            "totalMarginBalance": 0.0,
            "availableBalance": 0.0,
        }
    }
    mgr.listeners = {"BTCUSDT": MagicMock(constitution_executor=None)}

    assert mgr.seed_account_equity_from_rest() is True
    assert tracker.last_equity == 0.0
    assert tracker.metrics_snapshot()["daily_loss"] == 1.0
