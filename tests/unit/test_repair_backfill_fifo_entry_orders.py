"""Unit tests for repair_backfill_fifo_entry_orders."""

from __future__ import annotations

import sqlite3

import pytest

from scripts.repair_backfill_fifo_entry_orders import iter_fifo_pairs, repair


def _make_db(path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE positions (
            position_id TEXT PRIMARY KEY,
            symbol TEXT,
            side TEXT,
            entry_time TEXT,
            exit_time TEXT,
            entry_price REAL,
            exit_price REAL,
            realized_pnl REAL,
            status TEXT,
            strategy_id TEXT,
            stop_loss_price REAL,
            take_profit_price REAL,
            current_size REAL,
            exit_reason TEXT
        );
        CREATE TABLE orders (
            order_id TEXT PRIMARY KEY,
            symbol TEXT,
            side TEXT,
            status TEXT,
            order_type TEXT,
            quantity REAL,
            price REAL,
            stop_price REAL,
            filled_at TEXT,
            created_at TEXT,
            updated_at TEXT,
            average_price REAL,
            filled_quantity REAL,
            position_id TEXT,
            binance_order_id TEXT
        );
        """
    )
    conn.close()


def test_iter_fifo_pairs_matches_buy_then_sell() -> None:
    rows = [
        {
            "order_id": "buy1",
            "symbol": "ETHUSDT",
            "side": "BUY",
            "status": "filled",
            "average_price": 100.0,
            "filled_quantity": 1.0,
            "filled_at": "2024-01-01T10:00:00+00:00",
        },
        {
            "order_id": "sell1",
            "symbol": "ETHUSDT",
            "side": "SELL",
            "status": "filled",
            "average_price": 105.0,
            "filled_quantity": 1.0,
            "filled_at": "2024-01-01T14:00:00+00:00",
        },
    ]
    pairs = iter_fifo_pairs(rows)
    assert len(pairs) == 1
    assert pairs[0].entry["order_id"] == "buy1"
    assert pairs[0].exit["order_id"] == "sell1"
    assert pairs[0].match_qty == pytest.approx(1.0)


def test_repair_backfills_zero_qty_entry_from_fifo(tmp_path) -> None:
    db = tmp_path / "order_management.db"
    _make_db(db)
    pid = "ETHUSDT:1778904250177691000"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        INSERT INTO positions VALUES (
            ?, 'ETHUSDT', 'long',
            '2024-01-05T10:00:00+00:00', '2024-01-05T14:00:00+00:00',
            2244.51, 2227.47, -0.9713, 'closed', 'tpc', NULL, NULL, 0.057, 'backfill_fifo'
        )
        """,
        (pid,),
    )
    conn.execute(
        """
        INSERT INTO orders VALUES (
            'fifo_buy', 'ETHUSDT', 'BUY', 'filled', 'market',
            0.057, 2244.51, NULL,
            '2024-01-05T10:00:00+00:00', '2024-01-05T10:00:00+00:00',
            '2024-01-05T10:00:00+00:00',
            2244.51, 0.057, NULL, '4000001325799046'
        )
        """
    )
    conn.execute(
        """
        INSERT INTO orders VALUES (
            'ord_exit', 'ETHUSDT', 'SELL', 'filled', 'market',
            0.057, 2227.47, NULL,
            '2024-01-05T14:00:00+00:00', '2024-01-05T14:00:00+00:00',
            '2024-01-05T14:00:00+00:00',
            2227.47, 0.057, ?, '8389766180561052445'
        )
        """,
        (pid,),
    )
    conn.execute(
        """
        INSERT INTO orders VALUES (
            'ord_entry_placeholder', 'ETHUSDT', 'BUY', 'filled', 'market',
            0.0, 2244.51, NULL,
            '2024-01-05T10:00:00+00:00', '2024-01-05T10:00:00+00:00',
            '2024-01-05T10:00:00+00:00',
            2244.51, 0.0, ?, NULL
        )
        """,
        (pid,),
    )
    conn.commit()
    conn.close()

    stats = repair(db, dry_run=False)
    assert stats["repaired"] == 1

    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT filled_quantity, average_price, binance_order_id FROM orders WHERE order_id = ?",
        ("ord_entry_placeholder",),
    ).fetchone()
    conn.close()
    assert row[0] == pytest.approx(0.057)
    assert row[1] == pytest.approx(2244.51)
    assert row[2] == "4000001325799046"


def test_repair_backfills_canceled_entry_stub(tmp_path) -> None:
    db = tmp_path / "order_management.db"
    _make_db(db)
    pid = "ETHUSDT:1778904250177691000"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        INSERT INTO positions VALUES (
            ?, 'ETHUSDT', 'long',
            '2024-01-05T10:00:00+00:00', '2024-01-05T14:00:00+00:00',
            2244.51, 2227.47, -0.9713, 'closed', 'tpc', NULL, NULL, 0.057, 'backfill_fifo'
        )
        """,
        (pid,),
    )
    conn.execute(
        """
        INSERT INTO orders VALUES (
            'fifo_buy', 'ETHUSDT', 'BUY', 'filled', 'market',
            0.057, 2244.51, NULL,
            '2024-01-05T10:00:00+00:00', '2024-01-05T10:00:00+00:00',
            '2024-01-05T10:00:00+00:00',
            2244.51, 0.057, NULL, '4000001325799046'
        )
        """
    )
    conn.execute(
        """
        INSERT INTO orders VALUES (
            'ord_exit', 'ETHUSDT', 'SELL', 'filled', 'market',
            0.057, 2227.47, NULL,
            '2024-01-05T14:00:00+00:00', '2024-01-05T14:00:00+00:00',
            '2024-01-05T14:00:00+00:00',
            2227.47, 0.057, ?, '8389766180561052445'
        )
        """,
        (pid,),
    )
    conn.execute(
        """
        INSERT INTO orders VALUES (
            'ord_entry_canceled', 'ETHUSDT', 'BUY', 'canceled', 'market',
            0.057, NULL, NULL,
            '2024-01-05T10:00:00+00:00', '2024-01-05T10:00:00+00:00',
            '2024-01-05T10:00:00+00:00',
            NULL, 0.0, ?, '4000001325799046'
        )
        """,
        (pid,),
    )
    conn.commit()
    conn.close()

    stats = repair(db, dry_run=False)
    assert stats["repaired"] == 1

    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT status, filled_quantity, average_price FROM orders WHERE order_id = ?",
        ("ord_entry_canceled",),
    ).fetchone()
    conn.close()
    assert row[0] == "filled"
    assert row[1] == pytest.approx(0.057)
    assert row[2] == pytest.approx(2244.51)


def test_repair_skips_when_entry_already_filled(tmp_path) -> None:
    db = tmp_path / "order_management.db"
    _make_db(db)
    pid = "p_ok"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        INSERT INTO positions VALUES (
            ?, 'ETHUSDT', 'long',
            '2024-01-05T10:00:00+00:00', '2024-01-05T14:00:00+00:00',
            100.0, 105.0, 5.0, 'closed', 'tpc', NULL, NULL, 1.0, 'backfill_fifo'
        )
        """,
        (pid,),
    )
    conn.execute(
        """
        INSERT INTO orders VALUES (
            'buy_ok', 'ETHUSDT', 'BUY', 'filled', 'market',
            1.0, 100.0, NULL,
            '2024-01-05T10:00:00+00:00', '2024-01-05T10:00:00+00:00',
            '2024-01-05T10:00:00+00:00',
            100.0, 1.0, ?, '111'
        )
        """,
        (pid,),
    )
    conn.execute(
        """
        INSERT INTO orders VALUES (
            'sell_ok', 'ETHUSDT', 'SELL', 'filled', 'market',
            1.0, 105.0, NULL,
            '2024-01-05T14:00:00+00:00', '2024-01-05T14:00:00+00:00',
            '2024-01-05T14:00:00+00:00',
            105.0, 1.0, ?, '222'
        )
        """,
        (pid,),
    )
    conn.commit()
    conn.close()

    stats = repair(db, dry_run=False)
    assert stats["skipped_ok"] == 1
    assert stats["repaired"] == 0


def test_reconcile_b_uses_entry_after_fifo_repair(tmp_path, monkeypatch) -> None:
    from unittest.mock import MagicMock, patch

    from mlbot_console.services.reconciliation_hub import ReconciliationHub

    db = tmp_path / "order_management.db"
    _make_db(db)
    pid = "ETHUSDT:1778904250177691000"
    exit_oid = "8389766180561052445"
    entry_oid = "4000001325799046"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        INSERT INTO positions VALUES (
            ?, 'ETHUSDT', 'long',
            '2024-01-05T10:00:00+00:00', '2024-01-05T14:00:00+00:00',
            2244.51, 2227.47, -0.9713, 'closed', 'tpc', NULL, NULL, 0.057, 'backfill_fifo'
        )
        """,
        (pid,),
    )
    conn.execute(
        """
        INSERT INTO orders VALUES (
            'fifo_buy', 'ETHUSDT', 'BUY', 'filled', 'market',
            0.057, 2244.51, NULL,
            '2024-01-05T10:00:00+00:00', '2024-01-05T10:00:00+00:00',
            '2024-01-05T10:00:00+00:00',
            2244.51, 0.057, NULL, ?
        )
        """,
        (entry_oid,),
    )
    conn.execute(
        """
        INSERT INTO orders VALUES (
            'ord_exit', 'ETHUSDT', 'SELL', 'filled', 'market',
            0.057, 2227.47, NULL,
            '2024-01-05T14:00:00+00:00', '2024-01-05T14:00:00+00:00',
            '2024-01-05T14:00:00+00:00',
            2227.47, 0.057, ?, ?
        )
        """,
        (pid, exit_oid),
    )
    conn.execute(
        """
        INSERT INTO orders VALUES (
            'ord_entry_placeholder', 'ETHUSDT', 'BUY', 'filled', 'market',
            0.0, 2244.51, NULL,
            '2024-01-05T10:00:00+00:00', '2024-01-05T10:00:00+00:00',
            '2024-01-05T10:00:00+00:00',
            2244.51, 0.0, ?, NULL
        )
        """,
        (pid,),
    )
    conn.commit()
    conn.close()

    repair(db, dry_run=False)

    entry_trade = {
        "id": "e1",
        "order": entry_oid,
        "price": 2244.51,
        "amount": 0.057,
        "side": "buy",
        "fee": {"cost": 0.1},
        "timestamp": 1_700_000_000_000,
    }
    exit_trade = {
        "id": "x1",
        "order": exit_oid,
        "price": 2227.47,
        "amount": 0.057,
        "side": "sell",
        "fee": {"cost": 0.2},
        "timestamp": 1_700_010_000_000,
        "info": {"realizedPnl": "-0.97"},
    }

    def fetch_my_trades(_sym, since=0, params=None):
        oid = str((params or {}).get("orderId") or "")
        if oid == entry_oid:
            return [entry_trade]
        if oid == exit_oid:
            return [exit_trade]
        return []

    monkeypatch.setenv("BINANCE_FUTURES_API_KEY", "test-key")
    monkeypatch.setenv("BINANCE_FUTURES_API_SECRET", "test-secret")

    rec_db = tmp_path / "reconciliation_after_repair.db"
    hub = ReconciliationHub(rec_db)
    mock_exchange = MagicMock()
    mock_exchange.fetch_my_trades.side_effect = fetch_my_trades
    mock_exchange.fetch_funding_history.return_value = []

    with patch("ccxt.binance", return_value=mock_exchange):
        report = hub.reconcile_B(
            db,
            strategy="tpc",
            symbols=["ETHUSDT"],
            since=__import__("datetime").datetime(
                2024, 1, 1, tzinfo=__import__("datetime").timezone.utc
            ),
        )

    pair_item = next(
        i for i in report.items if i.pair_id == "ord_entry_placeholder→ord_exit"
    )
    assert pair_item.local_qty == pytest.approx(0.057)
    assert pair_item.entry_order_id == entry_oid
    assert pair_item.status == "OK"
    assert pair_item.exchange_pnl == pytest.approx(-0.9713)
    assert report.n_mismatch == 0
