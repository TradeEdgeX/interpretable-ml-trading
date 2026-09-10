import json
import sys

import pandas as pd

from scripts import build_backtest_time_windows as btw


def test_build_backtest_time_windows_from_trades_json(tmp_path, monkeypatch):
    trades = [
        {
            "Entry Timestamp": "2025-01-01T00:00:00Z",
            "Exit Timestamp": "2025-01-01T04:00:00Z",
            "Symbol": "BTCUSDT",
        },
        {
            "Entry Timestamp": "2025-01-02T00:00:00Z",
            "Exit Timestamp": "2025-01-02T02:00:00Z",
            "Symbol": "ETHUSDT",
        },
    ]
    trades_path = tmp_path / "trades.json"
    trades_path.write_text(json.dumps(trades), encoding="utf-8")
    out_path = tmp_path / "windows.json"

    argv = [
        "build_backtest_time_windows.py",
        "--trades",
        str(trades_path),
        "--out",
        str(out_path),
        "--pre-minutes",
        "60",
        "--post-minutes",
        "60",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    btw.main()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "windows" in payload
    assert len(payload["windows"]) == 2
    assert payload["windows"][0]["symbol"] in {"BTCUSDT", "ETHUSDT"}


def test_build_backtest_time_windows_with_negative_sampling(tmp_path, monkeypatch):
    trades = [
        {
            "Entry Timestamp": "2025-01-01T00:00:00Z",
            "Exit Timestamp": "2025-01-01T01:00:00Z",
            "Symbol": "BTCUSDT",
        }
    ]
    trades_path = tmp_path / "trades.json"
    trades_path.write_text(json.dumps(trades), encoding="utf-8")
    out_path = tmp_path / "windows.json"

    timeline = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="H", tz="UTC"),
            "symbol": ["BTCUSDT"] * 10,
        }
    )
    timeline_path = tmp_path / "timeline.parquet"
    timeline.to_parquet(timeline_path)

    argv = [
        "build_backtest_time_windows.py",
        "--trades",
        str(trades_path),
        "--out",
        str(out_path),
        "--pre-minutes",
        "30",
        "--post-minutes",
        "30",
        "--negative-ratio",
        "1.0",
        "--timeline-parquet",
        str(timeline_path),
        "--timeline-ts-col",
        "timestamp",
        "--timeline-symbol-col",
        "symbol",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    btw.main()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(payload["windows"]) >= 2
