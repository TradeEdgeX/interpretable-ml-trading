"""Unit tests for depth snapshot downloader helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.data_tools.download_depth_snapshots import DepthSnapshotDownloader


def test_poll_once_parses_mock_depth(tmp_path):
    dl = DepthSnapshotDownloader(parquet_dir=tmp_path)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "bids": [["64900", "100"], ["64800", "200"]],
        "asks": [["65100", "50"], ["65200", "30"]],
    }
    with patch.object(dl.session, "get", return_value=mock_resp):
        df = dl.poll_once(
            "BTCUSDT", ts=datetime(2026, 6, 15, 10, 0, tzinfo=timezone.utc)
        )
    assert len(df) == 1
    assert df["wall_bid_notional_usd_max"].iloc[0] > 0
    assert df["_symbol"].iloc[0] == "BTCUSDT"


def test_append_snapshot_merges_same_day(tmp_path):
    dl = DepthSnapshotDownloader(parquet_dir=tmp_path)
    ts1 = datetime(2026, 6, 15, 10, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 6, 15, 11, 0, tzinfo=timezone.utc)
    df1 = pd.DataFrame(
        {"_symbol": "ETHUSDT", "wall_bid_notional_usd_max": [1e7]},
        index=pd.DatetimeIndex([ts1]),
    )
    df2 = pd.DataFrame(
        {"_symbol": "ETHUSDT", "wall_bid_notional_usd_max": [2e7]},
        index=pd.DatetimeIndex([ts2]),
    )
    dl.append_snapshot("ETHUSDT", df1)
    path = dl.append_snapshot("ETHUSDT", df2)
    merged = pd.read_parquet(path)
    assert len(merged) == 2
