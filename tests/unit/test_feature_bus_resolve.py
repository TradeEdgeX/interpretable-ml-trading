from __future__ import annotations

from pathlib import Path

from src.live_data_stream.feature_bus import (
    list_feature_bus_timeframe_dirs,
    resolve_disk_primary_timeframe,
)


def test_resolve_disk_primary_prefers_120t(tmp_path: Path) -> None:
    root = tmp_path / "bus"
    (root / "features" / "120T").mkdir(parents=True)
    (root / "features" / "120T" / "BTCUSDT.parquet").write_bytes(b"x")
    tf, legacy = resolve_disk_primary_timeframe(root, "120T")
    assert tf == "120T"
    assert legacy is False
    assert "120T" in list_feature_bus_timeframe_dirs(root)


def test_resolve_disk_primary_legacy_fallback(tmp_path: Path) -> None:
    root = tmp_path / "bus"
    (root / "features" / "primary").mkdir(parents=True)
    (root / "features" / "primary" / "SOLUSDT.parquet").write_bytes(b"x")
    tf, legacy = resolve_disk_primary_timeframe(root, "120T")
    assert tf == "primary"
    assert legacy is True
