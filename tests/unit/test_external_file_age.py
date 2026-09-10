"""Tests for process-external OMS/monitor file-age metrics."""

from __future__ import annotations

from pathlib import Path

from src.monitoring.external_file_age import (
    collect_file_ages,
    file_age_seconds,
    render_prometheus_text,
    resolve_watched_files,
    update_external_file_age_gauges,
)


def test_file_age_uses_newer_wal(tmp_path: Path, monkeypatch) -> None:
    import os
    import time

    oms = tmp_path / "order_management.db"
    oms.write_bytes(b"db")
    wal = tmp_path / "order_management.db-wal"
    wal.write_bytes(b"wal")
    # Make .db look old, -wal look fresh
    old = time.time() - 10_000
    os.utime(oms, (old, old))
    age = file_age_seconds(oms)
    assert age is not None
    assert age < 60  # dominated by fresh -wal


def test_file_age_missing() -> None:
    assert file_age_seconds(Path("/nonexistent/oms.db")) is None


def test_collect_and_render(tmp_path: Path) -> None:
    oms = tmp_path / "order_management.db"
    oms.write_bytes(b"x")
    ages = collect_file_ages({"trend_oms": oms, "missing": tmp_path / "no.db"})
    assert "trend_oms" in ages
    assert ages["trend_oms"] >= 0
    assert "missing" not in ages
    text = render_prometheus_text(ages)
    assert 'mlbot_external_file_age_seconds{role="trend_oms"}' in text


def test_update_gauges(tmp_path: Path) -> None:
    class _G:
        def __init__(self) -> None:
            self.values: dict[str, float] = {}

        def labels(self, role: str):
            g = self

            class _L:
                def set(self, v: float) -> None:
                    g.values[role] = float(v)

            return _L()

    oms = tmp_path / "order_management.db"
    oms.write_bytes(b"x")
    gauge = _G()
    out = update_external_file_age_gauges(gauge, paths={"trend_oms": oms})
    assert out["trend_oms"] >= 0
    assert gauge.values["trend_oms"] >= 0


def test_resolve_live_monitor_prefers_db_subdir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MLBOT_EXTERNAL_FILE_AGE_LIVE_MONITOR", raising=False)
    monkeypatch.delenv("MLBOT_LIVE_STORAGE_BASE", raising=False)
    monkeypatch.delenv("MLBOT_CONSOLE_LIVE_DATA_ROOT", raising=False)
    live_data = tmp_path / "live_data"
    (live_data / "db").mkdir(parents=True)
    mon = live_data / "db" / "live_monitor.db"
    mon.write_bytes(b"x")
    monkeypatch.setenv("MLBOT_LIVE_STORAGE_BASE", str(live_data))
    paths = resolve_watched_files(data_root=tmp_path / "engine")
    assert paths["live_monitor"] == mon


def test_resolve_watched_files_uses_data_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MLBOT_EXTERNAL_FILE_AGE_DATA_ROOT", raising=False)
    monkeypatch.delenv("MLBOT_CONSOLE_ENGINE_DATA_ROOT", raising=False)
    monkeypatch.delenv("MLBOT_ENGINE_DATA_ROOT", raising=False)
    monkeypatch.delenv("MLBOT_LIVE_STORAGE_BASE", raising=False)
    monkeypatch.delenv("MLBOT_CONSOLE_LIVE_DATA_ROOT", raising=False)
    for role, _rel in (
        ("trend_oms", "order_management.db"),
        ("bc_trend_oms", "bc_coin_trend_order_management.db"),
    ):
        monkeypatch.delenv(f"MLBOT_EXTERNAL_FILE_AGE_{role.upper()}", raising=False)
    paths = resolve_watched_files(data_root=tmp_path)
    assert paths["trend_oms"] == tmp_path / "order_management.db"
