"""Unit tests for monitor manifest runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.monitoring.run_monitor_manifest import execute_manifest, _load_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEEKLY_MANIFEST = PROJECT_ROOT / "config" / "monitoring" / "weekly_rule_stack.yaml"


def test_window_key_legacy_short_long_aliases():
    manifest = {
        "windows": {
            "near": {"parquet": "a.parquet"},
            "deep": {"parquet": "b.parquet"},
        }
    }
    from scripts.monitoring.run_monitor_manifest import _window_cfg

    assert _window_cfg(manifest, "short")["parquet"] == "a.parquet"
    assert _window_cfg(manifest, "long")["parquet"] == "b.parquet"


def test_weekly_manifest_loads_and_has_four_steps():
    manifest = _load_manifest(WEEKLY_MANIFEST)
    assert manifest["monitor_id"] == "weekly_rule_stack"
    assert manifest.get("strategies_source") == "constitution"
    steps = manifest["steps"]
    assert [next(iter(s)) for s in steps] == [
        "export-window",
        "export-window",
        "watchdog",
        "drift",
    ]
    assert manifest["windows"]["deep"]["source"] == "feature_bus_export"
    assert "near" in manifest["windows"]
    assert "deep" in manifest["windows"]


def test_weekly_c_manifest_has_watchdog_c_step():
    path = PROJECT_ROOT / "config/monitoring/weekly_c_regime.yaml"
    manifest = _load_manifest(path)
    assert manifest["strategies_layer"] == "multi_leg"
    assert [next(iter(s)) for s in manifest["steps"]] == [
        "export-window",
        "watchdog-c",
    ]


def test_execute_manifest_dry_run_substitutes_run_ts(capsys):
    manifest = yaml.safe_load(WEEKLY_MANIFEST.read_text(encoding="utf-8"))
    rc, run_ts, _out = execute_manifest(
        manifest,
        config_path=WEEKLY_MANIFEST,
        run_ts="20260101_1200",
        dry_run=True,
    )
    assert rc == 0
    assert run_ts == "20260101_1200"
    out = capsys.readouterr().out
    assert "20260101_1200" in out
    assert "features_current_7d.parquet" in out
    assert "features_current_deep.parquet" in out
    assert out.count("[dry-run] export-window") == 2
    assert "[dry-run] watchdog" in out
    assert "[dry-run] drift" in out


def test_execute_manifest_rejects_unknown_step(tmp_path):
    manifest = {
        "monitor_id": "bad",
        "windows": {"near": {"parquet": str(tmp_path / "near.parquet")}},
        "steps": [{"noop": {}}],
    }
    with pytest.raises(ValueError, match="unknown manifest step"):
        execute_manifest(
            manifest,
            config_path=tmp_path / "bad.yaml",
            run_ts="20260101_1200",
            dry_run=True,
        )


def test_execute_manifest_writes_heartbeat_on_success(tmp_path, monkeypatch):
    """Watchdog/drift are stubbed (supports both in-process and legacy subprocess paths)."""
    manifest = {
        "monitor_id": "test_stack",
        "output_dir": str(tmp_path / "out/{run_ts}"),
        "windows": {
            "near": {"parquet": str(tmp_path / "near.parquet")},
            "deep": {"parquet": str(tmp_path / "deep.parquet")},
        },
        "strategies": ["tpc"],
        "steps": [
            {"watchdog": {"window": "near"}},
            {"drift": {"window": "deep"}},
        ],
    }
    (tmp_path / "near.parquet").write_bytes(b"")  # not read when mocked
    (tmp_path / "deep.parquet").write_bytes(b"")

    def fake_run(script: str, argv):  # noqa: ANN001
        return 0

    import scripts.monitoring.run_monitor_manifest as mod

    monkeypatch.setattr(mod, "_run_monitor_script", fake_run)

    # Force legacy subprocess path for this test so the existing monkeypatch is sufficient.
    # The test only cares that heartbeat is written, not the execution engine.
    monkeypatch.setenv("MLBOT_MONITOR_FORCE_SUBPROCESS", "1")

    rc, _, out_dir = execute_manifest(
        manifest,
        config_path=tmp_path / "m.yaml",
        run_ts="20260102_0000",
        dry_run=False,
    )
    assert rc == 0
    assert out_dir == tmp_path / "out" / "20260102_0000"
    hb_path = tmp_path / "out" / "20260102_0000" / "heartbeat.json"
    assert hb_path.is_file()
    hb = json.loads(hb_path.read_text(encoding="utf-8"))
    assert hb["status"] == "OK"
    assert hb["task"] == "test_stack"


def test_execute_manifest_watchdog_in_process(tmp_path, monkeypatch):
    """watchdog step runs via direct function call (in-process) by default."""
    import argparse

    # Create dummy parquet files so the code gets past file-existence checks
    short_pq = tmp_path / "near.parquet"
    short_pq.write_bytes(b"")  # 0-byte is fine — we monkeypatch before read

    manifest = {
        "monitor_id": "inproc_watchdog",
        "output_dir": str(tmp_path / "out/{run_ts}"),
        "windows": {"near": {"parquet": str(short_pq)}},
        "strategies": ["tpc"],
        "steps": [{"watchdog": {"window": "near"}}],
    }

    monkeypatch.delenv("MLBOT_MONITOR_FORCE_SUBPROCESS", raising=False)

    called = {"flag": False}

    def fake_run_watchdog(ns: argparse.Namespace) -> int:
        called["flag"] = True
        assert str(ns.window_parquet).endswith("near.parquet")
        return 0

    # Patch at the module that will import it at runtime
    import scripts.regime_watchdog as rw_mod

    monkeypatch.setattr(rw_mod, "run_watchdog", fake_run_watchdog)

    rc, _, _ = execute_manifest(
        manifest,
        config_path=tmp_path / "m.yaml",
        run_ts="20260103_1200",
        dry_run=False,
    )
    assert rc == 0
    assert called["flag"] is True
