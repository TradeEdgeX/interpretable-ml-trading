"""Unit tests for oms_state_backup (tmp_path only; no prod FS)."""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.monitoring.oms_state_backup import (
    copy_file,
    dated_dir_name,
    parse_dated_dir,
    plan_prune,
    run_backup,
    sqlite_backup,
    staging_dir_name,
)


def _make_sqlite(path: Path, marker: str = "ok") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE t (v TEXT)")
        conn.execute("INSERT INTO t VALUES (?)", (marker,))
        conn.commit()
    finally:
        conn.close()
    return path


def _seed_root(root: Path) -> None:
    data = root / "data"
    for name in (
        "multi_leg_order_management.db",
        "order_management.db",
        "bc_coin_multileg.db",
        "bc_coin_trend_order_management.db",
        "reconciliation.db",
    ):
        _make_sqlite(data / name, marker=name)

    ml_state = data / "multi_leg_live" / "state"
    ml_state.mkdir(parents=True)
    (ml_state / "chop_grid_BTCUSDT.json").write_text('{"seg":1}\n')
    (ml_state / "fee_churn_state.json").write_text('{"halt":false}\n')
    (ml_state / "logs").mkdir()
    (ml_state / "logs" / "audit.log").write_text("should-not-copy\n")

    bc_ml = data / "bc_coin_multileg_live" / "state"
    bc_ml.mkdir(parents=True)
    (bc_ml / "chop_grid_SOLUSDT.json").write_text('{"seg":2}\n')

    bc_tr = data / "bc_coin_trend_live"
    (bc_tr / "position_tracker").mkdir(parents=True)
    (bc_tr / "fee_churn_state.json").write_text('{"halt":false}\n')
    (bc_tr / "position_tracker" / "ETHUSDT.json").write_text('{"pos":1}\n')

    hc = root / "live" / "highcap" / "data"
    (hc / "position_tracker").mkdir(parents=True)
    (hc / "fee_churn_state.json").write_text('{"halt":false}\n')
    (hc / "position_tracker" / "BTCUSDT.json").write_text('{"pos":1}\n')
    _make_sqlite(hc / "order_management.db", marker="highcap")
    _make_sqlite(hc / "spot_order_management.db", marker="spot_oms")
    _make_sqlite(hc / "spot_accum_ledger.db", marker="spot_ledger")

    roll = data / "rolling_usdt_live" / "btc"
    roll.mkdir(parents=True)
    (roll / "rolling_usdt_state.json").write_text('{"sym":"BTCUSDT"}\n')


PRIMARY_NAMES = (
    "multi_leg_order_management.db",
    "order_management.db",
)


def test_dated_dir_name_and_parse() -> None:
    assert dated_dir_name(date(2026, 7, 15)) == "20260715"
    assert parse_dated_dir("20260715") == date(2026, 7, 15)
    assert parse_dated_dir("backups") is None
    assert staging_dir_name("20260715") == ".20260715.partial"


def test_plan_prune_keeps_last_n_days_and_ignores_flat_files(tmp_path: Path) -> None:
    today = date(2026, 7, 15)
    backup = tmp_path / "backups"
    backup.mkdir()
    keep = backup / "20260709"
    drop = backup / "20260708"
    older = backup / "20260701"
    for d in (keep, drop, older):
        d.mkdir()
        (d / "x").write_text("1")
    flat = backup / "multi_leg_order_management_20260715.db.backup"
    flat.write_text("stray")

    to_delete = plan_prune(backup, retention_days=7, today=today)
    names = {p.name for p in to_delete}
    assert names == {"20260708", "20260701"}
    assert flat.exists()


def test_plan_prune_rejects_retention_zero() -> None:
    with pytest.raises(ValueError, match="retention_days"):
        plan_prune(Path("/tmp"), retention_days=0, today=date(2026, 7, 15))


def test_sqlite_backup_roundtrip(tmp_path: Path) -> None:
    src = _make_sqlite(tmp_path / "live.db", marker="hello")
    dest = tmp_path / "out.db.backup"
    item = sqlite_backup(src, dest, dry_run=False)
    assert item.ok and not item.skipped
    conn = sqlite3.connect(str(dest))
    try:
        row = conn.execute("SELECT v FROM t").fetchone()
    finally:
        conn.close()
    assert row == ("hello",)


def test_sqlite_backup_atomic_keeps_old_on_failure(tmp_path: Path) -> None:
    src = _make_sqlite(tmp_path / "live.db", marker="v1")
    dest = tmp_path / "out.db.backup"
    assert sqlite_backup(src, dest, dry_run=False).ok
    src.write_bytes(b"not-sqlite")
    item = sqlite_backup(src, dest, dry_run=False)
    assert not item.ok
    conn = sqlite3.connect(str(dest))
    try:
        row = conn.execute("SELECT v FROM t").fetchone()
    finally:
        conn.close()
    assert row == ("v1",)


def test_run_backup_writes_manifest_dbs_and_json(tmp_path: Path) -> None:
    root = tmp_path / "quant"
    _seed_root(root)
    backup_dir = root / "data" / "backups"
    day = date(2026, 7, 15)

    report = run_backup(
        root=root,
        backup_dir=backup_dir,
        retention_days=7,
        dry_run=False,
        day=day,
    )
    assert report.ok
    assert not report.primary_failures
    assert not report.critical_failures

    day_dir = backup_dir / "20260715"
    assert (day_dir / "MANIFEST.json").is_file()
    assert not (backup_dir / staging_dir_name("20260715")).exists()
    assert (day_dir / "db" / "multi_leg_order_management.db.backup").is_file()
    assert (day_dir / "state" / "multi_leg_live" / "chop_grid_BTCUSDT.json").is_file()
    assert (
        day_dir / "state" / "bc_coin_multileg_live" / "chop_grid_SOLUSDT.json"
    ).is_file()
    assert (day_dir / "state" / "bc_coin_trend_live" / "fee_churn_state.json").is_file()
    assert (
        day_dir / "state" / "bc_coin_trend_live" / "position_tracker" / "ETHUSDT.json"
    ).is_file()
    assert not (day_dir / "state" / "multi_leg_live" / "logs").exists()
    assert (day_dir / "state" / "highcap" / "spot_accum_ledger.db.backup").is_file()

    manifest = json.loads((day_dir / "MANIFEST.json").read_text())
    assert manifest["ok"] is True
    assert "critical_failures" in manifest


def test_run_backup_prunes_old_dated_dirs(tmp_path: Path) -> None:
    root = tmp_path / "quant"
    _seed_root(root)
    backup_dir = root / "data" / "backups"
    old = backup_dir / "20260701"
    old.mkdir(parents=True)
    (old / "stale").write_text("x")
    recent = backup_dir / "20260714"
    recent.mkdir(parents=True)
    (recent / "ok").write_text("y")

    report = run_backup(
        root=root,
        backup_dir=backup_dir,
        retention_days=7,
        dry_run=False,
        day=date(2026, 7, 15),
    )
    assert report.ok
    assert any("20260701" in p for p in report.pruned)
    assert not old.exists()
    assert recent.exists()


def test_run_backup_skips_prune_when_primary_fails(tmp_path: Path) -> None:
    root = tmp_path / "quant"
    data = root / "data"
    data.mkdir(parents=True)
    (data / "multi_leg_order_management.db").write_bytes(b"not-a-sqlite-database")
    for name in PRIMARY_NAMES[1:]:
        _make_sqlite(data / name)

    backup_dir = data / "backups"
    old = backup_dir / "20260701"
    old.mkdir(parents=True)
    (old / "keep").write_text("x")

    report = run_backup(
        root=root,
        backup_dir=backup_dir,
        retention_days=7,
        dry_run=False,
        day=date(2026, 7, 15),
    )
    assert not report.ok
    assert report.pruned == []
    assert old.exists()
    assert not (backup_dir / staging_dir_name("20260715")).exists()
    last_run = json.loads((backup_dir / "LAST_RUN.json").read_text())
    assert last_run["ok"] is False


def test_run_backup_primary_failure_sets_not_ok(tmp_path: Path) -> None:
    root = tmp_path / "quant"
    data = root / "data"
    data.mkdir(parents=True)
    (data / "multi_leg_order_management.db").write_bytes(b"not-a-sqlite-database")
    for name in PRIMARY_NAMES[1:]:
        _make_sqlite(data / name)

    report = run_backup(
        root=root,
        backup_dir=data / "backups",
        dry_run=False,
        day=date(2026, 7, 15),
    )
    assert not report.ok
    assert "multi_leg_order_management.db" in report.primary_failures


def test_run_backup_missing_primary_fails(tmp_path: Path) -> None:
    root = tmp_path / "quant"
    data = root / "data"
    for name in PRIMARY_NAMES[1:]:
        _make_sqlite(data / name)

    report = run_backup(
        root=root,
        backup_dir=data / "backups",
        dry_run=False,
        day=date(2026, 7, 15),
    )
    assert not report.ok
    assert "multi_leg_order_management.db" in report.primary_failures


def test_run_backup_missing_optional_dapi_dbs_still_ok(tmp_path: Path) -> None:
    """bc_coin_* DBs are optional — absent lane must not fail the backup."""
    root = tmp_path / "quant"
    data = root / "data"
    for name in PRIMARY_NAMES:
        _make_sqlite(data / name)
    # Minimal state trees required by critical JSON globs.
    (data / "multi_leg_live" / "state").mkdir(parents=True)
    (data / "multi_leg_live" / "state" / "chop_grid_BTCUSDT.json").write_text("{}")
    (root / "live" / "highcap" / "data" / "position_tracker").mkdir(parents=True)
    (root / "live" / "highcap" / "data" / "fee_churn_state.json").write_text("{}")
    _make_sqlite(root / "live" / "highcap" / "data" / "order_management.db")
    _make_sqlite(root / "live" / "highcap" / "data" / "spot_order_management.db")
    _make_sqlite(root / "live" / "highcap" / "data" / "spot_accum_ledger.db")
    (data / "rolling_usdt_live" / "btc").mkdir(parents=True)
    (data / "rolling_usdt_live" / "btc" / "rolling_usdt_state.json").write_text("{}")

    report = run_backup(
        root=root,
        backup_dir=data / "backups",
        dry_run=False,
        day=date(2026, 7, 15),
    )
    assert "bc_coin_multileg.db" not in report.primary_failures
    assert "bc_coin_trend_order_management.db" not in report.primary_failures
    assert report.ok or not report.primary_failures


def test_existing_fund_json_copy_failure_is_critical(tmp_path: Path) -> None:
    root = tmp_path / "quant"
    _seed_root(root)
    backup_dir = root / "data" / "backups"

    def _boom(src, dest, *, dry_run, kind="json", critical=False):
        from scripts.monitoring.oms_state_backup import ItemResult

        if src.name == "fee_churn_state.json" and "highcap" in str(src):
            return ItemResult(
                kind=kind,
                source=str(src),
                dest=str(dest),
                ok=False,
                error="forced",
                critical=critical,
            )
        return copy_file(src, dest, dry_run=dry_run, kind=kind, critical=critical)

    with patch("scripts.monitoring.oms_state_backup.copy_file", side_effect=_boom):
        report = run_backup(
            root=root,
            backup_dir=backup_dir,
            dry_run=False,
            day=date(2026, 7, 15),
        )
    assert not report.ok
    assert report.critical_failures
    assert any("fee_churn_state.json" in x for x in report.critical_failures)
    # Failed run must not publish a partial day dir.
    assert not (backup_dir / "20260715").exists()
    assert json.loads((backup_dir / "LAST_RUN.json").read_text())["ok"] is False


def test_publish_failure_does_not_prune_old_recovery_point(tmp_path: Path) -> None:
    root = tmp_path / "quant"
    _seed_root(root)
    backup_dir = root / "data" / "backups"
    old = backup_dir / "20260701"
    old.mkdir(parents=True)
    (old / "keep").write_text("x")

    with patch(
        "scripts.monitoring.oms_state_backup.publish_day_dir",
        side_effect=OSError("forced publish failure"),
    ):
        report = run_backup(
            root=root,
            backup_dir=backup_dir,
            dry_run=False,
            day=date(2026, 7, 15),
        )

    assert not report.ok
    assert any("publish failed" in x for x in report.critical_failures)
    assert old.exists()
    assert report.pruned == []
    assert json.loads((backup_dir / "LAST_RUN.json").read_text())["ok"] is False


def test_same_day_success_drops_stale_json(tmp_path: Path) -> None:
    root = tmp_path / "quant"
    _seed_root(root)
    backup_dir = root / "data" / "backups"
    day = date(2026, 7, 15)
    first = run_backup(root=root, backup_dir=backup_dir, dry_run=False, day=day)
    assert first.ok
    stale = (
        backup_dir / "20260715" / "state" / "multi_leg_live" / "chop_grid_SOLUSDT.json"
    )
    # Inject a stale file into the published day dir (simulates retired symbol).
    stale.write_text('{"old":1}\n')

    # Re-run: staging rebuild must replace whole day dir without the stale file.
    second = run_backup(root=root, backup_dir=backup_dir, dry_run=False, day=day)
    assert second.ok
    assert not stale.exists()
    assert (
        backup_dir / "20260715" / "state" / "multi_leg_live" / "chop_grid_BTCUSDT.json"
    ).is_file()


def test_same_day_rerun_preserves_dest_on_second_failure(tmp_path: Path) -> None:
    root = tmp_path / "quant"
    _seed_root(root)
    backup_dir = root / "data" / "backups"
    day = date(2026, 7, 15)
    first = run_backup(root=root, backup_dir=backup_dir, dry_run=False, day=day)
    assert first.ok
    dest = backup_dir / "20260715" / "db" / "multi_leg_order_management.db.backup"
    marker = dest.read_bytes()

    (root / "data" / "multi_leg_order_management.db").write_bytes(b"corrupt")
    second = run_backup(root=root, backup_dir=backup_dir, dry_run=False, day=day)
    assert not second.ok
    assert dest.is_file()
    assert dest.read_bytes() == marker
    assert (backup_dir / "20260715" / "MANIFEST.json").is_file()


def test_run_backup_dry_run_writes_nothing(tmp_path: Path) -> None:
    root = tmp_path / "quant"
    _seed_root(root)
    backup_dir = root / "data" / "backups"
    report = run_backup(
        root=root,
        backup_dir=backup_dir,
        dry_run=True,
        day=date(2026, 7, 15),
    )
    assert report.ok
    assert not (backup_dir / "20260715").exists()
    assert not list(backup_dir.glob(".*")) if backup_dir.exists() else True


def test_run_backup_missing_optional_paths_ok(tmp_path: Path) -> None:
    root = tmp_path / "quant"
    data = root / "data"
    for name in PRIMARY_NAMES:
        _make_sqlite(data / name)

    report = run_backup(
        root=root,
        backup_dir=data / "backups",
        dry_run=False,
        day=date(2026, 7, 15),
    )
    assert report.ok


def test_run_backup_rejects_retention_zero(tmp_path: Path) -> None:
    root = tmp_path / "quant"
    _seed_root(root)
    with pytest.raises(ValueError, match="retention_days"):
        run_backup(
            root=root,
            backup_dir=root / "data" / "backups",
            retention_days=0,
            day=date(2026, 7, 15),
        )
