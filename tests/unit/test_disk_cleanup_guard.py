"""Unit tests for disk_cleanup_guard's pure planning/deletion logic.

These tests never touch a real Docker daemon or journald — only the
filesystem-scoped helpers (find_old_files / plan_backup_cleanup /
delete_paths) and the threshold gate in run_cleanup (with docker/journal
steps skipped).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

from scripts.monitoring.disk_cleanup_guard import (
    CleanupConfig,
    backup_group_key,
    cleanup_still_above_threshold,
    delete_paths,
    find_old_files,
    human_size,
    plan_backup_cleanup,
    run_cleanup,
    _format_telegram_message,
)


def _touch(path: Path, *, age_days: float = 0.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x" * 10)
    if age_days:
        mtime = time.time() - age_days * 86400
        os.utime(path, (mtime, mtime))
    return path


# ── human_size ───────────────────────────────────────────────────────────


def test_human_size_formats_units() -> None:
    assert human_size(500) == "500.0B"
    assert human_size(2048) == "2.0KB"
    assert human_size(5 * 1024 * 1024) == "5.0MB"
    assert human_size(3 * 1024**3) == "3.0GB"


# ── find_old_files ───────────────────────────────────────────────────────


def test_find_old_files_only_returns_files_older_than_cutoff(tmp_path: Path) -> None:
    old = _touch(tmp_path / "old.zip", age_days=10)
    _touch(tmp_path / "new.zip", age_days=1)

    found = find_old_files(tmp_path, max_age_days=7)
    assert found == [old]


def test_find_old_files_missing_directory_is_noop(tmp_path: Path) -> None:
    assert find_old_files(tmp_path / "does_not_exist", max_age_days=7) == []


def test_find_old_files_respects_pattern(tmp_path: Path) -> None:
    old_log = _touch(tmp_path / "trend.log.1", age_days=20)
    _touch(tmp_path / "data.parquet", age_days=20)  # not *.log* — must be ignored

    found = find_old_files(tmp_path, max_age_days=14, pattern="*.log*")
    assert found == [old_log]


# ── backup_group_key / plan_backup_cleanup ──────────────────────────────


def test_backup_group_key_matches_bak_and_backup_suffixes() -> None:
    assert (
        backup_group_key(Path("multi_leg_order_management.db.bak.20260619035733"))
        == "multi_leg_order_management.db"
    )
    assert (
        backup_group_key(Path("order_management.db.backup_20260618_160618"))
        == "order_management.db"
    )
    assert (
        backup_group_key(
            Path("multi_leg_order_management.db.bak_stale_fix_20260520_030126Z")
        )
        == "multi_leg_order_management.db"
    )


def test_backup_group_key_ignores_non_backup_files() -> None:
    assert backup_group_key(Path("multi_leg_order_management.db")) is None
    assert backup_group_key(Path("order_management.db.backup")) is None  # no _ suffix


def test_plan_backup_cleanup_always_keeps_newest_per_db(tmp_path: Path) -> None:
    """Regression guard: the newest backup must never be a deletion
    candidate, even if it happens to be older than max_age_days — it is the
    only rollback point for that DB."""
    _touch(tmp_path / "order_management.db.bak_a", age_days=60)
    newest = _touch(tmp_path / "order_management.db.bak_b", age_days=45)
    _touch(tmp_path / "order_management.db", age_days=0)  # live db — never touched

    to_delete = plan_backup_cleanup(tmp_path, max_age_days=14)

    assert newest not in to_delete
    assert (tmp_path / "order_management.db.bak_a") in to_delete
    assert (tmp_path / "order_management.db") not in to_delete


def test_plan_backup_cleanup_skips_backups_newer_than_threshold(tmp_path: Path) -> None:
    old = _touch(tmp_path / "order_management.db.bak_old", age_days=30)
    _touch(tmp_path / "order_management.db.bak_newest", age_days=1)
    recent = _touch(tmp_path / "order_management.db.bak_recent", age_days=5)

    to_delete = plan_backup_cleanup(tmp_path, max_age_days=14)

    assert old in to_delete
    assert recent not in to_delete  # newer than 14d cutoff


def test_plan_backup_cleanup_never_touches_managed_backups_subdir(
    tmp_path: Path,
) -> None:
    """Managed daily backups live in a backups/ subdir (owned by
    DatabaseBackup, 30d self-prune). plan_backup_cleanup scans the top level
    only, so those must never be candidates."""
    managed = tmp_path / "backups"
    managed.mkdir()
    old_managed = _touch(managed / "order_management_20260101.db.backup", age_days=90)
    stray = _touch(tmp_path / "order_management.db.bak.20260101000000", age_days=90)
    _touch(tmp_path / "order_management.db.bak.20260601000000", age_days=1)

    to_delete = plan_backup_cleanup(tmp_path, max_age_days=14)

    assert old_managed not in to_delete
    assert stray in to_delete


def test_plan_backup_cleanup_never_matches_live_db_or_wal_shm(tmp_path: Path) -> None:
    live = _touch(tmp_path / "order_management.db", age_days=90)
    wal = _touch(tmp_path / "order_management.db-wal", age_days=90)
    shm = _touch(tmp_path / "order_management.db-shm", age_days=90)

    to_delete = plan_backup_cleanup(tmp_path, max_age_days=14)

    assert live not in to_delete
    assert wal not in to_delete
    assert shm not in to_delete


def test_plan_backup_cleanup_groups_independently_per_db(tmp_path: Path) -> None:
    a_old = _touch(tmp_path / "multi_leg_order_management.db.bak_old", age_days=30)
    _touch(tmp_path / "multi_leg_order_management.db.bak_newest", age_days=1)
    b_old = _touch(tmp_path / "order_management.db.bak_old", age_days=30)
    _touch(tmp_path / "order_management.db.bak_newest", age_days=1)

    to_delete = set(plan_backup_cleanup(tmp_path, max_age_days=14))

    assert to_delete == {a_old, b_old}


# ── delete_paths ─────────────────────────────────────────────────────────


def test_delete_paths_dry_run_does_not_remove_files(tmp_path: Path) -> None:
    f = _touch(tmp_path / "a.zip")
    count, freed = delete_paths([f], dry_run=True)
    assert count == 1
    assert freed == 10
    assert f.exists()


def test_delete_paths_removes_files_and_counts_bytes(tmp_path: Path) -> None:
    f1 = _touch(tmp_path / "a.zip")
    f2 = _touch(tmp_path / "b.zip")
    count, freed = delete_paths([f1, f2], dry_run=False)
    assert count == 2
    assert freed == 20
    assert not f1.exists()
    assert not f2.exists()


def test_delete_paths_skips_missing_files_without_error(tmp_path: Path) -> None:
    missing = tmp_path / "gone.zip"
    count, freed = delete_paths([missing], dry_run=False)
    assert count == 0
    assert freed == 0


# ── run_cleanup threshold gate ───────────────────────────────────────────


def test_run_cleanup_noop_below_threshold(tmp_path: Path) -> None:
    with patch(
        "scripts.monitoring.disk_cleanup_guard.disk_usage_percent", return_value=50.0
    ):
        cfg = CleanupConfig(
            threshold_percent=90.0,
            warmup_dir=tmp_path / "warmup",
            log_dirs=(tmp_path / "logs",),
            backup_dir=tmp_path,
            skip_docker=True,
            skip_journal=True,
        )
        result = run_cleanup(cfg)

    assert result["ran"] is False
    assert "50.0" in result["reason"]


def test_run_cleanup_force_ignores_threshold_and_deletes_old_files(
    tmp_path: Path,
) -> None:
    warmup_dir = tmp_path / "warmup"
    old_tick = _touch(warmup_dir / "old.zip", age_days=10)
    _touch(warmup_dir / "new.zip", age_days=1)

    with patch(
        "scripts.monitoring.disk_cleanup_guard.disk_usage_percent", return_value=50.0
    ):
        cfg = CleanupConfig(
            threshold_percent=90.0,
            force=True,
            warmup_dir=warmup_dir,
            warmup_max_age_days=7,
            log_dirs=(tmp_path / "logs",),
            backup_dir=tmp_path / "backups",
            skip_docker=True,
            skip_journal=True,
        )
        result = run_cleanup(cfg)

    assert result["ran"] is True
    assert not old_tick.exists()
    assert (warmup_dir / "new.zip").exists()
    assert result["files_deleted"] == 1


def test_run_cleanup_skips_db_snapshots_by_default(tmp_path: Path) -> None:
    """The unattended timer must never auto-delete db snapshots unless the
    opt-in flag is set — this is the highest-consequence action."""
    old_snap = _touch(tmp_path / "order_management.db.bak.20260101000000", age_days=60)
    _touch(tmp_path / "order_management.db.bak.20260601000000", age_days=1)

    with patch(
        "scripts.monitoring.disk_cleanup_guard.disk_usage_percent", return_value=95.0
    ):
        cfg = CleanupConfig(
            threshold_percent=90.0,
            warmup_dir=tmp_path / "warmup",
            log_dirs=(tmp_path / "logs",),
            backup_dir=tmp_path,
            clean_db_snapshots=False,  # default
            skip_docker=True,
            skip_journal=True,
        )
        result = run_cleanup(cfg)

    step_names = {s["name"] for s in result["steps"]}
    assert "oms_db_snapshots" not in step_names
    assert old_snap.exists()  # never touched by default


def test_run_cleanup_deletes_stray_db_snapshots_when_opted_in(tmp_path: Path) -> None:
    old_snap = _touch(tmp_path / "order_management.db.bak.20260101000000", age_days=60)
    newest_snap = _touch(
        tmp_path / "order_management.db.bak.20260601000000", age_days=1
    )
    live_db = _touch(tmp_path / "order_management.db", age_days=0)

    with patch(
        "scripts.monitoring.disk_cleanup_guard.disk_usage_percent", return_value=95.0
    ):
        cfg = CleanupConfig(
            threshold_percent=90.0,
            warmup_dir=tmp_path / "warmup",
            log_dirs=(tmp_path / "logs",),
            backup_dir=tmp_path,
            backup_max_age_days=14,
            clean_db_snapshots=True,
            skip_docker=True,
            skip_journal=True,
        )
        result = run_cleanup(cfg)

    step_names = {s["name"] for s in result["steps"]}
    assert "oms_db_snapshots" in step_names
    assert not old_snap.exists()
    assert newest_snap.exists()  # newest snapshot kept as rollback point
    assert live_db.exists()  # live db never touched


def test_run_cleanup_dry_run_reports_without_deleting(tmp_path: Path) -> None:
    warmup_dir = tmp_path / "warmup"
    old_tick = _touch(warmup_dir / "old.zip", age_days=10)

    with patch(
        "scripts.monitoring.disk_cleanup_guard.disk_usage_percent", return_value=95.0
    ):
        cfg = CleanupConfig(
            threshold_percent=90.0,
            dry_run=True,
            warmup_dir=warmup_dir,
            warmup_max_age_days=7,
            log_dirs=(tmp_path / "logs",),
            backup_dir=tmp_path / "backups",
            skip_docker=True,
            skip_journal=True,
        )
        result = run_cleanup(cfg)

    assert result["ran"] is True
    assert result["dry_run"] is True
    assert old_tick.exists()  # nothing actually deleted
    assert result["files_deleted"] == 1


def test_cleanup_config_from_env_reads_overrides(monkeypatch) -> None:
    monkeypatch.setenv("MLBOT_DISK_CLEANUP_THRESHOLD_PERCENT", "80")
    monkeypatch.setenv("MLBOT_DISK_CLEANUP_WARMUP_MAX_AGE_DAYS", "3")
    monkeypatch.setenv("MLBOT_DISK_CLEANUP_LOG_DIRS", "/tmp/a,/tmp/b")

    cfg = CleanupConfig.from_env()

    assert cfg.threshold_percent == 80.0
    assert cfg.warmup_max_age_days == 3.0
    assert cfg.log_dirs == (Path("/tmp/a"), Path("/tmp/b"))


def test_cleanup_still_above_threshold_true_when_usage_remains_high() -> None:
    summary = {
        "ran": True,
        "threshold_percent": 90.0,
        "usage_after_percent": 91.2,
    }
    assert cleanup_still_above_threshold(summary) is True


def test_cleanup_still_above_threshold_false_when_usage_drops_below_threshold() -> None:
    summary = {
        "ran": True,
        "threshold_percent": 90.0,
        "usage_after_percent": 87.5,
    }
    assert cleanup_still_above_threshold(summary) is False


def test_format_telegram_message_mentions_manual_cleanup() -> None:
    msg = _format_telegram_message(
        {
            "threshold_percent": 90.0,
            "usage_before_percent": 92.1,
            "usage_after_percent": 90.4,
            "human_freed": "1.2GB",
            "files_deleted": 42,
        }
    )
    assert "磁盘自动清理无效" in msg
    assert "请人工" in msg
    assert "92.1% → 90.4%" in msg
