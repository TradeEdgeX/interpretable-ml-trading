"""Unit tests for scripts/check_truth_sync_health.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_check_truth_sync_health_dry_run_includes_days(tmp_path: Path) -> None:
    db_path = tmp_path / "missing.db"
    script = _REPO_ROOT / "scripts" / "check_truth_sync_health.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--days",
            "3",
            "--db-path",
            str(db_path),
            "--state-dir",
            str(tmp_path / "tracker"),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(_REPO_ROOT),
    )
    report = json.loads(proc.stdout)
    assert report["days"] == 3
    metrics_check = next(
        c for c in report["checks"] if c["check"] == "metrics_no_duplicate_closed"
    )
    assert metrics_check["days"] == 3
    assert "3 trading day" in metrics_check["expect"]
    assert report["summary"]["overall"] == "PASS"
