"""backfill_plateau_baseline dry-run / write roundtrip."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.research.writeback.plateau_baseline import (
    load_yaml,
    merge_plateau_baseline_from_json,
    write_yaml,
)


def test_merge_plateau_baseline_dry_run(tmp_path: Path) -> None:
    target = tmp_path / "entry_filters.yaml"
    target.write_text(
        yaml.dump({"filters": [], "last_calibration": {"timestamp": None}}),
        encoding="utf-8",
    )
    plateau = tmp_path / "plateau.json"
    plateau.write_text(
        json.dumps(
            {
                "feature": "pulse_z",
                "operator": "<=",
                "start_threshold": 0.1,
                "end_threshold": 0.3,
                "recommended": 0.2,
                "is_plateau": True,
            }
        ),
        encoding="utf-8",
    )
    raw = load_yaml(target)
    payload = json.loads(plateau.read_text(encoding="utf-8"))
    updated = merge_plateau_baseline_from_json(raw, payload)
    entry = updated["last_calibration"]["plateaus"][-1]
    assert entry["feature"] == "pulse_z"
    assert entry["action"] == "BASELINE"
    assert entry["plateau"]["start"] == 0.1
    write_yaml(tmp_path / "out.yaml", updated)
    assert (tmp_path / "out.yaml").is_file()
