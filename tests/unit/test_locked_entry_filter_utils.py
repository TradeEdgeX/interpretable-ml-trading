from __future__ import annotations

from pathlib import Path

import yaml

from scripts.locked_entry_filter_utils import (
    load_locked_entry_filters,
    merge_locked_entry_filters,
)


def test_load_locked_entry_filters_only_returns_locked(tmp_path: Path) -> None:
    p = tmp_path / "entry_filters.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "filters": [
                    {"id": "a", "enabled": True, "locked": True, "conditions": []},
                    {"id": "b", "enabled": True, "conditions": []},
                ],
                "combination_mode": "or",
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    locked = load_locked_entry_filters(p)
    assert [f.get("id") for f in locked] == ["a"]


def test_merge_locked_entry_filters_appends_missing_as_disabled(tmp_path: Path) -> None:
    p = tmp_path / "entry_filters.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "filters": [
                    {"id": "x", "enabled": True, "conditions": []},
                ],
                "combination_mode": "or",
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    locked = [
        {
            "id": "locked_a",
            "locked": True,
            "enabled": True,
            "conditions": [{"feature": "ofci_pct", "operator": ">=", "value": 0.7}],
        }
    ]
    stat = merge_locked_entry_filters(p, locked)
    assert stat["added"] == 1

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    by_id = {f.get("id"): f for f in (raw.get("filters") or [])}
    assert "locked_a" in by_id
    assert by_id["locked_a"]["locked"] is True
    assert by_id["locked_a"]["enabled"] is False
