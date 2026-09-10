#!/usr/bin/env python3

from pathlib import Path

import yaml

from scripts.locked_prefilter_utils import (
    load_locked_prefilter_rules,
    merge_locked_prefilter_rules,
)


def test_load_locked_prefilter_rules_filters_locked_only(tmp_path: Path):
    prefilter_path = tmp_path / "prefilter.yaml"
    prefilter_path.write_text(
        yaml.safe_dump(
            {
                "rules": [
                    {"feature": "a", "operator": ">=", "value": 0.1, "locked": True},
                    {"feature": "b", "operator": "<=", "value": 0.2},
                ]
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    locked = load_locked_prefilter_rules(prefilter_path)
    assert len(locked) == 1
    assert locked[0]["feature"] == "a"
    assert locked[0]["locked"] is True


def test_merge_locked_prefilter_rules_adds_missing_locked_rule(tmp_path: Path):
    prefilter_path = tmp_path / "prefilter.yaml"
    prefilter_path.write_text(
        yaml.safe_dump(
            {
                "rules": [
                    {"feature": "x", "operator": ">=", "value": 0.0},
                ]
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    locked_rules = [
        {
            "feature": "fer_signed_efficiency_pct",
            "operator": "<=",
            "value": 0.35,
            "locked": True,
            "lock_reason": "semantic anchor",
        }
    ]

    result = merge_locked_prefilter_rules(prefilter_path, locked_rules)
    assert result["added"] == 1

    merged = yaml.safe_load(prefilter_path.read_text(encoding="utf-8")) or {}
    rules = merged.get("rules", [])
    assert any(r.get("feature") == "x" for r in rules)
    added_rule = next(
        r for r in rules if r.get("feature") == "fer_signed_efficiency_pct"
    )
    assert added_rule.get("locked") is True
    assert added_rule.get("lock_reason") == "semantic anchor"


def test_merge_locked_prefilter_rules_deduplicates_by_feature_operator_value(
    tmp_path: Path,
):
    prefilter_path = tmp_path / "prefilter.yaml"
    prefilter_path.write_text(
        yaml.safe_dump(
            {
                "rules": [
                    {
                        "feature": "dist_to_nearest_sr",
                        "operator": "<=",
                        "value": 1.2,
                        "rationale": "existing",
                    },
                ]
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    locked_rules = [
        {
            "feature": "dist_to_nearest_sr",
            "operator": "<=",
            "value": 1.2,
            "locked": True,
            "lock_reason": "anchor",
        }
    ]
    result = merge_locked_prefilter_rules(prefilter_path, locked_rules)
    assert result["added"] == 0

    merged = yaml.safe_load(prefilter_path.read_text(encoding="utf-8")) or {}
    rules = [
        r
        for r in merged.get("rules", [])
        if r.get("feature") == "dist_to_nearest_sr" and r.get("operator") == "<="
    ]
    assert len(rules) == 1
