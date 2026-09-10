"""Tests for gate_when utilities and calibrate all_of preservation."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.research.calibrate import calibrate_draft_result, calibrate_draft_text
from src.research.gate_when import (
    apply_gate_threshold_to_when,
    can_apply_gate_threshold,
    resolve_gate_deny_operator,
)


def test_resolve_gate_deny_operator_gt() -> None:
    assert resolve_gate_deny_operator(">") == "gt"
    assert resolve_gate_deny_operator("gt") == "gt"
    assert resolve_gate_deny_operator("value_gt") == "gt"


def test_apply_gate_threshold_preserves_all_of_siblings() -> None:
    when = {
        "all_of": [
            {"vol_persistence": {"value_gt": 0.0029}},
            {"vol_persistence": {"value_lt": 0.0616}},
            {"ema_1200_position": {"value_gt": 0.10}},
        ]
    }
    updated = apply_gate_threshold_to_when(
        when,
        "vol_persistence",
        "gt",
        0.003,
        interval=(0.003, 0.05),
    )
    assert "all_of" in updated
    clauses = updated["all_of"]
    assert any("ema_1200_position" in c for c in clauses)
    assert sum(1 for c in clauses if "vol_persistence" in c) == 2
    assert not any(
        c.get("vol_persistence", {}).get("value_gt") == 0.0029 for c in clauses
    )


def test_any_of_rule_left_unchanged() -> None:
    when = {
        "any_of": [
            {"vol_persistence": {"value_gt": 0.01}},
            {"evt_var_99": {"value_gt": 0.5}},
        ]
    }
    assert can_apply_gate_threshold(when, "vol_persistence") is False
    updated = apply_gate_threshold_to_when(when, "vol_persistence", "gt", 0.02)
    assert updated == when  # untouched, OR semantics preserved


def test_band_rule_without_interval_left_unchanged() -> None:
    when = {
        "all_of": [
            {"evt_var_99": {"value_gt": 0.67}},
            {"evt_var_99": {"value_lt": 0.79}},
        ]
    }
    # single point would drop the upper bound → must stay unchanged
    assert can_apply_gate_threshold(when, "evt_var_99", interval=None) is False
    updated = apply_gate_threshold_to_when(when, "evt_var_99", "gt", 0.70)
    assert updated == when


def test_band_rule_with_interval_rewrites_both_bounds() -> None:
    when = {
        "all_of": [
            {"evt_var_99": {"value_gt": 0.67}},
            {"evt_var_99": {"value_lt": 0.79}},
        ]
    }
    updated = apply_gate_threshold_to_when(
        when, "evt_var_99", "gt", 0.70, interval=(0.70, 0.85)
    )
    clauses = updated["all_of"]
    gts = [
        c["evt_var_99"]["value_gt"]
        for c in clauses
        if "value_gt" in c.get("evt_var_99", {})
    ]
    lts = [
        c["evt_var_99"]["value_lt"]
        for c in clauses
        if "value_lt" in c.get("evt_var_99", {})
    ]
    assert gts == [0.70]
    assert lts == [0.85]


def test_calibrate_batch_preserves_regime_conditional_rule(tmp_path: Path) -> None:
    arch = tmp_path / "strategies" / "tpc" / "archetypes"
    arch.mkdir(parents=True)
    arch.joinpath("gate.yaml").write_text(
        yaml.safe_dump(
            {
                "system_safety": [
                    {
                        "id": "gate_vol_test",
                        "when": {
                            "all_of": [
                                {"vol_persistence": {"value_gt": 0.01}},
                                {"vol_persistence": {"value_lt": 0.05}},
                                {"ema_1200_position": {"value_gt": 0.10}},
                            ]
                        },
                        "then": {"action": "deny"},
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    batch = {
        "kpi": "lift",
        "strategy": "tpc",
        "rules": {
            "gate_vol_test": {
                "feature": "vol_persistence",
                "operator": "gt",
                "status": "stable_plateau_found",
                "recommended_threshold": 0.012,
                "threshold_interval": {"start": 0.012, "end": 0.048},
            }
        },
    }
    src = tmp_path / "batch.json"
    src.write_text(json.dumps(batch), encoding="utf-8")
    text = calibrate_draft_text(
        batch,
        src,
        strategy="tpc",
        strategies_root=tmp_path / "strategies",
    )
    draft = yaml.safe_load(
        "\n".join(l for l in text.splitlines() if not l.startswith("#"))
    )
    rule = draft["system_safety"][0]
    clauses = rule["when"]["all_of"]
    assert any("ema_1200_position" in c for c in clauses)
    assert sum(1 for c in clauses if "vol_persistence" in c) == 2


def test_calibrate_batch_records_unsafe_any_of_skip(tmp_path: Path) -> None:
    arch = tmp_path / "strategies" / "tpc" / "archetypes"
    arch.mkdir(parents=True)
    arch.joinpath("gate.yaml").write_text(
        yaml.safe_dump(
            {
                "hard_gates": [
                    {
                        "id": "gate_anyof",
                        "when": {
                            "any_of": [
                                {"vol_persistence": {"value_gt": 0.01}},
                                {"tpc_semantic_chop": {"value_gt": 0.9}},
                            ]
                        },
                        "then": {"action": "deny"},
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    batch = {
        "kpi": "lift",
        "strategy": "tpc",
        "rules": {
            "gate_anyof": {
                "feature": "vol_persistence",
                "operator": "gt",
                "status": "stable_plateau_found",
                "recommended_threshold": 0.02,
            }
        },
    }
    src = tmp_path / "batch.json"
    src.write_text(json.dumps(batch), encoding="utf-8")
    result = calibrate_draft_result(
        batch,
        src,
        strategy="tpc",
        strategies_root=tmp_path / "strategies",
    )
    assert len(result.skips) == 1
    assert result.skips[0].rule_id == "gate_anyof"
    assert result.skips[0].reason == "unsafe_any_of"
    assert "calibrate skips (1)" in result.draft_text


def test_calibrate_batch_records_band_without_interval_skip(tmp_path: Path) -> None:
    arch = tmp_path / "strategies" / "tpc" / "archetypes"
    arch.mkdir(parents=True)
    arch.joinpath("gate.yaml").write_text(
        yaml.safe_dump(
            {
                "system_safety": [
                    {
                        "id": "gate_band",
                        "when": {
                            "all_of": [
                                {"evt_var_99": {"value_gt": 0.67}},
                                {"evt_var_99": {"value_lt": 0.79}},
                            ]
                        },
                        "then": {"action": "deny"},
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    batch = {
        "kpi": "lift",
        "strategy": "tpc",
        "rules": {
            "gate_band": {
                "feature": "evt_var_99",
                "operator": "gt",
                "status": "stable_plateau_found",
                "recommended_threshold": 0.70,
            }
        },
    }
    src = tmp_path / "batch.json"
    src.write_text(json.dumps(batch), encoding="utf-8")
    result = calibrate_draft_result(
        batch,
        src,
        strategy="tpc",
        strategies_root=tmp_path / "strategies",
    )
    assert len(result.skips) == 1
    assert result.skips[0].reason == "unsafe_band_no_interval"
