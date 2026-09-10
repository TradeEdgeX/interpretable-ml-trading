"""Unit tests for scripts/locked_gate_utils.py"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture()
def gate_yaml(tmp_path: Path) -> Path:
    """Create a sample gate.yaml with locked + non-locked rules."""
    content = textwrap.dedent(
        """\
        schema:
          phases: [system_safety, hard_gate, guardrail]
          governance:
            selection_method: gate_score
            min_gate_score: 0.0
        hard_gates:
          - id: gate_evt_var_99
            tag: HARD_EVT_VAR_99
            phase: hard_gate
            priority: 10
            reason: test rule
            when:
              all_of:
                - evt_var_99:
                    value_gt: 0.67
                - evt_var_99:
                    value_lt: 0.80
            then:
              action: deny
            locked: true
            lock_reason: "EVT 左尾风险"
          - id: gate_unlocked
            tag: HARD_UNLOCKED
            phase: hard_gate
            priority: 11
            reason: unlocked rule
            when:
              some_feature:
                value_gt: 0.5
            then:
              action: deny
        system_safety:
          - id: gate_vol_persistence
            tag: HARD_STAT_3
            phase: system_safety
            priority: 22
            reason: stat fallback
            when:
              all_of:
                - vol_persistence:
                    value_gt: 0.03
                - vol_persistence:
                    value_lt: 0.06
            then:
              action: deny
            locked: true
            lock_reason: "波动率持久性"
        guardrails: []
    """
    )
    p = tmp_path / "gate.yaml"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """DataFrame with features and is_bad label."""
    rng = np.random.default_rng(42)
    n = 500
    evt = rng.uniform(0, 1, n)
    vol = rng.uniform(0, 0.15, n)
    bad = (evt > 0.6) & (evt < 0.85)
    bad = bad | (rng.random(n) < 0.1)
    return pd.DataFrame(
        {
            "evt_var_99": evt,
            "vol_persistence": vol,
            "some_feature": rng.uniform(0, 1, n),
            "is_bad": bad.astype(int),
        }
    )


class TestLoadLockedGateRules:
    def test_loads_only_locked(self, gate_yaml: Path):
        from scripts.locked_gate_utils import load_locked_gate_rules

        locked = load_locked_gate_rules(gate_yaml)
        assert len(locked) == 2
        ids = {r["id"] for r in locked}
        assert "gate_evt_var_99" in ids
        assert "gate_vol_persistence" in ids
        assert "gate_unlocked" not in ids

    def test_empty_file(self, tmp_path: Path):
        from scripts.locked_gate_utils import load_locked_gate_rules

        p = tmp_path / "empty.yaml"
        p.write_text("hard_gates: []\n", encoding="utf-8")
        assert load_locked_gate_rules(p) == []

    def test_nonexistent(self, tmp_path: Path):
        from scripts.locked_gate_utils import load_locked_gate_rules

        assert load_locked_gate_rules(tmp_path / "nope.yaml") == []


class TestMergeLockedGateRules:
    def test_merge_adds_missing(self, tmp_path: Path):
        from scripts.locked_gate_utils import (
            load_locked_gate_rules,
            merge_locked_gate_rules,
        )

        target = tmp_path / "target.yaml"
        target.write_text("hard_gates: []\nguardrails: []\n", encoding="utf-8")

        locked = [
            {
                "id": "gate_x",
                "locked": True,
                "lock_reason": "test",
                "when": {"x": {"value_gt": 0.5}},
                "then": {"action": "deny"},
            },
        ]
        result = merge_locked_gate_rules(target, locked)
        assert result["added"] == 1
        assert result["total"] == 1

        raw = yaml.safe_load(target.read_text())
        assert len(raw["hard_gates"]) == 1
        assert raw["hard_gates"][0]["id"] == "gate_x"

    def test_no_duplicate(self, gate_yaml: Path):
        from scripts.locked_gate_utils import (
            load_locked_gate_rules,
            merge_locked_gate_rules,
        )

        locked = load_locked_gate_rules(gate_yaml)
        result = merge_locked_gate_rules(gate_yaml, locked)
        assert result["added"] == 0

    def test_empty_locked_list(self, gate_yaml: Path):
        from scripts.locked_gate_utils import merge_locked_gate_rules

        result = merge_locked_gate_rules(gate_yaml, [])
        assert result == {"added": 0, "total": 0}


class TestCalibrateLockedGateRule:
    def test_calibrate_finds_threshold(self, sample_df: pd.DataFrame):
        from scripts.locked_gate_utils import calibrate_locked_gate_rule

        rule = {
            "id": "gate_evt_var_99",
            "locked": True,
            "lock_reason": "test",
            "when": {"evt_var_99": {"value_gt": 0.5}},
            "then": {"action": "deny"},
        }
        out = calibrate_locked_gate_rule(rule, sample_df)
        assert out["locked"] is True
        assert "disabled" not in out or out.get("disabled") is not True
        assert "when" in out
        assert "last_calibration_score" in out

    def test_calibrate_disables_on_no_signal(self):
        from scripts.locked_gate_utils import calibrate_locked_gate_rule

        rng = np.random.default_rng(7)
        n = 500
        df = pd.DataFrame(
            {
                "flat_feature": np.full(n, 0.5),
                "is_bad": rng.choice([0, 1], n),
            }
        )
        rule = {
            "id": "gate_flat",
            "locked": True,
            "lock_reason": "test",
            "when": {"flat_feature": {"value_gt": 0.5}},
            "then": {"action": "deny"},
        }
        out = calibrate_locked_gate_rule(rule, df)
        assert out.get("disabled") is True
        assert "disabled_reason" in out

    def test_calibrate_missing_feature(self, sample_df: pd.DataFrame):
        from scripts.locked_gate_utils import calibrate_locked_gate_rule

        rule = {
            "id": "gate_missing",
            "locked": True,
            "lock_reason": "test",
            "when": {"nonexistent_col": {"value_gt": 0.5}},
            "then": {"action": "deny"},
        }
        out = calibrate_locked_gate_rule(rule, sample_df)
        assert out.get("disabled") is True

    def test_calibrate_range_result(self, sample_df: pd.DataFrame):
        from scripts.locked_gate_utils import calibrate_locked_gate_rule

        rule = {
            "id": "gate_evt_var_99",
            "locked": True,
            "lock_reason": "test",
            "when": {
                "all_of": [
                    {"evt_var_99": {"value_gt": 0.6}},
                    {"evt_var_99": {"value_lt": 0.8}},
                ]
            },
            "then": {"action": "deny"},
        }
        out = calibrate_locked_gate_rule(rule, sample_df)
        assert out["locked"] is True
        when = out["when"]
        direction = out.get("comment", "")
        assert "direction=" in direction


class TestCalibrateAllLockedGates:
    def test_batch_calibrate(self, gate_yaml: Path, sample_df: pd.DataFrame):
        from scripts.locked_gate_utils import calibrate_all_locked_gates

        results = calibrate_all_locked_gates(gate_yaml, sample_df, write_back=True)
        assert len(results) == 2

        raw = yaml.safe_load(gate_yaml.read_text())
        all_rules = (raw.get("hard_gates") or []) + (raw.get("system_safety") or [])
        locked_rules = [r for r in all_rules if r.get("locked")]
        assert len(locked_rules) == 2

        non_locked = [r for r in all_rules if not r.get("locked")]
        assert len(non_locked) == 1
        assert non_locked[0]["id"] == "gate_unlocked"


class TestLoaderDisabledSkip:
    def test_disabled_rule_skipped(self, tmp_path: Path):
        """disabled: true rules should not trigger deny."""
        content = textwrap.dedent(
            """\
            schema:
              phases: [hard_gate]
              governance: {}
            hard_gates:
              - id: gate_disabled
                tag: DISABLED
                phase: hard_gate
                priority: 10
                reason: disabled test
                when:
                  feat_a:
                    value_gt: 0.0
                then:
                  action: deny
                locked: true
                disabled: true
            system_safety: []
            guardrails: []
        """
        )
        p = tmp_path / "gate.yaml"
        p.write_text(content, encoding="utf-8")

        from src.time_series_model.archetype.loader import (
            GateConfig,
            StrategyArchetype,
            EvidenceConfig,
            ExecutionConfig,
            PrefilterConfig,
        )

        gate = GateConfig.from_yaml(p)
        assert len(gate.all_rules) == 1
        assert gate.all_rules[0].disabled is True

        arch = StrategyArchetype(
            name="test",
            gate=gate,
            evidence=EvidenceConfig(),
            execution=ExecutionConfig(),
            prefilter=PrefilterConfig(),
        )
        passed, reasons, _ = arch.apply_gate({"feat_a": 0.5})
        assert passed is True
        assert len(reasons) == 0

    def test_enabled_rule_triggers(self, tmp_path: Path):
        """Non-disabled locked rules should trigger normally."""
        content = textwrap.dedent(
            """\
            schema:
              phases: [hard_gate]
              governance: {}
            hard_gates:
              - id: gate_active
                tag: ACTIVE
                phase: hard_gate
                priority: 10
                reason: active test
                when:
                  feat_a:
                    value_gt: 0.0
                then:
                  action: deny
                locked: true
            system_safety: []
            guardrails: []
        """
        )
        p = tmp_path / "gate.yaml"
        p.write_text(content, encoding="utf-8")

        from src.time_series_model.archetype.loader import (
            GateConfig,
            StrategyArchetype,
            EvidenceConfig,
            ExecutionConfig,
            PrefilterConfig,
        )

        gate = GateConfig.from_yaml(p)
        arch = StrategyArchetype(
            name="test",
            gate=gate,
            evidence=EvidenceConfig(),
            execution=ExecutionConfig(),
            prefilter=PrefilterConfig(),
        )
        passed, reasons, _ = arch.apply_gate({"feat_a": 0.5})
        assert passed is False
        assert "ACTIVE" in reasons
