from src.time_series_model.live.tree_gate import apply_gate_rules


def test_tree_gate_deny():
    rules = {
        "rules": [
            {"name": "deny_big", "kind": "value_gt", "key": "x", "threshold": 10}
        ],
        "deny_if": ["deny_big"],
        "allow_if": [],
        "allow_mode": "any",
        "default_action": "allow",
    }
    ok, reasons = apply_gate_rules(gate_rules=rules, features={"x": 11})
    assert ok is False
    assert "gate_deny" in reasons[0]


def test_tree_gate_allow_any():
    rules = {
        "rules": [
            {"name": "allow_a", "kind": "value_gt", "key": "x", "threshold": 1},
            {"name": "allow_b", "kind": "value_gt", "key": "y", "threshold": 1},
        ],
        "deny_if": [],
        "allow_if": ["allow_a", "allow_b"],
        "allow_mode": "any",
        "default_action": "deny",
    }
    ok, _ = apply_gate_rules(gate_rules=rules, features={"x": 2, "y": 0})
    assert ok is True


def test_tree_gate_allow_min2():
    rules = {
        "rules": [
            {"name": "allow_a", "kind": "value_gt", "key": "x", "threshold": 1},
            {"name": "allow_b", "kind": "value_gt", "key": "y", "threshold": 1},
            {"name": "allow_c", "kind": "value_gt", "key": "z", "threshold": 1},
        ],
        "deny_if": [],
        "allow_if": ["allow_a", "allow_b", "allow_c"],
        "allow_mode": "min2",
        "default_action": "deny",
    }
    ok, _ = apply_gate_rules(gate_rules=rules, features={"x": 2, "y": 0, "z": 2})
    assert ok is True


import numpy as np
import pytest

from src.time_series_model.gating.tree_gate import (
    TreeGateTrainConfig,
    evaluate_gate_effect,
    export_tree_gate_artifact,
    gate_predict,
    train_tree_gate,
)


@pytest.mark.unit
def test_tree_gate_train_export_and_eval_smoke():
    # Synthetic: feature0 high => allow, low => veto
    rng = np.random.default_rng(42)
    n = 2000
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    X = np.stack([x0, x1], axis=1)
    y = (x0 > 0.0).astype(int)

    # Returns: when x0<=0 it's worse tail; gate should veto those.
    ret = rng.normal(loc=0.001, scale=0.02, size=n)
    ret = np.where(x0 <= 0.0, ret - 0.05, ret)

    clf = train_tree_gate(
        X,
        y,
        gate_name="trend_gate",
        feature_names=["f0", "f1"],
        cfg=TreeGateTrainConfig(max_depth=2, min_samples_leaf=50),
    )
    pred = gate_predict(clf, X)
    met = evaluate_gate_effect(allow=pred, ret_used=ret, tail_q=0.05)
    assert 0.0 <= met["activation_rate"] <= 1.0
    # In this synthetic setup, vetoed trades are worse => veto_loss_avoided should be negative - positive? here avg(veto) < avg(allow) => veto_loss_avoided < 0
    assert met["veto_loss_avoided"] < 0.0

    art = export_tree_gate_artifact(
        clf=clf,
        gate_name="trend_gate",
        feature_names=["f0", "f1"],
        metrics=met,
        rules_depth_limit=2,
    )
    assert art.gate_name == "trend_gate"
    assert "type" in art.rules
    assert "activation_rate" in art.metrics
