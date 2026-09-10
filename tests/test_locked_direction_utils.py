"""Tests for promote merge of direction_rules (locked from archetypes, unlocked from workspace)."""

from scripts.locked_direction_utils import merge_direction_rules_for_promote


def test_merge_keeps_locked_order_and_replaces_unlocked_tail():
    arch = [
        {
            "method": "dual_position_agree_deadband",
            "features": ["a", "b"],
            "epsilon": 0.1,
            "locked": True,
            "id": "keep_me",
        },
        {"method": "feature_sign", "feature": "old_feat", "transform": "sign"},
    ]
    ws = [
        {"method": "feature_sign", "feature": "new_feat", "transform": "sign"},
        {
            "method": "dual_position_agree_deadband",
            "features": ["x", "y"],
            "epsilon": 0.2,
            "locked": True,
            "id": "ignored_locked_from_ws",
        },
    ]
    merged = merge_direction_rules_for_promote(arch, ws)
    assert len(merged) == 2
    assert merged[0].get("id") == "keep_me"
    assert merged[0].get("locked") is True
    assert merged[1].get("feature") == "new_feat"
    assert all(not r.get("locked") for r in merged[1:])


def test_merge_workspace_empty_list_yields_only_locked():
    arch = [
        {"method": "feature_sign", "feature": "x", "transform": "sign", "locked": True},
    ]
    assert merge_direction_rules_for_promote(arch, []) == arch


def test_merge_no_archetypes_rules_uses_workspace_unlocked_only():
    ws = [{"method": "feature_sign", "feature": "f", "transform": "raw"}]
    assert merge_direction_rules_for_promote([], ws) == ws
