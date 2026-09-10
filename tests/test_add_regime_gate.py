"""Unit tests for add-on ``add_regime_gate`` (chop / ML score AND rules)."""

from __future__ import annotations

from src.time_series_model.core.constitution.add_position_rules import (
    add_regime_gate_allows,
)


def test_gate_disabled_passes():
    ok, why = add_regime_gate_allows({"bpc_semantic_chop_ts_q": 0.99}, {})
    assert ok and why == ""
    ok, why = add_regime_gate_allows(
        {"bpc_semantic_chop_ts_q": 0.99},
        {
            "add_regime_gate": {
                "enabled": False,
                "allow_if_all": [{"feature": "x", "lte": 0}],
            }
        },
    )
    assert ok


def test_chop_lte_rule_blocks():
    cfg = {
        "add_regime_gate": {
            "enabled": True,
            "allow_if_all": [{"feature": "bpc_semantic_chop_ts_q", "lte": 0.5}],
        }
    }
    ok, why = add_regime_gate_allows({"bpc_semantic_chop_ts_q": 0.4}, cfg)
    assert ok
    ok, why = add_regime_gate_allows({"bpc_semantic_chop_ts_q": 0.51}, cfg)
    assert not ok and ">" in why


def test_missing_feature_skips_rule():
    cfg = {
        "add_regime_gate": {
            "enabled": True,
            "allow_if_all": [{"feature": "bpc_semantic_chop_ts_q", "lte": 0.5}],
        }
    }
    ok, why = add_regime_gate_allows({}, cfg)
    assert ok and why == ""


def test_and_rules_ml_score():
    cfg = {
        "add_regime_gate": {
            "enabled": True,
            "allow_if_all": [
                {"feature": "bpc_semantic_chop_ts_q", "lte": 0.6},
                {"feature": "add_ml_score", "gte": 0.5},
            ],
        }
    }
    ok, _ = add_regime_gate_allows(
        {"bpc_semantic_chop_ts_q": 0.3, "add_ml_score": 0.6}, cfg
    )
    assert ok
    ok, why = add_regime_gate_allows(
        {"bpc_semantic_chop_ts_q": 0.3, "add_ml_score": 0.4}, cfg
    )
    assert not ok and "add_ml_score" in why


def test_align_with_side_rule():
    cfg = {
        "add_regime_gate": {
            "enabled": True,
            "allow_if_all": [
                {"feature": "macd_atr", "align_with_side": True, "gte": 0.2}
            ],
        }
    }
    ok, _ = add_regime_gate_allows({"macd_atr": 0.3, "position_action": "LONG"}, cfg)
    assert ok
    ok, _ = add_regime_gate_allows({"macd_atr": -0.4, "position_action": "SHORT"}, cfg)
    assert ok
    ok, why = add_regime_gate_allows({"macd_atr": -0.1, "position_action": "LONG"}, cfg)
    assert not ok and "macd_atr" in why
