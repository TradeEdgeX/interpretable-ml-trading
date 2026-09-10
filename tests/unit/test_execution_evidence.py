import pytest

from src.time_series_model.core.constitution.execution_evidence import (
    compute_execution_evidence,
)


@pytest.mark.unit
def test_compute_execution_evidence_any_key_contains():
    feats = {"vpin_x": 1.0, "sqs_hal_high": 0.2, "close": 1.0}
    rules = [
        {
            "name": "has_orderflow",
            "kind": "any_key_contains",
            "any_key_contains": ["vpin", "footprint"],
        },
        {
            "name": "has_sr_quality",
            "kind": "any_key_contains",
            "any_key_contains": ["sqs_", "sr_"],
        },
        {"name": "has_wick", "kind": "any_key_contains", "any_key_contains": ["wick"]},
    ]
    ev = compute_execution_evidence(features=feats, rules=rules)
    assert ev["has_orderflow"] is True
    assert ev["has_sr_quality"] is True
    assert ev["has_wick"] is False


@pytest.mark.unit
def test_compute_execution_evidence_value_level_rules():
    feats = {"ttm": 5.0, "absorption": -0.2}
    rules = [
        {"name": "ttm_break", "kind": "value_gt", "key": "ttm", "threshold": 3.0},
        {
            "name": "absorption_strong",
            "kind": "abs_gt",
            "key": "absorption",
            "threshold": 0.1,
        },
        {
            "name": "missing_defaults_false",
            "kind": "value_gt",
            "key": "nope",
            "threshold": 1.0,
        },
    ]
    ev = compute_execution_evidence(features=feats, rules=rules)
    assert ev["ttm_break"] is True
    assert ev["absorption_strong"] is True
    assert ev["missing_defaults_false"] is False
