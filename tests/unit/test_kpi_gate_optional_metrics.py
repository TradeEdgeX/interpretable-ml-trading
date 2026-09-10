import json

import pytest

from src.time_series_model.diagnostics.kpi_gate import check_kpi_gate


@pytest.mark.unit
def test_kpi_gate_optional_metric_missing_is_ok():
    metrics = {"a": 1.0}
    gate = {
        "hard_fail": {"a": {"min": 0.0, "max": 2.0}},
        "warn": {"missing_metric": {"min": 0.0, "max": 1.0, "optional": True}},
    }
    res = check_kpi_gate(metrics=metrics, gate=gate)
    assert res.ok is True
    assert res.hard_failures == []


@pytest.mark.unit
def test_kpi_gate_optional_metric_present_is_checked():
    metrics = {"a": 1.0, "missing_metric": 10.0}
    gate = {
        "hard_fail": {"a": {"min": 0.0, "max": 2.0}},
        "warn": {"missing_metric": {"min": 0.0, "max": 1.0, "optional": True}},
    }
    res = check_kpi_gate(metrics=metrics, gate=gate)
    assert res.ok is True
    assert any("missing_metric" in w for w in res.warnings)
