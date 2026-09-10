import pytest

from src.time_series_model.core.constitution.execution_evidence import (
    compute_execution_evidence,
)


@pytest.mark.unit
def test_execution_evidence_quantile_rules():
    features = {"vpin": 0.12, "cvd_change_5": -200.0}
    quantiles = {
        "vpin": {"0.1": 0.01, "0.9": 0.10},
        "cvd_change_5": {"0.1": -500.0, "0.9": 500.0},
    }
    rules = [
        {"name": "vpin_high", "kind": "quantile_gt", "key": "vpin", "quantile": 0.9},
        {
            "name": "cvd_not_low",
            "kind": "quantile_lt",
            "key": "cvd_change_5",
            "quantile": 0.1,
        },
    ]
    out = compute_execution_evidence(
        features=features, rules=rules, quantiles=quantiles
    )
    assert out["vpin_high"] is True
    assert out["cvd_not_low"] is False
