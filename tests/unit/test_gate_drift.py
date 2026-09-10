import pandas as pd
import pytest

from src.time_series_model.gating.gate_drift import (
    compute_gate_drift_series,
    degradation_decision,
)


@pytest.mark.unit
def test_degradation_decision():
    assert degradation_decision(feature_available=True, feature_stable=True) == "normal"
    assert (
        degradation_decision(
            feature_available=False, feature_stable=True, policy="ignore"
        )
        == "ignore"
    )


@pytest.mark.unit
def test_gate_drift_series_smoke():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=120, freq="H", tz="UTC"),
            # allow alternates
            "gate_allow": [1 if i % 2 == 0 else 0 for i in range(120)],
            # vetoed points have worse returns
            "ret_used": [0.01 if i % 2 == 0 else -0.05 for i in range(120)],
        }
    )
    out = compute_gate_drift_series(df, window=60, min_periods=30)
    assert len(out) > 0
    assert "activation_rate" in out.columns
    assert out["activation_rate"].iloc[-1] == pytest.approx(0.5, abs=1e-6)
