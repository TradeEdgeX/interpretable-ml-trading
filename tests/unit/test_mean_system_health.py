import pytest

from src.time_series_model.portfolio.mean_system import compute_mean_system_health


@pytest.mark.unit
def test_mean_system_health_viable():
    m = {
        "mean_only_avg_total_return": 0.1,
        "mean_only_avg_max_dd": 0.2,
        "mean_only_sharpe_mean": 0.5,
    }
    out = compute_mean_system_health(m).as_metrics()
    assert out["mean_system__viable"] == 1.0


@pytest.mark.unit
def test_mean_system_health_not_viable_when_dd_too_high():
    m = {
        "mean_only_avg_total_return": 0.1,
        "mean_only_avg_max_dd": 0.5,
        "mean_only_sharpe_mean": 1.5,
    }
    out = compute_mean_system_health(m).as_metrics()
    assert out["mean_system__viable"] == 0.0
