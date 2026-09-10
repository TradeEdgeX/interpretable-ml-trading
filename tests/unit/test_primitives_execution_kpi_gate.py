from pathlib import Path

import yaml

from src.time_series_model.diagnostics.kpi_gate import check_kpi_gate


def test_primitives_execution_kpi_gate_passes_minimums() -> None:
    gate_path = Path("config/kpi_gates/nnmh_primitives_model.yaml")
    gate = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
    metrics = {
        "router_diag__trade_n": 100,
        "router_diag__trade_rate": 0.05,
        "router_diag__trade_win_rate": 0.55,
        "router_diag__trade_avg_ret": 0.001,
    }
    res = check_kpi_gate(metrics=metrics, gate=gate)
    assert res.ok is True
    assert res.hard_failures == []


def test_primitives_execution_kpi_gate_fails_on_missing_core_metrics() -> None:
    gate_path = Path("config/kpi_gates/nnmh_primitives_model.yaml")
    gate = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
    metrics = {
        "router_diag__trade_rate": 0.05,
    }
    res = check_kpi_gate(metrics=metrics, gate=gate)
    assert res.ok is False
