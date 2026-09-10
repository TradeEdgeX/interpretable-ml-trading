"""Parity: snotio_calc kernel wired into entry plateau script."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.research.entry_plateau_scan import generate_scan_range
from src.research.stat_kernels.snotio_calc import (
    compute_snotio,
    find_snotio_plateau,
    width_to_confidence,
)


def _synthetic_scan(n: int = 12) -> list[dict]:
    rows = []
    for i in range(n):
        th = round(0.1 * i, 2)
        rows.append(
            {
                "threshold": th,
                "snotio": 0.15 + 0.01 * (i % 5),
                "trades": 80 + i * 3,
                "too_few": i < 1,
            }
        )
    return rows


def test_entry_plateau_scan_uses_shared_kernel():
    grid = generate_scan_range(0.5, ">=", n_steps=5)
    assert len(grid) == 5
    assert width_to_confidence(0.35) == "HIGH"


def test_width_to_confidence_tiers():
    assert width_to_confidence(0.35) == "HIGH"
    assert width_to_confidence(0.2) == "MEDIUM"
    assert width_to_confidence(0.05) == "LOW"


def test_compute_snotio_mean():
    assert compute_snotio(pd.Series([1.0, 2.0, 3.0])) == pytest.approx(2.0)
    assert compute_snotio(pd.Series([])) == 0.0


def test_find_snotio_plateau_stable_window():
    results = _synthetic_scan()
    for operator in (">=", "<="):
        out = find_snotio_plateau(results, operator=operator, window=5)
        if out.get("is_plateau"):
            assert "recommended" in out
            assert out["confidence"] in ("HIGH", "MEDIUM", "LOW")


def test_find_snotio_plateau_fallback_best_single():
    results = [
        {"threshold": 0.1, "snotio": 0.05, "trades": 10, "too_few": False},
        {"threshold": 0.2, "snotio": 0.25, "trades": 12, "too_few": False},
        {"threshold": 0.3, "snotio": 0.04, "trades": 11, "too_few": False},
        {"threshold": 0.4, "snotio": 0.22, "trades": 13, "too_few": False},
        {"threshold": 0.5, "snotio": 0.03, "trades": 14, "too_few": False},
    ]
    out = find_snotio_plateau(results, window=5)
    assert out["is_plateau"] is False
    assert "best_single" in out
    assert out["best_single"]["threshold"] == 0.2


def test_scan_snotio_thresholds_and_payload():
    from src.research.stat_kernels.snotio_calc import (
        scan_snotio_thresholds,
        snotio_plateau_payload,
    )

    rng = np.random.default_rng(1)
    n = 300
    feat = rng.normal(size=n)
    forward_rr = np.where(feat > 0, 0.3, -0.1) + rng.normal(0, 0.05, size=n)
    df = pd.DataFrame({"pulse_z": feat, "forward_rr": forward_rr})
    mask = pd.Series(True, index=df.index)
    grid = [float(x) for x in "0,0.5,1".split(",")]
    rows = scan_snotio_thresholds(df, "pulse_z", "<=", grid, mask, min_trades=10)
    assert len(rows) == 3
    assert any(r["snotio"] > 0 for r in rows if not r["too_few"])
    payload = snotio_plateau_payload(
        df, "pulse_z", "<=", grid, mask, min_trades=10, window=2
    )
    assert payload["kpi"] == "snotio"
    assert "rows" in payload
