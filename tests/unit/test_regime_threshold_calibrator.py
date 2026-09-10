"""Unit tests for Tier-0 regime threshold calibrator core."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.time_series_model.regime.threshold_calibrator import (
    StrategyCalibration,
    build_updated_regime,
    calibrate_strategies,
    find_rule,
    get_current_value,
    get_last_plateau,
    scan_chop_plateau,
)


def _make_synthetic_chop_parquet(tmp_path: Path) -> Path:
    """合成数据：低 chop → 高成功率；高 chop → 低成功率。"""
    rng = np.random.default_rng(42)
    n = 1000
    chop = rng.uniform(0.0, 1.0, n)
    # success_no_rr_extreme = 1 if chop < 0.4 (mostly), else 0
    base_p_good = np.where(chop < 0.4, 0.7, 0.2)
    success = (rng.uniform(0, 1, n) < base_p_good).astype(int)
    df = pd.DataFrame({"tpc_semantic_chop": chop, "success_no_rr_extreme": success})
    pq = tmp_path / "features_labeled.parquet"
    df.to_parquet(pq, index=False)
    return pq


def _write_regime_yaml(
    path: Path, value: float, last_plateau: dict | None = None
) -> None:
    payload = {
        "allowed_regimes": ["bull", "bear", "neutral"],
        "allowed_sides": ["long", "short"],
        "rules": [
            {
                "feature": "tpc_semantic_chop",
                "operator": "<=",
                "value": float(value),
                "locked": True,
            }
        ],
    }
    if last_plateau is not None:
        payload["last_calibration"] = {
            "timestamp": "2025-01-01T00:00:00Z",
            "data_source": "previous",
            "plateaus": [
                {
                    "feature": "tpc_semantic_chop",
                    "operator": "<=",
                    "plateau": last_plateau,
                    "action": "ADOPT",
                }
            ],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_scan_chop_plateau_returns_valid_range(tmp_path: Path):
    pq = _make_synthetic_chop_parquet(tmp_path)
    df = pd.read_parquet(pq)
    # 用更宽的 band_fraction 容纳合成数据的噪声
    res = scan_chop_plateau(df, scan_points=25, plateau_band_fraction=0.20)
    assert res is not None
    # plateau 是合法区间 + 落在数据范围内
    assert 0.0 <= res.plateau.start <= res.plateau.end <= 1.0
    # 至少有一些 score 数据被记录
    assert len(res.scores) >= 5
    # 合成数据 low chop → high lift；plateau 中点应该偏向 low side
    # （band=0.20 较宽，允许把 0.4 邻域纳入）
    assert res.plateau.mid <= 0.55


def test_find_rule_and_current_value():
    raw = {
        "rules": [
            {"feature": "tpc_semantic_chop", "operator": "<=", "value": 0.40},
            {"feature": "box_pos_120", "operator": "<=", "value": 0.15},
        ]
    }
    rule = find_rule(raw, feature="tpc_semantic_chop", operator_str="<=")
    assert rule is not None and rule["value"] == 0.40
    assert (
        get_current_value(raw, feature="tpc_semantic_chop", operator_str="<=") == 0.40
    )
    assert get_current_value(raw, feature="nonexistent", operator_str=">=") is None


def test_get_last_plateau_roundtrip():
    raw = {
        "last_calibration": {
            "plateaus": [
                {
                    "feature": "tpc_semantic_chop",
                    "operator": "<=",
                    "plateau": {"start": 0.30, "end": 0.45, "mid": 0.375},
                }
            ]
        }
    }
    p = get_last_plateau(raw, feature="tpc_semantic_chop", operator_str="<=")
    assert p is not None
    assert p.mid == pytest.approx(0.375)


def test_build_updated_regime_adopt_writes_value_and_plateau():
    raw = {"rules": [{"feature": "tpc_semantic_chop", "operator": "<=", "value": 0.40}]}
    from scripts.plateau_stability import PlateauRange

    new_plateau = PlateauRange(0.35, 0.50, 0.425)
    updated = build_updated_regime(
        raw,
        feature="tpc_semantic_chop",
        operator_str="<=",
        chosen_value=0.425,
        new_plateau=new_plateau,
        timestamp_iso="2026-05-21T00:00:00Z",
        data_source="features_labeled.parquet",
        decision_reason="overlap",
        action="ADOPT",
    )
    assert updated["rules"][0]["value"] == 0.425
    cal = updated["last_calibration"]
    assert cal["timestamp"] == "2026-05-21T00:00:00Z"
    assert cal["plateaus"][0]["plateau"]["mid"] == 0.425
    assert cal["plateaus"][0]["action"] == "ADOPT"


def test_build_updated_regime_alert_keeps_value():
    raw = {"rules": [{"feature": "tpc_semantic_chop", "operator": "<=", "value": 0.40}]}
    from scripts.plateau_stability import PlateauRange

    new_plateau = PlateauRange(0.60, 0.80, 0.70)
    updated = build_updated_regime(
        raw,
        feature="tpc_semantic_chop",
        operator_str="<=",
        chosen_value=0.40,  # keep current
        new_plateau=new_plateau,
        timestamp_iso="2026-05-21T00:00:00Z",
        data_source="features_labeled.parquet",
        decision_reason="plateau_drift_detected",
        action="ALERT",
    )
    # value 不变
    assert updated["rules"][0]["value"] == 0.40
    # plateau 仍写进 last_calibration（供下一轮比较）
    assert updated["last_calibration"]["plateaus"][0]["action"] == "ALERT"


def test_calibrate_strategies_end_to_end(tmp_path: Path):
    pq = _make_synthetic_chop_parquet(tmp_path)
    yaml_a = tmp_path / "a" / "archetypes" / "regime.yaml"
    yaml_b = tmp_path / "b" / "archetypes" / "regime.yaml"
    _write_regime_yaml(
        yaml_a, value=0.40, last_plateau={"start": 0.30, "end": 0.50, "mid": 0.40}
    )
    _write_regime_yaml(yaml_b, value=0.40, last_plateau=None)  # 首轮

    items = [
        StrategyCalibration(strategy="a", regime_yaml_path=yaml_a, parquet_path=pq),
        StrategyCalibration(strategy="b", regime_yaml_path=yaml_b, parquet_path=pq),
    ]
    out = calibrate_strategies(
        items,
        feature="tpc_semantic_chop",
        operator_str="<=",
        label_col="success_no_rr_extreme",
        scan_points=20,
        policy="keep_if_no_overlap",
        timestamp_iso=datetime.now(timezone.utc).isoformat(),
    )
    assert len(out) == 2
    for it in out:
        assert it.skipped_reason is None
        assert it.new_plateau is not None
        assert it.decision is not None
        assert it.updated_regime is not None
    # strategy b 是首轮 → ADOPT
    assert out[1].decision["action"] == "ADOPT"


def test_calibrate_strategies_skips_missing_files(tmp_path: Path):
    items = [
        StrategyCalibration(
            strategy="x",
            regime_yaml_path=tmp_path / "nonexistent.yaml",
            parquet_path=tmp_path / "nonexistent.parquet",
        )
    ]
    out = calibrate_strategies(
        items,
        timestamp_iso=datetime.now(timezone.utc).isoformat(),
    )
    assert out[0].skipped_reason is not None
    assert "regime.yaml not found" in out[0].skipped_reason
