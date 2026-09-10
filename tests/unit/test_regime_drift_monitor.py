"""Unit tests for regime drift monitor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.regime_drift_monitor import evaluate_strategy_drift


def _regime_cfg_with_plateau(feature: str, start: float, end: float) -> dict:
    return {
        "rules": [{"feature": feature, "operator": "<=", "value": (start + end) / 2}],
        "last_calibration": {
            "plateaus": [
                {
                    "feature": feature,
                    "operator": "<=",
                    "plateau": {"start": start, "end": end, "mid": (start + end) / 2},
                }
            ]
        },
    }


def test_ok_when_window_median_in_plateau():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"tpc_semantic_chop": rng.uniform(0.30, 0.45, 200)})
    cfg = _regime_cfg_with_plateau("tpc_semantic_chop", 0.30, 0.45)
    r = evaluate_strategy_drift(
        strategy="bpc", regime_yaml=cfg, window_df=df, drift_quantile=0.5
    )
    assert r["any_alert"] is False
    assert r["items"][0]["status"] == "OK"


def test_alert_when_window_median_outside_plateau():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"tpc_semantic_chop": rng.uniform(0.60, 0.80, 200)})
    cfg = _regime_cfg_with_plateau("tpc_semantic_chop", 0.30, 0.45)
    r = evaluate_strategy_drift(
        strategy="bpc", regime_yaml=cfg, window_df=df, drift_quantile=0.5
    )
    assert r["any_alert"] is True
    assert r["items"][0]["status"] == "DRIFT"


def test_alert_when_feature_missing():
    df = pd.DataFrame({"other_col": [1, 2, 3, 4, 5]})
    cfg = _regime_cfg_with_plateau("tpc_semantic_chop", 0.30, 0.45)
    r = evaluate_strategy_drift(
        strategy="bpc", regime_yaml=cfg, window_df=df, drift_quantile=0.5
    )
    assert r["any_alert"] is True
    assert r["items"][0]["status"] == "MISSING_FEATURE"


def test_alert_when_too_few_rows():
    df = pd.DataFrame({"tpc_semantic_chop": [0.4]})
    cfg = _regime_cfg_with_plateau("tpc_semantic_chop", 0.30, 0.45)
    r = evaluate_strategy_drift(
        strategy="bpc", regime_yaml=cfg, window_df=df, drift_quantile=0.5
    )
    assert r["any_alert"] is True
    assert r["items"][0]["status"] == "INSUFFICIENT_DATA"


def test_no_plateaus_reports_uncalibrated_not_ok():
    df = pd.DataFrame({"tpc_semantic_chop": [0.4] * 50})
    cfg = {"rules": []}  # 没 last_calibration → no plateaus
    r = evaluate_strategy_drift(
        strategy="bpc", regime_yaml=cfg, window_df=df, drift_quantile=0.5
    )
    assert r["any_alert"] is False
    assert r["status"] == "NO_PLATEAUS"
    assert r["items"] == []
    assert "plateaus" in (r.get("skipped") or "")


def test_labeled_regime_uses_share_drift_not_no_plateaus():
    cfg = {
        "allowed_regimes": {
            "bull": {
                "match": "all",
                "rules": [
                    {"feature": "adx_50", "operator": ">=", "value": 25},
                    {"feature": "ema_1200_position", "operator": ">=", "value": 0.1},
                ],
            },
            "bear": {
                "match": "any",
                "rules": [
                    {"feature": "ema_1200_position", "operator": "<=", "value": -0.1}
                ],
            },
        }
    }
    df = pd.DataFrame({"adx_50": [30.0] * 20, "ema_1200_position": [0.2] * 20})
    r = evaluate_strategy_drift(strategy="tpc", regime_yaml=cfg, window_df=df)
    assert r["status"] == "BASELINE_MISSING"
    assert r["items"][0]["kind"] == "regime_shares"
