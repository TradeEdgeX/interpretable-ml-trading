"""Unit tests for regime-aware monitor helpers."""

from __future__ import annotations

import pandas as pd

from src.monitoring.regime_health import (
    evaluate_multileg_entry_health,
    evaluate_regime_share_drift,
    has_labeled_regime_schema,
    has_multileg_regime_schema,
    regime_shares_from_window,
)

TPC_LABELED_REGIME = {
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
                {"feature": "adx_50", "operator": "<=", "value": 20},
                {"feature": "ema_1200_position", "operator": "<=", "value": -0.1},
            ],
        },
        "neutral": {
            "match": "any",
            "rules": [{"feature": "adx_50", "operator": "<=", "value": 25}],
        },
    }
}


def test_has_labeled_regime_schema_true_for_tpc_style():
    assert has_labeled_regime_schema(TPC_LABELED_REGIME) is True
    assert has_labeled_regime_schema({"allowed_regimes": ["bull", "bear"]}) is False


def test_regime_shares_classify_rows():
    df = pd.DataFrame(
        {
            "adx_50": [30.0, 15.0, 10.0],
            "ema_1200_position": [0.2, -0.2, 0.0],
        }
    )
    shares = regime_shares_from_window(df, TPC_LABELED_REGIME)
    assert shares["bull"] == 1 / 3
    assert shares["bear"] == 2 / 3
    assert shares["neutral"] == 0.0


def test_regime_share_drift_alerts_on_delta():
    df = pd.DataFrame(
        {
            "adx_50": [30.0] * 100,
            "ema_1200_position": [0.2] * 100,
        }
    )
    r = evaluate_regime_share_drift(
        strategy="tpc",
        regime_yaml=TPC_LABELED_REGIME,
        window_df=df,
        baseline_entry={"regime_shares": {"bull": 0.0, "bear": 0.5, "neutral": 0.5}},
        share_tol=0.10,
    )
    assert r["status"] == "ALERT"
    assert any("REGIME_SHARE_DRIFT" in a for a in r.get("alerts") or [])


def test_regime_share_drift_baseline_missing_not_alert():
    df = pd.DataFrame({"adx_50": [10.0], "ema_1200_position": [0.0]})
    r = evaluate_regime_share_drift(
        strategy="tpc",
        regime_yaml=TPC_LABELED_REGIME,
        window_df=df,
        baseline_entry=None,
    )
    assert r["any_alert"] is False
    assert r["status"] == "BASELINE_MISSING"


CHOP_GRID_REGIME = {
    "extensions": {
        "multileg": {
            "entry_feature": "bpc_semantic_chop",
            "entry_min": 0.52,
        }
    },
    "last_calibration": {
        "multileg_baseline": {
            "chop_grid": {"entry_pass_rate": 0.50, "median_entry_feature": 0.55}
        }
    },
}


def test_has_multileg_regime_schema():
    assert has_multileg_regime_schema(CHOP_GRID_REGIME) is True
    assert has_multileg_regime_schema({"extensions": {}}) is False


def test_multileg_pass_rate_drift_alerts():
    df = pd.DataFrame({"bpc_semantic_chop": [0.8] * 50})
    r = evaluate_multileg_entry_health(
        strategy="chop_grid",
        regime_yaml=CHOP_GRID_REGIME,
        window_df=df,
        pass_rate_tol=0.10,
    )
    assert r["status"] == "ALERT"
    assert any("MULTILEG_PASS_RATE_DRIFT" in a for a in r.get("alerts") or [])


def test_multileg_baseline_missing_not_alert():
    df = pd.DataFrame({"bpc_semantic_chop": [0.6] * 50})
    regime = {
        "extensions": {
            "multileg": {"entry_feature": "bpc_semantic_chop", "entry_min": 0.52}
        }
    }
    r = evaluate_multileg_entry_health(
        strategy="chop_grid",
        regime_yaml=regime,
        window_df=df,
    )
    assert r["any_alert"] is False
    assert r["status"] == "BASELINE_MISSING"
