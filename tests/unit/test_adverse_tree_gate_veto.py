from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from scripts.event_backtest.backtester import _inject_scores_has_usable_columns
from src.time_series_model.live.adverse_tree_gate_veto import AdverseTreeGateVeto


def test_inject_scores_accepts_dual_head_columns() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
            "score_long": [0.6],
            "score_short": [0.4],
        }
    )
    assert _inject_scores_has_usable_columns(df) is True


def test_tree_gate_veto_rejects_high_p_bad() -> None:
    clf = DecisionTreeClassifier(max_depth=1, random_state=0)
    X = np.array([[0.0], [1.0], [0.5], [0.2]])
    y = np.array([1, 0, 0, 1])
    clf.fit(X, y)
    veto = AdverseTreeGateVeto(
        clf=clf, feature_names=["vol_accel"], reject_threshold=0.55
    )
    passed, reasons = veto.evaluate({"vol_accel": 1.0})
    assert passed is False
    assert reasons and "tree_gate_veto" in reasons[0]
