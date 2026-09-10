"""Common helpers for evaluating feature sets in rolling workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from time_series_model.utils.training import train_lightgbm_model


@dataclass
class FeatureEvaluationResult:
    metrics: Dict[str, float]
    best_iteration: Optional[int] = None


class FeatureEvaluator:
    """Train/evaluate a LightGBM model on a feature set."""

    def __init__(
        self,
        *,
        validation_ratio: float = 0.2,
        min_improvement: float = 0.005,
        task_type: str = "regression",
        lgb_params: Optional[Dict[str, float]] = None,
    ) -> None:
        self.validation_ratio = validation_ratio
        self.min_improvement = min_improvement
        self.task_type = task_type
        self.params = lgb_params or {}

    def evaluate_feature_set(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> FeatureEvaluationResult:
        if len(X) == 0:
            raise ValueError("Empty dataset provided for evaluation")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_ratio, random_state=42
        )

        objective = "regression" if self.task_type == "regression" else "binary"
        metric = "l2" if objective == "regression" else "binary_logloss"

        params = {
            "objective": objective,
            "metric": metric,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "boosting_type": "gbdt",
            "verbose": -1,
        }
        params.update(self.params)

        model = train_lightgbm_model(
            X_train,
            y_train,
            use_gpu=True,
            num_boost_round=100,
            params=params,
            X_val=X_val,
            y_val=y_val,
            early_stopping_rounds=10,
            eval_period=0,
        )

        y_pred = model.predict(X_val)

        if objective == "regression":
            mse = mean_squared_error(y_val, y_pred)
            rmse = float(np.sqrt(mse))
            r2 = r2_score(y_val, y_pred)
            metrics = {"mse": float(mse), "rmse": rmse, "r2": float(r2)}
        else:
            auc = roc_auc_score(y_val, y_pred)
            metrics = {"auc": float(auc)}

        return FeatureEvaluationResult(
            metrics=metrics,
            best_iteration=getattr(model, "best_iteration", None),
        )


__all__ = ["FeatureEvaluator", "FeatureEvaluationResult"]
