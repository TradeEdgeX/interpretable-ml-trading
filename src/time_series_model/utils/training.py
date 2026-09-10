"""Training utilities shared across dimensionality workflows."""

from __future__ import annotations

import os
from typing import Any, Dict

import lightgbm as lgb
from lightgbm.basic import LightGBMError
import numpy as np
import pandas as pd


def train_lightgbm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_gpu: bool = True,
    num_boost_round: int = 200,
    params: Dict[str, Any] | None = None,
    *,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    early_stopping_rounds: int | None = None,
    eval_period: int | None = 50,
    categorical_feature: Any | None = None,
    feature_names: list | None = None,
    preserve_multiclass: bool = False,
    task_type: str | None = None,
) -> lgb.Booster:
    """Train a LightGBM model with optional validation support.

    Args:
        preserve_multiclass: If True, preserve 3-class labels (0=Hold, 1=Long, 2=Short) instead of converting to binary.
                             If False (default), filters Hold and converts to binary (1=Long, 0=Short).
        task_type: Explicit task type: "multiclass", "binary", or "regression".
                   If None, auto-detects from labels.

    Task types:
    - "multiclass": 3-class classification (0=Hold, 1=Long, 2=Short)
    - "binary": Binary classification (0/1)
    - "regression": Continuous prediction (returns, volatility, etc.)
    """

    # Determine task type
    if task_type is None:
        # Auto-detect task type based on labels
        unique_labels = np.unique(y_train)
        num_unique = len(unique_labels)

        # Check if labels are 3-class format (0=Hold, 1=Long, 2=Short)
        is_3class = (
            num_unique <= 3
            and np.all(np.equal(np.mod(unique_labels, 1), 0))
            and np.all(unique_labels >= 0)
            and np.all(unique_labels <= 2)
        )

        if is_3class and preserve_multiclass:
            # Use true multiclass (3-class: Hold, Long, Short)
            task_type = "multiclass"
            num_classes = len(unique_labels)
            objective = "multiclass"
            metric = "multi_logloss"
            task_params = {"num_class": num_classes}
        elif is_3class and not preserve_multiclass:
            # Filter neutral labels (0=Hold) and convert to binary (1=Long, 0=Short)
            train_mask = (y_train == 1) | (y_train == 2)
            if train_mask.sum() == 0:
                raise ValueError(
                    "No valid long/short samples in training set after removing neutral labels."
                )
            X_train = X_train[train_mask]
            y_train = np.where(y_train[train_mask] == 1, 1, 0).astype(int)

            # Also filter validation set if provided
            if X_val is not None and y_val is not None:
                val_mask = (y_val == 1) | (y_val == 2)
                if val_mask.sum() == 0:
                    raise ValueError(
                        "No valid long/short samples in validation set after removing neutral labels."
                    )
                X_val = X_val[val_mask]
                y_val = np.where(y_val[val_mask] == 1, 1, 0).astype(int)

            task_type = "binary"
            objective = "binary"
            metric = ["auc", "binary_logloss"]
            task_params = {}
        elif num_unique == 2:
            task_type = "binary"
            objective = "binary"
            metric = ["auc", "binary_logloss"]
            task_params = {}
        else:
            task_type = "regression"
            objective = "regression"
            metric = "rmse"
            task_params = {}
    else:
        # Use explicit task_type
        if task_type == "multiclass":
            unique_labels = np.unique(y_train)
            num_classes = len(unique_labels)
            objective = "multiclass"
            metric = "multi_logloss"
            task_params = {"num_class": num_classes}
        elif task_type == "binary":
            objective = "binary"
            metric = ["auc", "binary_logloss"]
            task_params = {}
        elif task_type == "regression":
            objective = "regression"
            metric = "rmse"
            task_params = {}
        else:
            raise ValueError(
                f"Unknown task_type: {task_type}. Must be 'multiclass', 'binary', or 'regression'"
            )

    default_params = {
        "objective": objective,
        "metric": metric,
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "force_col_wise": True,
        # Multi-thread by default; MLBOT_DETERMINISTIC=1 forces single-thread.
        "deterministic": True,
        "num_threads": 1 if os.getenv("MLBOT_DETERMINISTIC", "0") == "1" else -1,
        "seed": 42,
        "feature_fraction_seed": 42,
        "bagging_seed": 42,
        "data_random_seed": 42,
        **task_params,  # Add num_class for multiclass
    }

    # Handle n_estimators -> num_boost_round conversion
    if params:
        # Extract n_estimators if present and convert to num_boost_round
        if "n_estimators" in params:
            num_boost_round = params.pop("n_estimators")
        default_params.update(params)

    # Remove n_estimators from default_params if it exists (LightGBM uses num_boost_round instead)
    default_params.pop("n_estimators", None)

    if use_gpu:
        default_params.update(
            {
                "device": "cuda",
                "gpu_platform_id": 0,
                "gpu_device_id": 0,
            }
        )

    # Prepare categorical feature specification
    cat_feature_indices = None
    if categorical_feature is not None:
        if isinstance(categorical_feature, (list, tuple)):
            if feature_names:
                # Map feature names to indices
                cat_feature_indices = [
                    i
                    for i, name in enumerate(feature_names)
                    if name in categorical_feature
                ]
            else:
                # If no feature names, assume categorical_feature is already indices
                cat_feature_indices = list(categorical_feature)
        elif isinstance(categorical_feature, (int, str)):
            # Single feature
            if feature_names:
                if categorical_feature in feature_names:
                    cat_feature_indices = [feature_names.index(categorical_feature)]
            else:
                cat_feature_indices = [categorical_feature]

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=feature_names if feature_names else None,
        categorical_feature=cat_feature_indices if cat_feature_indices else None,
        free_raw_data=False,
    )

    valid_sets = [train_data]
    valid_names = ["train"]

    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(
            X_val,
            label=y_val,
            reference=train_data,
            feature_name=feature_names if feature_names else None,
            categorical_feature=cat_feature_indices if cat_feature_indices else None,
            free_raw_data=False,
        )
        valid_sets.append(val_data)
        valid_names.append("valid")

    callbacks = []
    if eval_period is not None:
        callbacks.append(lgb.log_evaluation(period=eval_period))

    if early_stopping_rounds is not None and len(valid_sets) > 1:
        callbacks.append(lgb.early_stopping(early_stopping_rounds))

    try:
        model = lgb.train(
            default_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
    except LightGBMError as exc:
        if use_gpu and "CUDA" in str(exc).upper():
            print("⚠️  Falling back to CPU-based LightGBM training")
            return train_lightgbm_model(
                X_train,
                y_train,
                use_gpu=False,
                num_boost_round=num_boost_round,
                params=params,
                X_val=X_val,
                y_val=y_val,
                early_stopping_rounds=early_stopping_rounds,
                eval_period=eval_period,
                categorical_feature=categorical_feature,
                feature_names=feature_names,
            )
        raise

    return model


__all__ = [
    "train_lightgbm_model",
]
