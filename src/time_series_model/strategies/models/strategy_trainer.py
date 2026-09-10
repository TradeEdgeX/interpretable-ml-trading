"""
策略模型训练器 - Low-level model training functions

This module provides the core model training functionality. It ONLY handles:
- Cross-validation setup (TimeSeriesSplit)
- Model training (XGBoost/CatBoost/LightGBM)
- CV metric computation
- Returns trained models and metrics

IMPORTANT: This is different from train_strategy.py:
- strategy_trainer.py: Low-level model trainer (THIS FILE) - called by train_strategy.py
- train_strategy.py: Complete pipeline orchestrator - loads data, features, labels, then calls this

This module is a LIBRARY of training functions, not a standalone script.
It is imported and used by train_strategy.py and other training scripts.

Supported models and tasks:
- XGBoost (regression/classification)
- CatBoost (regression/classification)
- LightGBM (regression/classification)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from src.features.utils_feature_column_types import (
    get_categorical_columns,
    load_column_types_config,
)

logger = logging.getLogger(__name__)


def _timestamps_from_df(
    df: pd.DataFrame, date_col: Optional[str] = None
) -> Optional[pd.Series]:
    if date_col and date_col in df.columns:
        return pd.to_datetime(df[date_col], utc=True, errors="coerce")
    for col in ("datetime", "timestamp", "date"):
        if col in df.columns:
            return pd.to_datetime(df[col], utc=True, errors="coerce")
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(pd.DatetimeIndex(df.index), index=df.index)
    return None


def purge_train_indices(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    timestamps: Optional[pd.Series] = None,
    horizon_bars: int = 0,
    bar_minutes: Optional[int] = None,
) -> np.ndarray:
    """Drop train rows whose label window ``[t, t+H)`` overlaps the val fold.

    Timestamp path: drop ``t >= val_start - H * bar_minutes``.
    Fallback: chronological row index, ``train_idx < min(val_idx) - H``.
    """
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    if horizon_bars <= 0 or train_idx.size == 0 or val_idx.size == 0:
        return train_idx
    if timestamps is not None:
        ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
        if not isinstance(ts, pd.Series):
            ts = pd.Series(ts)
        val_start = ts.iloc[val_idx].min()
        if pd.isna(val_start):
            cutoff_i = int(val_idx.min()) - int(horizon_bars)
            return train_idx[train_idx < cutoff_i]
        if bar_minutes and int(bar_minutes) > 0:
            gap = pd.Timedelta(minutes=int(horizon_bars) * int(bar_minutes))
        else:
            unique = pd.Series(ts.dropna().unique()).sort_values()
            diffs = unique.diff().dropna()
            med = diffs.median() if len(diffs) else pd.Timedelta(0)
            gap = (
                med * int(horizon_bars)
                if pd.notna(med) and med > pd.Timedelta(0)
                else pd.Timedelta(0)
            )
        cutoff = val_start - gap if gap > pd.Timedelta(0) else val_start
        train_ts = ts.iloc[train_idx]
        keep = train_ts < cutoff
        return train_idx[np.asarray(keep.fillna(False), dtype=bool)]
    cutoff_i = int(val_idx.min()) - int(horizon_bars)
    return train_idx[train_idx < cutoff_i]


try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import catboost as cb  # type: ignore[import-untyped]

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


@dataclass(frozen=True)
class FeaturePreprocessor:
    """
    Deterministic feature preprocessing for model training/inference.

    - Numeric columns: to_numeric(coerce) + replace inf -> NaN (keep NaN)
    - Categorical columns: string mapping to integer codes (unknown -> __UNKNOWN__)
    """

    feature_cols: List[str]
    numeric_cols: List[str]
    categorical_cols: List[str]
    categorical_mappings: Dict[str, Dict[str, int]]
    # Optional per-feature multipliers (e.g., invert negative factors => -1.0).
    # Applied at transform-time for numeric features to keep inference consistent.
    feature_multipliers: Dict[str, float] = None

    def transform(
        self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None
    ) -> np.ndarray:
        cols = feature_cols if feature_cols is not None else self.feature_cols
        if not cols:
            return np.zeros((len(df), 0), dtype=float)

        out = pd.DataFrame(index=df.index)

        for col in cols:
            if col not in df.columns:
                # Keep missing as NaN to preserve semantics (NaN != 0).
                # Downstream tree models (LightGBM/XGBoost) can handle NaN as missing.
                logger.warning(
                    "Feature column missing at transform time: %s (filled with NaN)",
                    col,
                )
                out[col] = np.nan
                continue

            if col in self.categorical_mappings:
                mapping = self.categorical_mappings[col]
                unknown_code = mapping.get("__UNKNOWN__", 0)
                series = df[col].fillna("__MISSING__").astype(str)
                # Use float dtype to allow NaN if upstream produces it; codes are integer-like.
                out[col] = series.map(mapping).fillna(unknown_code).astype(float)
            else:
                series = pd.to_numeric(df[col], errors="coerce")
                # Convert inf/-inf to NaN and keep NaN (do NOT fill with 0).
                series = series.replace([np.inf, -np.inf], np.nan)
                if self.feature_multipliers and col in self.feature_multipliers:
                    try:
                        series = series * float(self.feature_multipliers[col])
                    except Exception:
                        pass
                out[col] = series.astype(float)

        # Ensure a numeric float matrix (NaN preserved).
        X = out[cols].to_numpy(dtype=float)

        return X


def train_strategy_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "label",
    model_type: Literal["xgboost", "catboost", "lightgbm"] = "xgboost",
    task_type: Literal["regression", "binary", "multiclass"] = "regression",
    n_splits: int = 5,
    tscv_gap: int = 24,
    purge_gap: Optional[int] = None,
    bar_minutes: Optional[int] = None,
    date_col: Optional[str] = None,
    tradable_col: Optional[str] = None,
    weight_col: Optional[str] = None,
    model_params: Optional[Dict] = None,
    use_gpu: bool = True,
) -> Tuple[List, float, pd.DataFrame, List[str], FeaturePreprocessor]:
    """
    训练策略模型（支持 XGBoost、CatBoost、LightGBM）

    Args:
        df: DataFrame with features and labels
        feature_cols: List of feature column names
        target_col: Name of target column
        model_type: Model type ('xgboost', 'catboost', 'lightgbm')
        task_type: Task type ('regression', 'binary', 'multiclass')
        n_splits: Number of CV folds
        tscv_gap: Fallback purge width (bars) when purge_gap is unset
        purge_gap: Drop train rows whose label window overlaps the val fold
        bar_minutes: Bar length for timestamp purge (e.g. 240 for 240T)
        date_col: Optional date column for sorting
        tradable_col: Optional tradable mask column
        weight_col: Optional sample weight column
        model_params: Model-specific parameters
        use_gpu: Whether to use GPU

    Returns:
        Tuple of (models, avg_metric, results_df, used_feature_cols, preprocessor)
    """

    # Prepare data - separate numeric and categorical features
    # Load categorical column types from configuration
    column_types_config = load_column_types_config()
    categorical_cols_from_config = get_categorical_columns(
        feature_cols, column_types_config=column_types_config
    )

    numeric_feature_cols = []
    categorical_feature_cols = []
    categorical_mappings: Dict[str, Dict[str, int]] = {}

    for col in feature_cols:
        if col not in df.columns:
            continue

        # Determine if this is a categorical feature
        # Priority: 1) Explicit config mark, 2) Auto-detect object types if not in config
        is_categorical_from_config = col in categorical_cols_from_config
        is_object_type = df[col].dtype == object or df[col].dtype.name == "category"

        # If column is explicitly marked as categorical in config, use it
        # If column is object type and NOT explicitly marked in config (as numeric), treat as categorical
        # (This provides backward compatibility for unconfigured object columns)
        if is_categorical_from_config or (
            is_object_type and col not in column_types_config
        ):
            # This is a categorical feature - build deterministic mapping from TRAIN data.
            series = df[col].fillna("__MISSING__").astype(str)
            unique_count = series.nunique()
            if unique_count < 2:
                logging.warning(
                    f"Skipping categorical feature '{col}': only {unique_count} unique value(s) (constant)"
                )
                continue
            elif unique_count > 1000:
                logging.warning(
                    f"Skipping categorical feature '{col}': too many unique values ({unique_count} > 1000)"
                )
                continue
            else:
                categorical_feature_cols.append(col)
                # Stable mapping across runs: sort categories
                categories = sorted(set(series.unique().tolist()))
                if "__UNKNOWN__" not in categories:
                    categories.append("__UNKNOWN__")
                mapping = {v: i for i, v in enumerate(categories)}
                categorical_mappings[col] = mapping
                logging.info(
                    "Built categorical mapping for '%s': %d categories (including __UNKNOWN__)",
                    col,
                    len(mapping),
                )
        else:
            # Try to convert to numeric
            try:
                converted = pd.to_numeric(df[col], errors="coerce")
                # Check if conversion was successful (not all NaN)
                if not converted.isna().all():
                    numeric_feature_cols.append(col)
                else:
                    logging.warning(
                        f"Skipping feature column '{col}': all values are non-numeric after conversion"
                    )
            except (ValueError, TypeError) as e:
                logging.warning(f"Skipping non-numeric feature column '{col}': {e}")
                continue

    if not numeric_feature_cols and not categorical_feature_cols:
        raise ValueError(
            f"No valid feature columns found. Checked {len(feature_cols)} columns."
        )

    # Combine numeric and categorical features
    all_feature_cols = numeric_feature_cols + categorical_feature_cols

    # ------------------------------------------------------------------
    # Feature direction config (invert selected features)
    # This is config-driven and does NOT look at test labels (no leakage).
    # ------------------------------------------------------------------
    feature_multipliers: Dict[str, float] = {}
    try:
        import yaml
        from pathlib import Path
        from src.time_series_model.strategies.models.feature_direction import (
            expand_invert_features,
        )

        if model_params and isinstance(model_params, dict):
            # Single-file mode: allow passing invert_features directly (recommended),
            # or reading from a YAML file path if provided.
            invert_list = model_params.get("invert_features")
            if isinstance(invert_list, list):
                inv_cols = expand_invert_features(invert_list)
                for col in inv_cols:
                    feature_multipliers[col] = -1.0
            else:
                fd_cfg_path = model_params.get("feature_direction_config")
                if fd_cfg_path:
                    p = Path(str(fd_cfg_path))
                    if p.exists():
                        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                        invert_list2 = (
                            cfg.get("invert_features")
                            or cfg.get("invert")
                            or (cfg.get("feature_pipeline") or {}).get(
                                "invert_features"
                            )
                            or []
                        )
                        if isinstance(invert_list2, list):
                            inv_cols = expand_invert_features(invert_list2)
                            for col in inv_cols:
                                feature_multipliers[col] = -1.0
    except Exception:
        # Silent by design; direction config is optional.
        pass

    # Warn (non-fatal) if user asked to invert columns that are not in model inputs.
    if feature_multipliers:
        missing = [c for c in feature_multipliers.keys() if c not in all_feature_cols]
        if missing:
            logger.warning(
                "invert_features specified columns not present in model input columns (ignored): %s",
                missing,
            )

    preprocessor = FeaturePreprocessor(
        feature_cols=all_feature_cols,
        numeric_cols=numeric_feature_cols,
        categorical_cols=categorical_feature_cols,
        categorical_mappings=categorical_mappings,
        feature_multipliers=feature_multipliers,
    )

    # Build training feature matrix from the raw df (train split only)
    X = preprocessor.transform(df, feature_cols=all_feature_cols)

    # Update feature_cols to include both numeric and categorical (and keep deterministic order)
    feature_cols = all_feature_cols

    # Store categorical feature indices for LightGBM
    categorical_feature_indices = [
        i for i, col in enumerate(feature_cols) if col in categorical_feature_cols
    ]

    y = df[target_col].values

    # Handle tradable mask
    if tradable_col and tradable_col in df.columns:
        tradable = df[tradable_col].values
    else:
        tradable = np.ones(len(df), dtype=bool)

    # Handle sample weights
    if weight_col and weight_col in df.columns:
        weights = df[weight_col].values
    else:
        weights = None

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    horizon_bars = int(purge_gap) if purge_gap is not None else int(tscv_gap)
    ts_pos = _timestamps_from_df(df, date_col)
    if ts_pos is not None:
        ts_pos = pd.Series(pd.to_datetime(ts_pos, utc=True).to_numpy())

    models = []
    metric_scores = []
    fold_results = []

    requested_model_type = model_type
    if model_type == "xgboost" and not XGBOOST_AVAILABLE:
        if LIGHTGBM_AVAILABLE:
            logger.warning(
                "XGBoost not available. Falling back to LightGBM for training."
            )
            model_type = "lightgbm"
        else:
            raise ImportError("XGBoost is not installed")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        if horizon_bars > 0:
            train_idx = purge_train_indices(
                train_idx,
                val_idx,
                timestamps=ts_pos,
                horizon_bars=horizon_bars,
                bar_minutes=bar_minutes,
            )
            if train_idx.size == 0:
                logger.warning(
                    "Fold %s: no train rows after purge_gap=%s", fold + 1, horizon_bars
                )
                continue

        # Filter by tradable mask
        train_mask = tradable[train_idx]
        X_train = X[train_idx][train_mask]
        y_train = y[train_idx][train_mask]
        w_train = weights[train_idx][train_mask] if weights is not None else None

        X_val = X[val_idx]
        y_val = y[val_idx]

        # Train model based on type
        if model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not installed")
            model = _train_xgboost(
                X_train,
                y_train,
                X_val,
                y_val,
                task_type=task_type,
                w_train=w_train,
                model_params=model_params,
                use_gpu=use_gpu,
            )
        elif model_type == "catboost":
            if not CATBOOST_AVAILABLE:
                raise ImportError("CatBoost is not installed")
            model = _train_catboost(
                X_train,
                y_train,
                X_val,
                y_val,
                task_type=task_type,
                w_train=w_train,
                model_params=model_params,
                use_gpu=use_gpu,
            )
        elif model_type == "lightgbm":
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM is not installed")
            # Categorical feature indices are already relative to feature_cols order
            # Since X_train/X_val use the same column order, indices are correct
            model = _train_lightgbm(
                X_train,
                y_train,
                X_val,
                y_val,
                task_type=task_type,
                w_train=w_train,
                model_params=model_params,
                use_gpu=use_gpu,
                categorical_feature_indices=categorical_feature_indices,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        models.append(model)

        # Evaluate
        if task_type == "regression":
            pred_val = model.predict(X_val)
            metric = np.corrcoef(pred_val, y_val)[0, 1]  # Pearson correlation

            # 详细日志
            valid_mask = ~(np.isnan(pred_val) | np.isnan(y_val))
            if valid_mask.sum() > 0:
                pred_valid = pred_val[valid_mask]
                y_valid = y_val[valid_mask]
                logger.info(
                    f"   Fold {fold+1} (regression): "
                    f"Metric={metric:.4f}, "
                    f"n_samples={valid_mask.sum()}, "
                    f"pred_range=[{pred_valid.min():.4f}, {pred_valid.max():.4f}], "
                    f"pred_mean={pred_valid.mean():.4f}, "
                    f"y_range=[{y_valid.min():.4f}, {y_valid.max():.4f}], "
                    f"y_mean={y_valid.mean():.4f}"
                )
        elif task_type == "binary":
            if hasattr(model, "predict_proba"):
                pred_val = model.predict_proba(X_val)[:, -1]  # Probability
            else:
                pred_val = model.predict(X_val)
                if pred_val.ndim > 1:
                    pred_val = pred_val[:, -1]
            metric = np.corrcoef(pred_val, y_val)[0, 1]

            # 详细日志
            valid_mask = ~(np.isnan(pred_val) | np.isnan(y_val))
            if valid_mask.sum() > 0:
                pred_valid = pred_val[valid_mask]
                y_valid = y_val[valid_mask]
                pos_rate = (y_valid == 1).mean()
                logger.info(
                    f"   Fold {fold+1} (binary): "
                    f"Metric={metric:.4f}, "
                    f"n_samples={valid_mask.sum()}, "
                    f"pred_range=[{pred_valid.min():.4f}, {pred_valid.max():.4f}], "
                    f"pred_mean={pred_valid.mean():.4f}, "
                    f"pos_rate={pos_rate:.4f}, "
                    f"y_pos={int((y_valid == 1).sum())}, "
                    f"y_neg={int((y_valid == 0).sum())}"
                )
        else:  # multiclass
            if hasattr(model, "predict_proba"):
                pred_val = model.predict_proba(X_val)
            else:
                pred_val = model.predict(X_val)
            # Use accuracy as metric
            pred_class = np.argmax(pred_val, axis=1)
            metric = (pred_class == y_val).mean()

        metric_scores.append(metric)
        print(f"   Fold {fold+1}: Metric = {metric:.4f}")

        # Store fold results
        fold_results.append(
            {
                "fold": fold + 1,
                "metric": metric,
                "n_train": len(X_train),
                "n_val": len(X_val),
            }
        )

    avg_metric = np.mean(metric_scores) if metric_scores else 0.0
    std_metric = np.std(metric_scores) if len(metric_scores) > 1 else 0.0
    results_df = pd.DataFrame(fold_results)

    # 详细日志：CV指标统计
    if metric_scores:
        logger.info(
            f"CV指标统计: "
            f"mean={avg_metric:.4f}, "
            f"std={std_metric:.4f}, "
            f"min={min(metric_scores):.4f}, "
            f"max={max(metric_scores):.4f}, "
            f"folds={len(metric_scores)}"
        )

        # 如果CV指标异常低，打印警告
        if task_type == "binary" and avg_metric < 0.01:
            logger.warning(
                f"⚠️  CV指标异常低 ({avg_metric:.4f})，可能原因："
                f"1. 样本分布不平衡"
                f"2. 模型预测能力差"
                f"3. 交叉验证不稳定（std={std_metric:.4f})"
            )
        elif task_type == "regression" and avg_metric < 0.01:
            logger.warning(
                f"⚠️  CV指标异常低 ({avg_metric:.4f})，可能原因："
                f"1. 标签噪声太大"
                f"2. 模型预测能力差"
                f"3. 交叉验证不稳定（std={std_metric:.4f})"
            )

    return models, avg_metric, results_df, feature_cols, preprocessor


def _train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task_type: str = "regression",
    w_train: Optional[np.ndarray] = None,
    model_params: Optional[Dict] = None,
    use_gpu: bool = True,
):
    """Train XGBoost model"""
    import xgboost as xgb

    # Default parameters
    default_params = {
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }

    if task_type == "regression":
        default_params["objective"] = "reg:squareerror"
        default_params["eval_metric"] = "rmse"
    elif task_type == "binary":
        default_params["objective"] = "binary:logistic"
        default_params["eval_metric"] = "logloss"
    else:  # multiclass
        default_params["objective"] = "multi:softprob"
        default_params["eval_metric"] = "mlogloss"

    if use_gpu:
        default_params["tree_method"] = "gpu_hist"
        # xgboost 3.1+ removed gpu_id; use device instead
        default_params["device"] = "cuda:0"
        default_params.pop("gpu_id", None)

    if model_params:
        default_params.update(model_params)

    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Train
    booster = xgb.train(
        default_params,
        dtrain,
        num_boost_round=default_params.get("n_estimators", 1000),
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    return _XGBBoosterWrapper(booster, task_type)


class _XGBBoosterWrapper:
    """Adapter to provide sklearn-like predict/predict_proba interface."""

    def __init__(self, booster: "xgb.Booster", task_type: str) -> None:  # type: ignore[name-defined]
        import xgboost as xgb  # local import for type hints

        self.booster: xgb.Booster = booster
        self.task_type = task_type

    def _predict_raw(self, X) -> np.ndarray:
        import xgboost as xgb

        if isinstance(X, xgb.DMatrix):
            dmat = X
        else:
            dmat = xgb.DMatrix(X)
        return self.booster.predict(dmat)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._predict_raw(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = self._predict_raw(X)
        if self.task_type == "binary":
            preds = np.clip(preds, 1e-6, 1 - 1e-6)
            return np.column_stack([1 - preds, preds])
        elif self.task_type == "multiclass":
            return preds
        else:
            raise ValueError("predict_proba is not defined for regression task")


def _train_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task_type: str = "regression",
    w_train: Optional[np.ndarray] = None,
    model_params: Optional[Dict] = None,
    use_gpu: bool = True,
):
    """Train CatBoost model"""
    import catboost as cb  # type: ignore[import-untyped]

    # Default parameters
    default_params = {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": 42,
        "verbose": False,
    }

    if task_type == "regression":
        default_params["loss_function"] = "RMSE"
    elif task_type == "binary":
        default_params["loss_function"] = "Logloss"
    else:  # multiclass
        default_params["loss_function"] = "MultiClass"

    if use_gpu:
        default_params["task_type"] = "GPU"
        default_params["devices"] = "0"

    if model_params:
        default_params.update(model_params)

    # Create Pool
    train_pool = cb.Pool(X_train, label=y_train, weight=w_train)
    val_pool = cb.Pool(X_val, label=y_val)

    # Train
    model = cb.CatBoost(**default_params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=50,
        verbose=False,
    )

    return model


def _train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task_type: str = "regression",
    w_train: Optional[np.ndarray] = None,
    model_params: Optional[Dict] = None,
    use_gpu: bool = True,
    categorical_feature_indices: Optional[List[int]] = None,
):
    """Train LightGBM model"""
    import lightgbm as lgb
    import os

    # Default parameters
    default_params = {
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        # Reproducibility defaults (can be overridden by model_params)
        "seed": 42,
        "feature_fraction_seed": 42,
        "bagging_seed": 42,
        "data_random_seed": 42,
        "deterministic": True,
        "force_col_wise": True,
        # num_threads: multi-thread by default for speed.
        # Multi-seed search handles model exploration; non-determinism is acceptable.
        # Set MLBOT_DETERMINISTIC=1 (or --deterministic CLI) to force single-thread
        # for debugging / exact reproducibility.
        "num_threads": 1 if os.getenv("MLBOT_DETERMINISTIC", "0") == "1" else -1,
    }

    if task_type == "regression":
        default_params["objective"] = "regression"
        default_params["metric"] = "rmse"
    elif task_type == "binary":
        default_params["objective"] = "binary"
        default_params["metric"] = "binary_logloss"
    else:  # multiclass
        default_params["objective"] = "multiclass"
        default_params["metric"] = "multi_logloss"

    if use_gpu:
        default_params["device"] = "cuda"
        default_params["gpu_platform_id"] = 0
        default_params["gpu_device_id"] = 0

    # Handle n_estimators -> num_boost_round conversion
    num_boost_round = 1000
    if model_params:
        # IMPORTANT: do not mutate caller-provided dict (it is reused across folds).
        mp = dict(model_params)
        if "n_estimators" in mp:
            try:
                num_boost_round = int(mp.get("n_estimators") or num_boost_round)
            except Exception:
                pass
            mp.pop("n_estimators", None)
        default_params.update(mp)

    # If user passed a single seed, mirror it into all lgbm seed knobs for stability.
    # (LightGBM uses multiple RNG streams depending on features/bagging.)
    if "seed" in default_params:
        s = int(default_params.get("seed") or 42)
        default_params.setdefault("feature_fraction_seed", s)
        default_params.setdefault("bagging_seed", s)
        default_params.setdefault("data_random_seed", s)
        default_params.setdefault("drop_seed", s)

    # Remove n_estimators from default_params if it exists (LightGBM uses num_boost_round instead)
    default_params.pop("n_estimators", None)

    # Create Dataset with categorical features if specified
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        weight=w_train,
        categorical_feature=(
            categorical_feature_indices if categorical_feature_indices else None
        ),
    )
    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        reference=train_data,
        categorical_feature=(
            categorical_feature_indices if categorical_feature_indices else None
        ),
    )

    # Train
    model = lgb.train(
        default_params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    return model
