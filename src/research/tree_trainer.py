"""LightGBM research trainer (layer-agnostic)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class FitResult:
    model_path: Path
    feature_cols: List[str]
    metrics: Dict[str, Any]


def _write_feature_importance_audit(
    booster: Any,
    X_val: pd.DataFrame,
    feature_cols: List[str],
    output_dir: Path,
    *,
    seed: int = 42,
    shap_sample: int = 500,
) -> None:
    """Write gain + optional SHAP mean-|value| audit (research-only, no yaml writes)."""
    gain = booster.feature_importance(importance_type="gain")
    gain_map = {
        feature_cols[i]: float(gain[i])
        for i in range(min(len(feature_cols), len(gain)))
    }
    gain_ranked = sorted(gain_map.items(), key=lambda x: -x[1])[:30]

    shap_map: Dict[str, float] = {}
    shap_note: Optional[str] = None
    try:
        import shap

        sample_n = min(shap_sample, len(X_val))
        if sample_n > 0:
            X_sample = X_val.iloc[:sample_n]
            explainer = shap.TreeExplainer(booster)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
            mean_abs = np.abs(shap_values).mean(axis=0)
            for i, col in enumerate(feature_cols):
                if i < len(mean_abs):
                    shap_map[col] = float(mean_abs[i])
    except ImportError:
        shap_note = "shap not installed; gain-only audit"
    except Exception as exc:  # pragma: no cover - optional audit path
        shap_note = f"shap audit skipped: {exc}"

    shap_ranked = sorted(shap_map.items(), key=lambda x: -x[1])[:30]
    audit: Dict[str, Any] = {
        "gain_top": [{"feature": f, "gain": v} for f, v in gain_ranked],
        "shap_mean_abs_top": [{"feature": f, "mean_abs_shap": v} for f, v in shap_ranked],
        "seed": seed,
        "shap_sample_n": min(shap_sample, len(X_val)),
    }
    if shap_note:
        audit["shap_note"] = shap_note
    (output_dir / "feature_importance.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )


def train_lightgbm_classifier(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    output_dir: Path,
    *,
    seed: int = 42,
) -> FitResult:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    output_dir.mkdir(parents=True, exist_ok=True)
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, shuffle=False
    )
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": seed,
        "num_leaves": 31,
        "learning_rate": 0.05,
    }
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=200,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    model_path = output_dir / "model.txt"
    booster.save_model(str(model_path))
    pred = booster.predict(X_val)
    auc = float(
        __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(
            y_val, pred
        )
    )
    metrics = {"val_auc": auc, "n_train": len(X_train), "n_val": len(X_val)}
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "model_manifest.json").write_text(
        json.dumps(
            {
                "feature_cols": feature_cols,
                "label_col": label_col,
                "model_path": "model.txt",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_feature_importance_audit(
        booster, X_val, feature_cols, output_dir, seed=seed
    )
    return FitResult(model_path=model_path, feature_cols=feature_cols, metrics=metrics)
