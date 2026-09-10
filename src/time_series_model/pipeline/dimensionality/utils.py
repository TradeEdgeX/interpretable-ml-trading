"""Utility functions for dimensionality comparison."""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Try to import yaml, but make it optional
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def _slugify(value: str, default: str = "unknown") -> str:
    """Create a filesystem-friendly slug."""
    if value is None:
        return default
    value = str(value).strip()
    if not value:
        return default
    # Replace commas with hyphens first to keep multi-symbol ordering visible
    value = value.replace(",", "-")
    slug = re.sub(r"[^A-Za-z0-9_\-]+", "-", value)
    slug = re.sub(r"-{2,}", "-", slug).strip("-_")
    return slug or default


def _get_primary_metric(perf: Dict) -> Tuple[str, Optional[float]]:
    """Return the primary evaluation metric name/value for a stage."""
    if not perf:
        return "", None

    financial = perf.get("financial_metrics") or {}
    win_rate = financial.get("win_rate")
    if win_rate is not None:
        return "win_rate", float(win_rate)

    classification = perf.get("classification_metrics") or {}
    for key in ("f1_macro", "f1_weighted", "accuracy"):
        val = classification.get(key)
        if val is not None:
            return key, float(val)

    return "r2", perf.get("r2")


def _derive_feature_insights(stage_baseline: Dict, stage_candidate: Dict) -> Dict:
    """Summarise whether representative features improve over baseline."""
    metric_name_base, metric_base = _get_primary_metric(stage_baseline)
    metric_name_cand, metric_cand = _get_primary_metric(stage_candidate)

    metric_name = metric_name_cand or metric_name_base or "r2"
    delta = None
    if metric_base is not None and metric_cand is not None:
        delta = float(metric_cand) - float(metric_base)

    effective = delta is not None and delta > 0

    return {
        "metric_name": metric_name,
        "baseline_value": float(metric_base) if metric_base is not None else None,
        "candidate_value": float(metric_cand) if metric_cand is not None else None,
        "delta": delta,
        "effective": effective,
        "baseline_stage": "stage1_all_features",
        "candidate_stage": "stage3_representatives",
        "recommended_stage": (
            "stage3_representatives" if effective else "stage1_all_features"
        ),
    }


def load_top_factors_list(path: str) -> List[str]:
    """Load top factors from JSON or YAML file.

    Supports both JSON (.json) and YAML (.yaml, .yml) formats.

    Args:
        path: Path to top_factors file (JSON or YAML)

    Returns:
        List of feature names

    Raises:
        ValueError: If file extension is not supported or YAML is not available
    """
    # Detect file format from extension
    file_ext = os.path.splitext(path)[1].lower()

    # Load data based on file format
    with open(path, "r", encoding="utf-8") as f:
        if file_ext in (".yaml", ".yml"):
            if not YAML_AVAILABLE:
                raise ValueError(
                    f"YAML file detected but 'yaml' package is not installed. "
                    f"Install with: pip install pyyaml"
                )
            data = yaml.safe_load(f)
        elif file_ext == ".json":
            data = json.load(f)
        else:
            # Try JSON first, then YAML if available
            try:
                f.seek(0)
                data = json.load(f)
            except json.JSONDecodeError:
                if YAML_AVAILABLE:
                    f.seek(0)
                    data = yaml.safe_load(f)
                else:
                    raise ValueError(
                        f"Unsupported file format: {file_ext}. "
                        f"Supported: .json, .yaml, .yml"
                    )

    # Parse data structure (same logic for both JSON and YAML)
    if isinstance(data, dict):
        # Check for top_factors (JSON format)
        if "top_factors" in data:
            top_factors_list = data["top_factors"]
            if isinstance(top_factors_list, list):
                return [
                    item.get("name", item) if isinstance(item, dict) else item
                    for item in top_factors_list
                ]
        # Check for features.required_features (YAML config format)
        elif "features" in data and isinstance(data["features"], dict):
            if "required_features" in data["features"]:
                required = data["features"]["required_features"]
                if isinstance(required, list):
                    return required
        # Check for top-level features (list)
        elif "features" in data:
            return data["features"] if isinstance(data["features"], list) else []
        # Check for top-level required_features
        elif "required_features" in data:
            return (
                data["required_features"]
                if isinstance(data["required_features"], list)
                else []
            )
    elif isinstance(data, list):
        # Direct list format (both JSON and YAML support this)
        return data

    return []


def filter_engineered_by_topk(
    engineered_data: Dict[str, pd.DataFrame], top_list: List[str]
) -> Dict[str, pd.DataFrame]:
    """Filter engineered data to only include top-k features.

    Args:
        engineered_data: Dictionary of {timeframe: DataFrame} with engineered features
        top_list: List of feature names to keep

    Returns:
        Filtered dictionary with only top-k features
    """
    if not top_list:
        return engineered_data

    top_set = set(top_list)
    result = {}

    for key, df in engineered_data.items():
        # Keep only columns in top_list plus data columns
        data_cols = {"open", "high", "low", "close", "volume", "timestamp", "datetime"}
        cols_to_keep = [
            c
            for c in df.columns
            if c in top_set
            or c in data_cols
            or not pd.api.types.is_numeric_dtype(df[c])
        ]
        result[key] = df[cols_to_keep]

    return result
