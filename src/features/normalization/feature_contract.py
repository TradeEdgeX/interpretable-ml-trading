"""
Normalization contract utilities.

Goal:
Turn "documentation declaration" into an executable contract:
- Every feature output column must have an explicit normalization contract.
- The contract is validated from config (today) and can be extended to runtime-returned
  metadata from feature functions (future).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable


@dataclass(frozen=True)
class FeatureNormalizationMeta:
    """Per-output-column normalization metadata (code contract)."""

    column: str
    method: str  # e.g. "position", "relative_close", "atr", "change_ratio", "tanh", "unitless"
    expected_range: Optional[Tuple[float, float]] = None
    cross_asset_comparable: bool = True
    notes: Optional[str] = None


def _infer_meta_for_feature(feature_name: str, feature_info: Dict[str, Any]) -> List[FeatureNormalizationMeta]:
    """
    Best-effort inference from feature_dependencies.yaml.
    This keeps the contract enforceable without requiring 200+ compute functions to be edited immediately.
    """
    output_cols = feature_info.get("output_columns") or [feature_name]
    compute_params = feature_info.get("compute_params") or {}
    desc = str(feature_info.get("description") or "")

    # Primary, explicit sources
    normalize_mode = compute_params.get("normalize_mode")
    output_norm = compute_params.get("output_normalization")
    # Per-output methods: prefer compute_params; also accept legacy top-level key on the feature node.
    output_norm_map = dict(feature_info.get("output_normalization_map") or {})
    output_norm_map.update(compute_params.get("output_normalization_map") or {})
    # Note: some features declare normalized at the top-level (older style); keep it honored.
    normalized_flag = bool(compute_params.get("normalized", False)) or bool(
        feature_info.get("normalized", False)
    )

    method: Optional[str] = None
    expected_range: Optional[Tuple[float, float]] = None

    if output_norm is not None:
        method = str(output_norm)
        if method == "tanh":
            expected_range = (-1.0, 1.0)
    elif normalize_mode is not None:
        method = str(normalize_mode)
    else:
        # Heuristic only when we are NOT explicitly saying "not normalized" / "未归一化"
        # (otherwise we'd misclassify raw-unit features as unitless).
        dl = desc.lower()
        says_not_norm = ("not normalized" in dl) or ("un-normalized" in dl) or ("未归一化" in desc)
        if (not says_not_norm) and (
            normalized_flag
            or ("normalized" in dl)
            or ("unitless" in dl)
            or ("similarity" in dl)
        ):
            # Explicitly mark as unitless when config/doc says so.
            method = "unitless"

    metas: List[FeatureNormalizationMeta] = []
    for col in output_cols:
        # Heuristics for common bounded indicators (explicit, unitless, cross-asset comparable)
        col_l = str(col).lower()
        col_expected_range: Optional[Tuple[float, float]] = expected_range
        col_method = method
        col_cross_asset = True

        # Per-column overrides (for multi-output features)
        if str(col) in output_norm_map:
            col_method = str(output_norm_map[str(col)])
            # If config explicitly marks a column as raw/price-unit, it is NOT cross-asset comparable.
            if col_method in {"raw", "price_unit", "usd"}:
                col_cross_asset = False
            if col_method == "tanh":
                col_expected_range = (-1.0, 1.0)
            elif col_method == "bounded_0_1":
                col_expected_range = (0.0, 1.0)
            elif col_method in {"bounded_-1_1", "bounded_neg1_1"}:
                col_expected_range = (-1.0, 1.0)
            elif col_method == "rank_rolling":
                col_expected_range = (0.0, 1.0)

        if col_method is None:
            if col_l in {"rsi", "rsi_f", "stoch_k", "stoch_d", "stochf_k", "stochf_d", "ultosc", "willr", "cmo"}:
                col_method = "bounded_0_100"
                col_expected_range = (0.0, 100.0)
            elif col_l in {"cci"}:
                col_method = "unitless"
            else:
                # Explicitly mark as raw if we cannot infer; this is still a contract (not missing).
                col_method = "raw"
                col_cross_asset = False

        metas.append(
            FeatureNormalizationMeta(
                column=col,
                method=col_method or "MISSING",
                expected_range=col_expected_range,
                cross_asset_comparable=col_cross_asset,
                notes=f"feature={feature_name}",
            )
        )
    return metas


def collect_feature_normalization_meta(
    feature_deps: Dict[str, Any],
    *,
    only_features: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Collect per-column normalization meta for features in deps.

    Returns a list of dicts with stable keys (friendly for JSON/Markdown reporting).
    """
    features = (feature_deps or {}).get("features", {}) or {}
    only = set(only_features) if only_features is not None else None

    rows: List[Dict[str, Any]] = []
    for feat_name, feat_info in features.items():
        if only is not None and feat_name not in only:
            continue
        metas = _infer_meta_for_feature(feat_name, feat_info)
        for m in metas:
            rows.append(
                {
                    "feature": feat_name,
                    "column": m.column,
                    "method": m.method,
                    "expected_range": m.expected_range,
                    "cross_asset_comparable": m.cross_asset_comparable,
                    "category": feat_info.get("category"),
                    "compute_func": feat_info.get("compute_func"),
                    "notes": m.notes,
                }
            )
    return rows


def validate_feature_dependencies_normalization(
    feature_deps: Dict[str, Any],
    *,
    mode: str = "error",  # "error" | "warn"
    allow_missing_for: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Validate that every feature output column has an explicit normalization method.
    Returns a report dict with details for diagnostics/CI.
    """
    allow_missing_for = allow_missing_for or []
    features = (feature_deps or {}).get("features", {}) or {}

    missing: List[Dict[str, str]] = []
    total_cols = 0
    raw: List[Dict[str, str]] = []

    for feat_name, feat_info in features.items():
        metas = _infer_meta_for_feature(feat_name, feat_info)
        for m in metas:
            total_cols += 1
            if m.method == "MISSING" and feat_name not in allow_missing_for:
                missing.append({"feature": feat_name, "column": m.column})
            if m.method == "raw" and feat_name not in allow_missing_for:
                raw.append({"feature": feat_name, "column": m.column})

    report = {
        "total_features": int(len(features)),
        "total_output_columns": int(total_cols),
        "missing_columns": missing,
        "missing_count": int(len(missing)),
        "raw_columns": raw,
        "raw_count": int(len(raw)),
        "ok": len(missing) == 0,
    }

    if missing and mode == "error":
        sample = ", ".join([f"{x['feature']}:{x['column']}" for x in missing[:10]])
        raise ValueError(
            f"Normalization contract violated: {len(missing)} output columns missing method. "
            f"Examples: {sample}"
        )

    return report


