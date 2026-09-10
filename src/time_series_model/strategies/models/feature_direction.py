from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _load_feature_deps(feature_deps_path: str) -> Dict[str, Any]:
    p = Path(feature_deps_path)
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def expand_invert_features(
    invert_features: List[str] | None,
    *,
    feature_deps: Optional[Dict[str, Any]] = None,
    feature_deps_path: str = "config/feature_dependencies.yaml",
) -> List[str]:
    """
    Expand a user-provided invert_features list into concrete output-column names.

    Supported inputs:
    - Output column name: "trend_r2_50"              -> invert that single column
    - Feature node name: "trend_r2_50_f" (preferred) -> invert ALL output_columns of that node

    Notes:
    - This is used only to multiply selected feature columns by -1 before training/inference.
    - Entries that cannot be resolved are returned as-is (best-effort), and callers can warn
      if the column isn't present in model input columns.
    """
    raw = invert_features or []
    # Stable de-dup while expanding
    out: List[str] = []
    seen = set()

    if feature_deps is None:
        feature_deps = _load_feature_deps(feature_deps_path)
    features = (
        (feature_deps or {}).get("features", {})
        if isinstance(feature_deps, dict)
        else {}
    )

    for item in raw:
        if not isinstance(item, str):
            continue
        name = item.strip()
        if not name:
            continue

        # If user passed a feature node, expand to its output columns.
        if name.endswith("_f") and name in features:
            cols = features.get(name, {}).get("output_columns", []) or []
            if isinstance(cols, list) and cols:
                for c in cols:
                    c = str(c).strip()
                    if c and c not in seen:
                        seen.add(c)
                        out.append(c)
                continue

        # Otherwise treat as a column name.
        if name not in seen:
            seen.add(name)
            out.append(name)

    return out
