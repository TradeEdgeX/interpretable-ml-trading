"""Utilities for loading globally excluded raw-scale feature columns."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_SCALE_PATH = PROJECT_ROOT / "config" / "raw_scale_columns.yaml"
DEFAULT_FEATURE_DEPS_PATH = PROJECT_ROOT / "config" / "feature_dependencies.yaml"


def _flatten_raw_scale_config(raw: Any) -> set[str]:
    cols: set[str] = set()
    if isinstance(raw, dict):
        for vals in raw.values():
            if isinstance(vals, (list, tuple, set)):
                cols.update(str(v) for v in vals if str(v).strip())
    elif isinstance(raw, (list, tuple, set)):
        cols.update(str(v) for v in raw if str(v).strip())
    return cols


def load_raw_scale_columns(
    raw_scale_path: str | Path | None = None,
    *,
    legacy_feature_deps_path: str | Path | None = None,
) -> set[str]:
    """Load columns that must be excluded from cross-asset feature scans.

    The standalone ``config/raw_scale_columns.yaml`` is the source of truth.
    ``legacy_feature_deps_path`` is only a compatibility fallback for older
    branches that still keep ``raw_scale_columns`` in ``feature_dependencies``.
    """
    candidates = [Path(raw_scale_path) if raw_scale_path else DEFAULT_RAW_SCALE_PATH]
    candidates.append(
        Path(legacy_feature_deps_path)
        if legacy_feature_deps_path
        else DEFAULT_FEATURE_DEPS_PATH
    )

    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cols = _flatten_raw_scale_config(cfg.get("raw_scale_columns", {}))
        if cols:
            return cols
    return set()
