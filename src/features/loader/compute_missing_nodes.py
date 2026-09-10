"""Compute DAG feature nodes whose outputs are still missing.

Used after FeatureStore merge: IFC may skip transitive tick-deps (e.g.
``src_archive_orderflow_confirm_f`` needing FS ``cvd`` / ``ofci_pct``), then FS
fills those inputs — this helper materializes the bar-only dependent outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yaml

from src.features.loader.feature_computer import _build_call_args
from src.features.registry import ensure_features_registered, get_compute_func

_DEFAULT_DEPS = Path("config/feature_dependencies.yaml")


def load_feature_deps(
    path: str | Path = _DEFAULT_DEPS,
) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    feats = raw.get("features")
    return feats if isinstance(feats, dict) else {}


def compute_missing_feature_nodes(
    df: pd.DataFrame,
    node_names: Sequence[str],
    *,
    feature_deps: Optional[Dict[str, Any]] = None,
    feature_deps_path: str | Path = _DEFAULT_DEPS,
) -> Tuple[pd.DataFrame, List[str]]:
    """Fill missing ``output_columns`` for nodes whose ``required_columns`` exist.

    Returns ``(df, computed_node_names)``. Idempotent when outputs already present.
    """
    if df is None or df.empty or not node_names:
        return df, []

    ensure_features_registered()
    feats_cfg = feature_deps if feature_deps is not None else load_feature_deps(feature_deps_path)
    if not feats_cfg:
        return df, []

    out = df
    computed: List[str] = []
    pending = [str(n) for n in node_names if str(n) in feats_cfg]

    # Iterate until fixed point (deps may unlock later nodes).
    while pending:
        progressed = False
        for n in list(pending):
            info = feats_cfg.get(n)
            if not isinstance(info, dict):
                pending.remove(n)
                continue
            output_cols = [str(c) for c in (info.get("output_columns") or [])]
            if output_cols and all(c in out.columns for c in output_cols):
                pending.remove(n)
                continue
            req_cols = set(info.get("required_columns") or [])
            if req_cols and not req_cols.issubset(out.columns):
                continue
            compute_func_name = info.get("compute_func", n)
            cfn = get_compute_func(str(compute_func_name))
            if cfn is None:
                pending.remove(n)
                continue
            try:
                call_args, call_kwargs = _build_call_args(info, out, n)
                result = cfn(*call_args, **call_kwargs)
            except Exception:
                # Hard fail on this node; do not spin forever.
                pending.remove(n)
                continue

            if isinstance(result, pd.DataFrame):
                for col in result.columns:
                    out[col] = result[col]
            elif isinstance(result, pd.Series):
                name = output_cols[0] if output_cols else (result.name or n)
                out[str(name)] = result
            else:
                pending.remove(n)
                continue
            computed.append(n)
            pending.remove(n)
            progressed = True
        if not progressed:
            break

    return out, computed
