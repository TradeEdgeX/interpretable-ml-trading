"""Map (strategy, layer) to writeback paths and optional subset masks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

LAYER_WRITEBACK: Dict[str, str] = {
    "regime": "archetypes/regime.yaml",
    "prefilter": "archetypes/prefilter.yaml",
    "gate": "archetypes/gate.yaml",
    "entry": "archetypes/entry_filters.yaml",
    "direction": "archetypes/direction.yaml",
}

FEATURES_POOL_YAML: Dict[str, str] = {
    "regime": "features.yaml",
    "prefilter": "features_prefilter.yaml",
    "gate": "features_gate.yaml",
    "entry": "features_entry_filter.yaml",
    "direction": "features.yaml",
}


def strategies_root(strategy: str, root: str = "config/strategies") -> Path:
    return Path(root) / strategy


def writeback_path(strategy: str, layer: str, root: str = "config/strategies") -> Path:
    rel = LAYER_WRITEBACK.get(layer)
    if not rel:
        raise ValueError(f"Unknown layer for writeback: {layer!r}")
    return strategies_root(strategy, root) / rel


def feature_pool_path(strategy: str, layer: str, root: str = "config/strategies") -> Path:
    rel = FEATURES_POOL_YAML.get(layer, "features.yaml")
    p = strategies_root(strategy, root) / rel
    if not p.exists():
        p = strategies_root(strategy, root) / "features.yaml"
    return p


def resolve_features_parquet(
    strategy: str,
    explicit: Optional[str] = None,
    results_root: str = "results/train_final",
) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    base = Path(results_root) / strategy
    if not base.exists():
        return None
    candidates = sorted(
        base.glob(f"train_final_*/{strategy}/features_labeled.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def build_layer_mask(df: pd.DataFrame, strategy: str, after_layer: str) -> pd.Series:
    """Best-effort layer pass mask from parquet decision columns."""
    col_map = {
        "prefilter": ("prefilter_pass", "prefilter_ok"),
        "gate": ("gate_decision", "gate_ok"),
        "entry": ("entry_filter_pass", "entry_ok"),
    }
    if after_layer == "regime":
        return pd.Series(True, index=df.index)
    for col in col_map.get(after_layer, ()):
        if col in df.columns:
            if col == "gate_decision":
                return df[col].astype(str).str.lower().eq("allow").fillna(False)
            return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(bool)
    return pd.Series(True, index=df.index)


def resolve_layer_context(
    strategy: str, layer: Optional[str], root: str = "config/strategies"
) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (feature_pool_yaml, writeback_yaml) for CLI --layer."""
    if not layer:
        return None, None
    return feature_pool_path(strategy, layer, root), writeback_path(strategy, layer, root)
