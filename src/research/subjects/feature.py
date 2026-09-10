from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass(frozen=True)
class Feature:
    col: str


@dataclass(frozen=True)
class RuleExpr:
    expr: str


@dataclass(frozen=True)
class ModelScore:
    col: str
    model_path: Optional[Path] = None


@dataclass(frozen=True)
class FeaturePool:
    features: List[str]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FeaturePool":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        fp = raw.get("feature_pipeline", {}) or {}
        requested = fp.get("requested_features", []) or []
        return cls(features=[str(x) for x in requested])


def resolve_pool_columns(
    pool: FeaturePool,
    df_columns: List[str],
    deps_path: str | Path = "config/feature_dependencies.yaml",
) -> List[str]:
    """Map ``requested_features`` (_f nodes) to parquet output column names."""
    deps_file = Path(deps_path)
    if not deps_file.exists():
        return []
    raw = yaml.safe_load(deps_file.read_text(encoding="utf-8")) or {}
    all_features = raw.get("features", {}) or {}
    available = set(df_columns)
    resolved: List[str] = []
    for feat in pool.features:
        feat_def = all_features.get(feat, {}) or {}
        for col in feat_def.get("output_columns", []) or []:
            if col in available:
                resolved.append(str(col))
    return sorted(set(resolved))
