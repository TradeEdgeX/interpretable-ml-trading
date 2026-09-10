"""Parse and materialize research subjects (Feature / ModelScore)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

from src.research.subjects.feature import Feature, ModelScore

DEFAULT_MODEL_SCORE_COL = "_research_model_score"


@dataclass(frozen=True)
class ResolvedSubject:
    kind: str
    column: str
    model_path: Optional[Path] = None


def parse_subject(spec: str) -> ResolvedSubject:
    """Parse ``feature:col``, ``model.score:path``, or ``model.score:col``."""
    raw = (spec or "").strip()
    if not raw:
        raise ValueError("empty subject spec")

    if raw.startswith("model.score:"):
        tail = raw.split(":", 1)[1].strip()
        path = Path(tail)
        if path.suffix == ".txt" and path.is_file():
            return ResolvedSubject(
                "model_score", DEFAULT_MODEL_SCORE_COL, model_path=path.resolve()
            )
        if path.is_dir():
            model_file = path / "model.txt"
            if model_file.is_file():
                return ResolvedSubject(
                    "model_score", DEFAULT_MODEL_SCORE_COL, model_path=model_file.resolve()
                )
        if "/" in tail or tail.endswith(".txt"):
            raise FileNotFoundError(f"model.score path not found: {tail}")
        return ResolvedSubject("model_score", tail, model_path=None)

    if raw.startswith("feature:"):
        return ResolvedSubject("feature", raw.split(":", 1)[1].strip())

    return ResolvedSubject("feature", raw)


def subject_from_args(
    *,
    subject: Optional[str],
    feature: Optional[str],
) -> ResolvedSubject:
    if subject:
        return parse_subject(subject)
    if feature:
        return ResolvedSubject("feature", feature)
    raise ValueError("pass --feature or --subject")


def load_model_manifest(model_path: Path) -> dict:
    base = model_path.parent if model_path.is_file() else model_path
    manifest_path = base / "model_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"model_manifest.json missing next to {model_path}; run research fit first"
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def add_model_score_column(
    df: pd.DataFrame,
    model_path: Path,
    *,
    out_col: str = DEFAULT_MODEL_SCORE_COL,
) -> pd.DataFrame:
    """Run LightGBM inference; return copy of df with score column attached."""
    import lightgbm as lgb

    manifest = load_model_manifest(model_path)
    feature_cols = list(manifest["feature_cols"])
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"model score inference missing columns: {missing[:5]}")

    booster = lgb.Booster(model_file=str(model_path))
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    out = df.copy()
    out[out_col] = booster.predict(X)
    return out


def attach_subject_column(
    df: pd.DataFrame,
    subject: Union[ResolvedSubject, Feature, ModelScore],
) -> Tuple[pd.DataFrame, str]:
    """Return (df, column_name) ready for scan/plateau/robustness kernels."""
    if isinstance(subject, Feature):
        if subject.col not in df.columns:
            raise KeyError(f"Feature column missing: {subject.col}")
        return df, subject.col

    if isinstance(subject, ModelScore):
        if subject.model_path is not None:
            col = subject.col or DEFAULT_MODEL_SCORE_COL
            return add_model_score_column(df, subject.model_path, out_col=col), col
        if subject.col not in df.columns:
            raise KeyError(f"ModelScore column missing: {subject.col}")
        return df, subject.col

    if subject.kind == "model_score" and subject.model_path is not None:
        return (
            add_model_score_column(df, subject.model_path, out_col=subject.column),
            subject.column,
        )
    if subject.column not in df.columns:
        raise KeyError(f"subject column missing: {subject.column}")
    return df, subject.column
