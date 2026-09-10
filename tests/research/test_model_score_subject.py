"""ModelScore subject resolution and inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.research.subjects.resolve import (
    attach_subject_column,
    parse_subject,
    subject_from_args,
)
from src.research.tree_trainer import train_lightgbm_classifier


def test_parse_subject_feature_and_model_column():
    assert parse_subject("feature:pulse_z").column == "pulse_z"
    assert parse_subject("pulse_z").column == "pulse_z"
    assert parse_subject("model.score:my_score_col").column == "my_score_col"


def test_subject_from_args_requires_one():
    with pytest.raises(ValueError):
        subject_from_args(subject=None, feature=None)
    assert subject_from_args(subject=None, feature="x").column == "x"


def test_model_score_inference_from_fit_artifact(tmp_path: Path):
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "y": (rng.random(n) > 0.5).astype(int),
        }
    )
    out_dir = tmp_path / "fit"
    train_lightgbm_classifier(df, ["f1", "f2"], "y", out_dir, seed=0)
    assert (out_dir / "model_manifest.json").is_file()

    subj = parse_subject(f"model.score:{out_dir / 'model.txt'}")
    scored, col = attach_subject_column(df, subj)
    assert col == "_research_model_score"
    assert col in scored.columns
    assert scored[col].between(0, 1).all()
