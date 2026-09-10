"""CLI end-to-end tests for research plateau (entry_rr + model.score subject)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research import plateau as plateau_mod
from scripts.research import robustness as robustness_mod
from src.research.tree_trainer import train_lightgbm_classifier


def _synthetic_entry_rr_df(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 0.5, size=n))
    high = close + rng.uniform(0.1, 0.5, size=n)
    low = close - rng.uniform(0.1, 0.5, size=n)
    feat = rng.normal(size=n)
    direction = np.where(feat > 0, 1.0, -1.0)
    return pd.DataFrame(
        {
            "symbol": ["BTC"] * n,
            "high": high,
            "low": low,
            "close": close,
            "atr": np.full(n, 1.0),
            "entry_direction": direction,
            "gate_decision": ["allow"] * n,
            "pulse_z": feat,
            "success_no_rr_extreme": (feat > 0).astype(int),
        }
    )


def test_plateau_snotio_entry_rr_cli(tmp_path: Path) -> None:
    pq = tmp_path / "logs.parquet"
    _synthetic_entry_rr_df(200).to_parquet(pq)
    out_json = tmp_path / "plateau.json"
    rc = plateau_mod.main(
        [
            "--features-parquet",
            str(pq),
            "--strategy",
            "srb",
            "--kpi",
            "snotio",
            "--snotio-mode",
            "entry_rr",
            "--feature",
            "pulse_z",
            "--operator",
            "<=",
            "--grid",
            "0,0.5,1",
            "--min-trades",
            "5",
            "--output",
            str(out_json),
        ]
    )
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload.get("snotio_mode") == "entry_rr"
    assert len(payload.get("rows", [])) == 3


def test_plateau_lift_cli(tmp_path: Path) -> None:
    from scripts.research import plateau as plateau_mod

    rng = __import__("numpy").random.default_rng(0)
    n = 200
    feat = rng.uniform(0, 1, n)
    df = __import__("pandas").DataFrame(
        {
            "tpc_semantic_chop": feat,
            "is_good": (feat < 0.4).astype(int),
        }
    )
    pq = tmp_path / "features.parquet"
    df.to_parquet(pq)
    out_json = tmp_path / "lift.json"
    rc = plateau_mod.main(
        [
            "--features-parquet",
            str(pq),
            "--kpi",
            "lift",
            "--feature",
            "tpc_semantic_chop",
            "--operator",
            "gt",
            "--grid",
            "0.2,0.3,0.4,0.5,0.6",
            "--label",
            "is_good",
            "--output",
            str(out_json),
        ]
    )
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload.get("kpi") == "lift"
    assert payload.get("deny_operator") == "gt"


def test_plateau_model_score_subject_cli(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "y": (rng.random(n) > 0.5).astype(int),
            "success_no_rr_extreme": (rng.random(n) > 0.45).astype(int),
        }
    )
    pq = tmp_path / "features.parquet"
    df.to_parquet(pq)
    fit_dir = tmp_path / "fit"
    train_lightgbm_classifier(df, ["f1", "f2"], "y", fit_dir, seed=0)
    model_path = fit_dir / "model.txt"
    out_json = tmp_path / "model_plateau.json"
    rc = plateau_mod.main(
        [
            "--features-parquet",
            str(pq),
            "--subject",
            f"model.score:{model_path}",
            "--kpi",
            "label",
            "--operator",
            "<=",
            "--grid",
            "0.3,0.5,0.7",
            "--label",
            "success_no_rr_extreme",
            "--output",
            str(out_json),
        ]
    )
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload.get("subject", "").startswith("model.score:")
    assert len(payload.get("rows", [])) == 3


def test_robustness_missing_label_exits_cleanly(tmp_path: Path) -> None:
    pq = tmp_path / "features.parquet"
    n = 50
    pd.DataFrame(
        {
            "pulse_z": np.linspace(-1, 1, n),
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        }
    ).to_parquet(pq)
    rc = robustness_mod.main(
        [
            "--features-parquet",
            str(pq),
            "--feature",
            "pulse_z",
            "--operator",
            "<=",
            "--threshold",
            "0.5",
            "--label",
            "missing_label_col",
        ]
    )
    assert rc == 3
