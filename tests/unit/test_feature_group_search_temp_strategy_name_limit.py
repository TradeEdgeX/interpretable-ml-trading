from __future__ import annotations

import json

import yaml
from pathlib import Path


def test_make_temp_strategy_handles_long_suffix(tmp_path: Path):
    from src.time_series_model.diagnostics import feature_group_search as fgs

    base_dir = tmp_path / "base_strategy"
    base_dir.mkdir(parents=True)
    (base_dir / "features.yaml").write_text(
        "name: base\nfeature_pipeline:\n  requested_features: []\n", encoding="utf-8"
    )

    tmp_root = tmp_path / "tmp_strategies"
    tmp_root.mkdir(parents=True)

    # Create an extremely long suffix that would exceed typical 255-byte component limits
    long_suffix = "beam_" + ("very_long_group_name__" * 80)

    out_dir = fgs._make_temp_strategy(
        base_dir=base_dir,
        tmp_root=tmp_root,
        name_suffix=long_suffix,
        requested_features=["atr_f"],
        invert_features=[],
    )

    # The directory should exist and its name should be bounded.
    assert out_dir.exists()
    assert len(out_dir.name) < 255

    feats = (
        yaml.safe_load((out_dir / "features.yaml").read_text(encoding="utf-8")) or {}
    )
    assert "name" in feats
    assert len(str(feats["name"])) < 512


def test_run_one_seed_reuses_existing_results(tmp_path: Path, monkeypatch):
    from src.time_series_model.diagnostics import feature_group_search as fgs

    # Minimal strategy dir so _strategy_name can resolve.
    strat_dir = tmp_path / "s"
    strat_dir.mkdir(parents=True)
    (strat_dir / "features.yaml").write_text(
        "name: s\nfeature_pipeline:\n  requested_features: []\n", encoding="utf-8"
    )

    cfg = fgs.SearchConfig(
        base_strategy_dir=strat_dir,
        timeframe="240T",
        symbol="BTCUSDT",
        start_date="2023-01-01",
        end_date="2025-12-31",
        test_size=0.3,
        seeds=[1],
        output_dir=tmp_path / "out",
        deterministic=True,
        no_docker=True,
    )

    out_root = tmp_path / "seed_out"
    existing = out_root / "s" / "results.json"
    existing.parent.mkdir(parents=True)
    existing.write_text(json.dumps({"strategy": "s"}), encoding="utf-8")

    def _boom(*args, **kwargs):
        raise AssertionError(
            "subprocess.run should not be called when results already exist"
        )

    monkeypatch.setattr(fgs.subprocess, "run", _boom)

    p = fgs._run_one_seed(strategy_dir=strat_dir, cfg=cfg, seed=1, out_root=out_root)
    assert p == existing
