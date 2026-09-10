from __future__ import annotations

import pandas as pd


def _fake_summary(*, strategy: str, score: float, trades: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "strategy": strategy,
                "task": "regression",
                "Sharpe_mean": float(score),
                "return%_mean": 0.0,
                "DD%_mean": 0.0,
                "trades_mean": float(trades),
                "CV_mean": 0.0,
                "corr_mean": 0.0,
            }
        ]
    )


def test_pipeline_halving_prefilter_then_beam_then_prune(tmp_path, monkeypatch):
    """
    Pipeline behavior:
    - Halving prefilter should keep B,C and drop A based on stage-1 cheap scores.
    - Beam on survivors should find synergy B+C.
    - Prune should remove C if B alone is better (simulated).
    """
    from src.time_series_model.diagnostics import feature_group_search as fgs

    base_dir = tmp_path / "base_strategy"
    base_dir.mkdir(parents=True)
    (base_dir / "features.yaml").write_text(
        "name: base\nfeature_pipeline:\n  requested_features: []\n", encoding="utf-8"
    )

    cfg = fgs.SearchConfig(
        base_strategy_dir=base_dir,
        timeframe="240T",
        symbol="BTCUSDT",
        start_date="2023-01-01",
        end_date="2025-10-31",
        test_size=0.3,
        seeds=[1, 2, 3, 4, 5],
        output_dir=tmp_path / "out",
        deterministic=True,
        no_docker=True,
    )

    groups = {"A": ["fa"], "B": ["fb"], "C": ["fc"]}

    def _stub_run_seed_sweep_for_strategy(*, strategy_dir, cfg, run_id):
        # Baseline
        if run_id == "baseline":
            return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=1.0)

        # Halving prefilter single-add evaluations
        if run_id.startswith("prefilter_add_") and run_id.endswith("__halving_s1"):
            if "prefilter_add_A" in run_id:
                return pd.DataFrame(), _fake_summary(
                    strategy=strategy_dir.name, score=0.5
                )
            if "prefilter_add_B" in run_id:
                return pd.DataFrame(), _fake_summary(
                    strategy=strategy_dir.name, score=1.4
                )
            if "prefilter_add_C" in run_id:
                return pd.DataFrame(), _fake_summary(
                    strategy=strategy_dir.name, score=1.3
                )
        if run_id.startswith("prefilter_add_") and run_id.endswith("__halving_s5"):
            if "prefilter_add_B" in run_id:
                return pd.DataFrame(), _fake_summary(
                    strategy=strategy_dir.name, score=1.45
                )
            if "prefilter_add_C" in run_id:
                return pd.DataFrame(), _fake_summary(
                    strategy=strategy_dir.name, score=1.35
                )

        # Beam evaluations on survivors
        if run_id.startswith("beam_step1_sel_"):
            if run_id.endswith("B"):
                return pd.DataFrame(), _fake_summary(
                    strategy=strategy_dir.name, score=1.4
                )
            if run_id.endswith("C"):
                return pd.DataFrame(), _fake_summary(
                    strategy=strategy_dir.name, score=1.3
                )
        if run_id.startswith("beam_step2_sel_"):
            if run_id.endswith("B__C"):
                return pd.DataFrame(), _fake_summary(
                    strategy=strategy_dir.name, score=2.0
                )  # synergy

        # Prune stage: make B alone better than B+C
        if run_id.startswith("prune_init__"):
            # initial is B__C
            return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=2.0)
        if run_id.startswith("prune_try__keep_") and "__rm_C" in run_id:
            # keep B only
            return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=2.1)
        if run_id.startswith("prune_try__keep_") and "__rm_B" in run_id:
            # keep C only
            return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=1.2)

        # Default
        return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=0.0)

    monkeypatch.setattr(
        fgs, "run_seed_sweep_for_strategy", _stub_run_seed_sweep_for_strategy
    )

    res = fgs.pipeline_sh_beam_sffs(
        cfg=cfg,
        base_features=[],
        groups=groups,
        max_steps=2,
        objective="Sharpe_mean",
        min_trades=10,
        stages=[1, 5],
        top_fraction=0.67,
        min_survivors=2,
        target_survivors=2,
        beam_width=2,
        sffs_max_backward_steps=2,
    )

    assert res["search_algo"] == "pipeline_sh_beam_sffs"
    # Prune should remove C, leaving only B
    assert res["selected_groups"] == ["B"]


def test_pipeline_invert_is_only_picked_when_raw_clearly_negative(
    tmp_path, monkeypatch
):
    """
    Invert verification should be conservative:
    - even if inverted is slightly better, we DO NOT pick it unless raw is clearly negative.
    """
    from src.time_series_model.diagnostics import feature_group_search as fgs

    base_dir = tmp_path / "base_strategy"
    base_dir.mkdir(parents=True)
    (base_dir / "features.yaml").write_text(
        "name: base\nfeature_pipeline:\n  requested_features: []\n", encoding="utf-8"
    )

    cfg = fgs.SearchConfig(
        base_strategy_dir=base_dir,
        timeframe="240T",
        symbol="BTCUSDT",
        start_date="2023-01-01",
        end_date="2025-10-31",
        test_size=0.3,
        seeds=[1, 2, 3, 4, 5],
        output_dir=tmp_path / "out",
        deterministic=True,
        no_docker=True,
    )

    groups = {"G": ["trend_r2_50_f"]}

    def _stub_run_seed_sweep_for_strategy(*, strategy_dir, cfg, run_id):
        # Baseline
        if run_id == "baseline":
            return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=1.0)

        # Raw: slightly positive (not "clearly negative")
        if run_id.startswith("prefilter_add_G") and run_id.endswith("__halving_s1"):
            if run_id.endswith("__inv"):
                return pd.DataFrame(), _fake_summary(
                    strategy=strategy_dir.name, score=0.20
                )
            return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=0.10)

        # Full budget stage (also not negative)
        if run_id.startswith("prefilter_add_G") and run_id.endswith("__halving_s5"):
            if run_id.endswith("__inv"):
                return pd.DataFrame(), _fake_summary(
                    strategy=strategy_dir.name, score=0.30
                )
            return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=0.10)

        return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=0.0)

    monkeypatch.setattr(
        fgs, "run_seed_sweep_for_strategy", _stub_run_seed_sweep_for_strategy
    )

    res = fgs.pipeline_sh_beam_sffs(
        cfg=cfg,
        base_features=[],
        groups=groups,
        max_steps=1,
        objective="Sharpe_mean",
        min_trades=10,
        stages=[1, 5],
        top_fraction=1.0,
        min_survivors=1,
        target_survivors=1,
        beam_width=1,
        sffs_max_backward_steps=1,
        invert_candidates=["G"],
    )

    # Should NOT pick inverted, hence no invert_by_group entry.
    assert res["prefilter"]["invert_by_group"] == {}


def test_pipeline_invert_is_picked_when_raw_clearly_negative_and_improves(
    tmp_path, monkeypatch
):
    """
    If raw is clearly negative and inverted improves enough, we should pick inverted,
    and propagate invert into final_invert_features.
    """
    from src.time_series_model.diagnostics import feature_group_search as fgs

    base_dir = tmp_path / "base_strategy"
    base_dir.mkdir(parents=True)
    (base_dir / "features.yaml").write_text(
        "name: base\nfeature_pipeline:\n  requested_features: []\n", encoding="utf-8"
    )

    cfg = fgs.SearchConfig(
        base_strategy_dir=base_dir,
        timeframe="240T",
        symbol="BTCUSDT",
        start_date="2023-01-01",
        end_date="2025-10-31",
        test_size=0.3,
        seeds=[1, 2, 3, 4, 5],
        output_dir=tmp_path / "out",
        deterministic=True,
        no_docker=True,
    )

    groups = {"G": ["trend_r2_50_f"]}

    def _stub_run_seed_sweep_for_strategy(*, strategy_dir, cfg, run_id):
        if run_id == "baseline":
            return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=0.0)
        # Raw: clearly negative
        if run_id.startswith("prefilter_add_G") and run_id.endswith("__halving_s1"):
            if run_id.endswith("__inv"):
                return pd.DataFrame(), _fake_summary(
                    strategy=strategy_dir.name, score=0.10
                )
            return pd.DataFrame(), _fake_summary(
                strategy=strategy_dir.name, score=-0.50
            )
        # Beam/prune can be no-op; return positive so pipeline completes
        if run_id.startswith("beam_step1_sel_"):
            return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=0.10)
        if run_id.startswith("prune_init__"):
            return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=0.10)
        return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=0.10)

    monkeypatch.setattr(
        fgs, "run_seed_sweep_for_strategy", _stub_run_seed_sweep_for_strategy
    )

    res = fgs.pipeline_sh_beam_sffs(
        cfg=cfg,
        base_features=[],
        groups=groups,
        max_steps=1,
        objective="Sharpe_mean",
        min_trades=10,
        stages=[1],
        top_fraction=1.0,
        min_survivors=1,
        target_survivors=1,
        beam_width=1,
        sffs_max_backward_steps=1,
        invert_candidates=["G"],
    )

    assert res["prefilter"]["invert_by_group"] == {"G": ["trend_r2_50_f"]}
    assert "trend_r2_50_f" in (res.get("final_invert_features") or [])
