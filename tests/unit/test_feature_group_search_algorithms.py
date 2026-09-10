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


def _mk_cfg(tmp_path, fgs):
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
    return cfg


def test_successive_halving_picks_best_survivor(tmp_path, monkeypatch):
    from src.time_series_model.diagnostics import feature_group_search as fgs

    cfg = _mk_cfg(tmp_path, fgs)
    groups = {"A": ["fa"], "B": ["fb"], "C": ["fc"], "D": ["fd"]}

    # Baseline = 1.0
    # Stage seeds=1: A=2.0, B=1.9, C=1.1, D=1.0  -> survivors A,B
    # Stage seeds=5: A=1.2, B=2.5 -> pick B
    def _stub_run_seed_sweep_for_strategy(*, strategy_dir, cfg, run_id):
        if run_id == "baseline":
            return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=1.0)
        if run_id.endswith("__halving_s1"):
            if "_add_A__" in run_id:
                return pd.DataFrame(), _fake_summary(
                    strategy=strategy_dir.name, score=2.0
                )
            if "_add_B__" in run_id:
                return pd.DataFrame(), _fake_summary(
                    strategy=strategy_dir.name, score=1.9
                )
            if "_add_C__" in run_id:
                return pd.DataFrame(), _fake_summary(
                    strategy=strategy_dir.name, score=1.1
                )
            if "_add_D__" in run_id:
                return pd.DataFrame(), _fake_summary(
                    strategy=strategy_dir.name, score=1.0
                )
        if run_id.endswith("__halving_s5"):
            if "_add_A__" in run_id:
                return pd.DataFrame(), _fake_summary(
                    strategy=strategy_dir.name, score=1.2
                )
            if "_add_B__" in run_id:
                return pd.DataFrame(), _fake_summary(
                    strategy=strategy_dir.name, score=2.5
                )
        raise AssertionError(f"Unexpected run_id: {run_id}")

    monkeypatch.setattr(
        fgs, "run_seed_sweep_for_strategy", _stub_run_seed_sweep_for_strategy
    )

    result = fgs.successive_halving_search(
        cfg=cfg,
        base_features=[],
        groups=groups,
        max_steps=1,
        objective="Sharpe_mean",
        min_trades=10,
        stages=[1, 5],
        top_fraction=0.5,
        min_survivors=2,
    )
    assert result["search_algo"] == "successive_halving"
    assert result["selected_groups"] == ["B"]


def test_beam_search_finds_synergy_path(tmp_path, monkeypatch):
    from src.time_series_model.diagnostics import feature_group_search as fgs

    cfg = _mk_cfg(tmp_path, fgs)
    groups = {"A": ["fa"], "B": ["fb"], "C": ["fc"]}

    # Synergy: best is B+C (2.0) but greedy would pick A (1.5) then stop.
    score_map = {
        "A": 1.5,
        "B": 1.4,
        "C": 0.9,
        "B__C": 2.0,
        "A__B": 1.45,
        "A__C": 1.2,
        "A__B__C": 1.6,
    }

    def _stub_run_seed_sweep_for_strategy(*, strategy_dir, cfg, run_id):
        if run_id == "baseline":
            return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=1.0)
        if run_id.startswith("beam_step"):
            sig = run_id.split("beam_step", 1)[1]
            sel = sig.split("_sel_", 1)[1]
            score = score_map.get(sel)
            if score is None:
                score = 0.0
            return pd.DataFrame(), _fake_summary(
                strategy=strategy_dir.name, score=score
            )
        raise AssertionError(f"Unexpected run_id: {run_id}")

    monkeypatch.setattr(
        fgs, "run_seed_sweep_for_strategy", _stub_run_seed_sweep_for_strategy
    )

    result = fgs.beam_search(
        cfg=cfg,
        base_features=[],
        groups=groups,
        max_steps=2,
        objective="Sharpe_mean",
        min_trades=10,
        beam_width=2,
    )
    assert result["search_algo"] == "beam"
    assert result["selected_groups"] == ["B", "C"]


def test_sffs_can_remove_redundant_group(tmp_path, monkeypatch):
    from src.time_series_model.diagnostics import feature_group_search as fgs

    cfg = _mk_cfg(tmp_path, fgs)
    groups = {"A": ["fa"], "B": ["fb"]}

    # Forward: add A -> 1.2, add B -> 1.25
    # Backward after B: removing A yields 1.3 (improves), so final should be ["B"]
    def _stub_run_seed_sweep_for_strategy(*, strategy_dir, cfg, run_id):
        if run_id == "baseline":
            return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=1.0)
        if "sffs_step1_fwd_sel_A" in run_id:
            return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=1.2)
        if "sffs_step2_fwd_sel_A__B" in run_id:
            return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=1.25)
        if "sffs_step2_bwd_sel_B__rm_A" in run_id:
            return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=1.3)
        # Fallback: any other eval is neutral/invalid
        return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=0.0)

    monkeypatch.setattr(
        fgs, "run_seed_sweep_for_strategy", _stub_run_seed_sweep_for_strategy
    )

    result = fgs.sffs_search(
        cfg=cfg,
        base_features=[],
        groups=groups,
        max_steps=3,
        objective="Sharpe_mean",
        min_trades=10,
        max_backward_per_step=2,
    )
    assert result["search_algo"] == "sffs"
    assert result["selected_groups"] == ["B"]
