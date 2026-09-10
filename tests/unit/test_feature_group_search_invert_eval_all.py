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
                "CV_mean": float(score),
                "corr_mean": 0.0,
            }
        ]
    )


def test_invert_eval_all_picks_inverted_when_better(tmp_path, monkeypatch):
    """
    Ensure invert-eval=all performs raw vs inverted comparison and picks inverted when it scores higher.

    We simulate a Pool-B inverted output column as a singleton group (poolb_invcol__xxx -> [xxx]).
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
        end_date="2025-12-31",
        test_size=0.3,
        seeds=[1, 2],
        output_dir=tmp_path / "out",
        deterministic=True,
        no_docker=True,
        invert_eval="all",
    )

    # A singleton group adding an OUTPUT COLUMN (as used by semantic singleton expansion / our poolb_invcol__*)
    groups = {"poolb_invcol__foo": ["foo"]}

    def _stub_run_seed_sweep_for_strategy(*, strategy_dir, cfg, run_id):
        # raw vs inv distinguished by run_id suffix
        if run_id.endswith("__inv"):
            return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=1.2)
        return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=1.0)

    monkeypatch.setattr(
        fgs, "run_seed_sweep_for_strategy", _stub_run_seed_sweep_for_strategy
    )

    pre = fgs.successive_halving_prefilter(
        cfg=cfg,
        base_features=[],
        groups=groups,
        objective="CV_mean",
        min_trades=10,
        stages=[1, 2],
        top_fraction=1.0,
        min_survivors=1,
        target_survivors=1,
        invert_candidates=["foo"],
    )

    inv_map = pre.get("invert_by_group") or {}
    assert "poolb_invcol__foo" in inv_map
    assert inv_map["poolb_invcol__foo"] == ["foo"]
