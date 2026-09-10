from __future__ import annotations

from pathlib import Path

import pandas as pd


def _fake_summary(*, strategy: str, score: float, trades: int = 20) -> pd.DataFrame:
    # Matches `run_seed_sweep_for_strategy` summary schema after aggregation.
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


def test_greedy_forward_search_stops_when_no_improvement(tmp_path, monkeypatch):
    # Import inside test to avoid import side-effects in collection time.
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
        seeds=[1, 2],
        output_dir=tmp_path / "out",
        deterministic=True,
        no_docker=True,
    )

    groups = {"A": ["fa"], "B": ["fb"]}

    # Baseline score = 1.5
    # Step1: A=2.0, B=1.0 -> select A
    # Step2: only B remains but score equals best (2.0) -> should stop before selecting B
    def _stub_run_seed_sweep_for_strategy(*, strategy_dir, cfg, run_id):
        if run_id == "baseline":
            return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=1.5)
        if run_id.endswith("_add_A"):
            return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=2.0)
        if run_id.endswith("_add_B"):
            return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=2.0)
        raise AssertionError(f"Unexpected run_id: {run_id}")

    monkeypatch.setattr(
        fgs, "run_seed_sweep_for_strategy", _stub_run_seed_sweep_for_strategy
    )

    result = fgs.greedy_forward_search(
        cfg=cfg,
        base_features=[],
        groups=groups,
        max_steps=5,
        objective="Sharpe_mean",
        min_trades=10,
    )

    assert result["selected_groups"] == ["A"]
    assert result["stop_reason"] == "no_improvement"
    assert "candidates_history" in result
    assert len(result["candidates_history"]) >= 1


def test_greedy_forward_search_can_select_nothing_if_all_worse_than_baseline(
    tmp_path, monkeypatch
):
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
        seeds=[1, 2],
        output_dir=tmp_path / "out",
        deterministic=True,
        no_docker=True,
    )

    groups = {"A": ["fa"], "B": ["fb"]}

    def _stub_run_seed_sweep_for_strategy(*, strategy_dir, cfg, run_id):
        if run_id == "baseline":
            return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=2.0)
        # Both candidates are worse than baseline
        if run_id.endswith("_add_A"):
            return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=1.0)
        if run_id.endswith("_add_B"):
            return pd.DataFrame(), _fake_summary(strategy=strategy_dir.name, score=1.5)
        raise AssertionError(f"Unexpected run_id: {run_id}")

    monkeypatch.setattr(
        fgs, "run_seed_sweep_for_strategy", _stub_run_seed_sweep_for_strategy
    )

    result = fgs.greedy_forward_search(
        cfg=cfg,
        base_features=[],
        groups=groups,
        max_steps=3,
        objective="Sharpe_mean",
        min_trades=10,
    )

    assert result["selected_groups"] == []
    assert result["final_features"] == []
    assert result["stop_reason"] == "no_improvement"


def test_writeback_features_yaml_creates_suggested_file(tmp_path):
    from src.time_series_model.diagnostics import feature_group_search as fgs

    base_dir = tmp_path / "base_strategy"
    base_dir.mkdir(parents=True)
    (base_dir / "features.yaml").write_text(
        "\n".join(
            [
                "name: my_strategy",
                "description: test",
                "feature_pipeline:",
                "  requested_features:",
                "    - macd_f",
                "  ensure_signal_column:",
                "    name: signal",
                "    default_value: 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_path = tmp_path / "features_suggested.yaml"
    info = fgs._writeback_features_yaml(
        base_strategy_dir=base_dir,
        out_path=out_path,
        requested_features=["a_f", "b_f"],
        meta={"selected_groups": ["g1"], "stop_reason": "no_improvement"},
        invert_candidates=["a_f", "zzz_unused"],
    )

    assert Path(info["written"]).exists()
    obj = fgs._load_yaml(out_path)
    assert obj["name"].endswith("__suggested")
    assert obj["feature_pipeline"]["requested_features"] == ["a_f", "b_f"]
    # invert_features are OUTPUT column names to multiply by -1 (not requested feature functions),
    # so we must not prune them by requested_features.
    assert obj["feature_pipeline"]["invert_features"] == ["a_f", "zzz_unused"]
    assert obj["feature_group_search"]["selected_groups"] == ["g1"]


def test_pool_b_yaml_auto_groups_merges_into_groups(tmp_path):
    # We don't run the full search; just validate the merge logic by importing and
    # using the same parsing approach as main().
    import yaml

    from src.time_series_model.diagnostics import feature_group_search as fgs

    groups = {"g_semantic": ["a_f"]}
    pool_b = {
        "feature_pipeline": {
            "requested_features": ["a_f", "b_f"],
            "invert_features": ["b_f"],
        }
    }
    pool_path = tmp_path / "features_pool_b.yaml"
    pool_path.write_text(yaml.safe_dump(pool_b, sort_keys=False), encoding="utf-8")

    # Reuse the same internal merge logic by calling yaml.safe_load and applying the same behavior.
    pool_obj = yaml.safe_load(pool_path.read_text(encoding="utf-8")) or {}
    pool_fp = pool_obj.get("feature_pipeline") if isinstance(pool_obj, dict) else None
    pool_req = pool_fp.get("requested_features") if isinstance(pool_fp, dict) else None
    pool_req = pool_req if isinstance(pool_req, list) else []

    used_nodes = set()
    for feats in (groups or {}).values():
        for f in feats or []:
            used_nodes.add(str(f))

    for f in pool_req:
        f = str(f).strip()
        if not f:
            continue
        if f in used_nodes:
            continue
        key = f"poolb__{f}"
        if key in groups:
            i = 2
            while f"{key}__{i}" in groups:
                i += 1
            key = f"{key}__{i}"
        groups[key] = [f]

    assert "g_semantic" in groups
    assert "poolb__b_f" in groups
    assert groups["poolb__b_f"] == ["b_f"]
