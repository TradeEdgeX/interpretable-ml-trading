import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.research.stat_kernels.ic import ic_decay_rows, shift_target_by_horizon
from src.research.stat_kernels.ic_prune import (
    build_monotone_payload,
    build_monotone_payload_columns,
    build_compute_request,
    invert_columns_for_nodes,
    run_ic_prune,
    screen_features,
    trim_columns,
)


def test_shift_target_by_horizon():
    df = pd.DataFrame({"forward_rr": [1.0, 2.0, 3.0, 4.0]})
    y = shift_target_by_horizon(df["forward_rr"], 2, df)
    assert y.iloc[0] == 3.0
    assert pd.isna(y.iloc[-1])


def test_shift_target_by_horizon_per_symbol():
    df = pd.DataFrame(
        {
            "_symbol": ["A", "A", "A", "B", "B", "B"],
            "forward_rr": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        }
    )
    y = shift_target_by_horizon(df["forward_rr"], 2, df)
    assert y.iloc[0] == 3.0
    assert y.iloc[3] == 30.0


def test_ic_decay_rows_with_shift():
    df = pd.DataFrame(
        {
            "feat": [1.0, 2.0, 3.0, 4.0, 5.0] * 25,
            "forward_rr": list(range(125)),
        }
    )
    rows = ic_decay_rows(df, ["feat"], [1, 3], "forward_rr")
    h3 = [r for r in rows if r["horizon"] == 3][0]
    assert h3["shifted"] is True
    assert "shift" in h3["target_col"]


def _synthetic_holdout_df(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    forward_rr = rng.normal(0, 1, size=n)
    pos_feat = forward_rr + rng.normal(0, 0.05, size=n)
    neg_feat = -forward_rr + rng.normal(0, 0.05, size=n)
    noise = rng.normal(0, 1, size=n)
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2025-10-01", periods=n, freq="120min"),
            "_symbol": ["BTCUSDT"] * n,
            "forward_rr": forward_rr,
            "pos_col": pos_feat,
            "neg_col": neg_feat,
            "noise_col": noise,
        }
    )


def test_screen_features_requires_forward_rr():
    df = pd.DataFrame({"datetime": pd.date_range("2025-10-01", periods=5, freq="h")})
    with pytest.raises(KeyError, match="forward_rr"):
        screen_features(
            df,
            holdout_start="2025-10-01",
            holdout_end="2025-10-02",
            horizons=[1],
            min_ic=0.01,
            max_lag=5,
            min_n=50,
        )


def test_screen_features_skips_forward_rr_h_and_target() -> None:
    n = 300
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-10-01", periods=n, freq="h"),
            "forward_rr_h3": rng.normal(0, 1, n),
            "label": rng.normal(0, 1, n),
            "feat_a": rng.normal(0, 1, n),
        }
    )
    rows, _, _ = screen_features(
        df,
        holdout_start="2025-10-01",
        holdout_end="2026-04-01",
        horizons=[1, 3],
        min_ic=0.01,
        max_lag=3,
        min_n=50,
        target="forward_rr_h3",
    )
    names = {r["feature"] for r in rows}
    assert "forward_rr_h3" not in names
    assert "label" not in names
    assert "feat_a" in names


def test_screen_features_records_sign(tmp_path: Path) -> None:
    df = _synthetic_holdout_df()
    rows, nodes, requested = screen_features(
        df,
        holdout_start="2025-10-01",
        holdout_end="2026-04-01",
        horizons=[1, 2, 3],
        min_ic=0.05,
        max_lag=3,
        min_n=100,
        feature_deps={
            "pos_f": {"output_columns": ["pos_col"]},
            "neg_f": {"output_columns": ["neg_col"]},
        },
    )
    assert "pos_f" in requested
    assert "neg_f" in requested
    pos_row = next(r for r in rows if r["feature"] == "pos_col")
    neg_row = next(r for r in rows if r["feature"] == "neg_col")
    assert pos_row["ic_sign"] == "+"
    assert neg_row["ic_sign"] == "-"
    neg_node = next(n for n in nodes if n["node"] == "neg_f")
    assert neg_node["ic_sign"] == "-"


def test_screen_features_allowed_best_lags() -> None:
    df = _synthetic_holdout_df()
    rows_all, _, _ = screen_features(
        df,
        holdout_start="2025-10-01",
        holdout_end="2026-04-01",
        horizons=[1, 2, 3],
        min_ic=0.05,
        max_lag=3,
        min_n=100,
    )
    rows_lag2, _, _ = screen_features(
        df,
        holdout_start="2025-10-01",
        holdout_end="2026-04-01",
        horizons=[1, 2, 3],
        min_ic=0.05,
        max_lag=3,
        min_n=100,
        allowed_best_lags=frozenset({2}),
    )
    assert len(rows_all) >= len(rows_lag2)
    assert all(r["best_lag"] == 2 for r in rows_lag2)


def test_screen_features_reject_peak_at() -> None:
    df = _synthetic_holdout_df()
    rows, _, _ = screen_features(
        df,
        holdout_start="2025-10-01",
        holdout_end="2026-04-01",
        horizons=[1, 2, 3],
        min_ic=0.05,
        max_lag=3,
        min_n=100,
        reject_peak_at=3,
    )
    assert all(r["best_lag"] != 3 for r in rows)


def test_invert_columns_for_nodes():
    summaries = [
        {"node": "a_f", "via_column": "a", "rank_ic": 0.1},
        {"node": "b_f", "via_column": "b", "rank_ic": -0.2},
    ]
    inv = invert_columns_for_nodes(summaries, ["a_f", "b_f"])
    assert inv == ["b"]


def test_run_ic_prune_invert_mode_none(tmp_path: Path) -> None:
    (tmp_path / "deps.yaml").write_text(
        "features:\n  pos_f:\n    output_columns: [pos_col]\n  neg_f:\n    output_columns: [neg_col]\n"
    )
    pq = tmp_path / "features.parquet"
    _synthetic_holdout_df().to_parquet(pq)
    out = tmp_path / "out"
    run_ic_prune(
        parquet=pq,
        output_dir=out,
        min_ic=0.05,
        min_n=100,
        invert_mode="none",
        project_root=tmp_path,
        feature_deps_path=tmp_path / "deps.yaml",
    )
    payload = __import__("json").loads((out / "ic_prune_holdout.json").read_text())
    assert "invert_features" not in payload
    assert any(n["ic_sign"] == "-" for n in payload["nodes"])


def test_run_ic_prune_invert_mode_auto(tmp_path: Path) -> None:
    (tmp_path / "deps.yaml").write_text(
        "features:\n  pos_f:\n    output_columns: [pos_col]\n  neg_f:\n    output_columns: [neg_col]\n"
    )
    pq = tmp_path / "features.parquet"
    _synthetic_holdout_df().to_parquet(pq)
    out = tmp_path / "out"
    run_ic_prune(
        parquet=pq,
        output_dir=out,
        min_ic=0.05,
        min_n=100,
        invert_mode="auto",
        writeback_mode="nodes",
        top_n_nodes=2,
        always_include=[],
        project_root=tmp_path,
        feature_deps_path=tmp_path / "deps.yaml",
    )
    payload = __import__("json").loads((out / "ic_prune_holdout.json").read_text())
    assert "neg_col" in payload.get("invert_features", [])
    assert "pos_col" not in payload.get("invert_features", [])


def test_writeback_preserves_keys_and_no_dup_description(tmp_path: Path) -> None:
    import yaml

    (tmp_path / "deps.yaml").write_text(
        "features:\n  pos_f:\n    output_columns: [pos_col]\n  neg_f:\n    output_columns: [neg_col]\n"
    )
    feat_yaml = tmp_path / "features.yaml"
    feat_yaml.write_text(
        "# top comment\n"
        "name: demo\n"
        "description: |\n  old desc\n"
        "feature_pipeline:\n"
        "  exclude_columns:\n    - atr\n"
        "  ensure_signal_column:\n    name: signal\n    default_value: 0\n"
        "  requested_features:\n    - stale_f\n"
    )
    pq = tmp_path / "features.parquet"
    _synthetic_holdout_df().to_parquet(pq)

    for _ in range(2):
        run_ic_prune(
            parquet=pq,
            output_dir=tmp_path / "out",
            min_ic=0.05,
            min_n=100,
            write_features_yaml=feat_yaml,
            writeback_mode="nodes",
            always_include=[],
            project_root=tmp_path,
            feature_deps_path=tmp_path / "deps.yaml",
        )

    raw = feat_yaml.read_text()
    assert raw.count("description:") == 1
    assert raw.startswith("# top comment")
    doc = yaml.safe_load(raw)
    assert doc["name"] == "demo"
    fp = doc["feature_pipeline"]
    assert fp["exclude_columns"] == ["atr"]
    assert fp["ensure_signal_column"] == {"name": "signal", "default_value": 0}
    assert "stale_f" not in fp["requested_features"]
    assert "pos_f" in fp["requested_features"]
    assert "invert_features" not in fp  # invert_mode none


def test_build_monotone_payload_order():
    deps = {
        "pos_f": {"output_columns": ["p1", "p2"]},
        "neg_f": {"output_columns": ["n1"]},
    }
    summaries = [
        {"node": "pos_f", "rank_ic": 0.2},
        {"node": "neg_f", "rank_ic": -0.3},
    ]
    payload = build_monotone_payload(["pos_f", "neg_f"], summaries, feature_deps=deps)
    assert payload["monotone_constraints"] == [1, 1, -1]
    assert [e["column"] for e in payload["expanded_columns"]] == ["p1", "p2", "n1"]


def test_trim_columns_respects_top_n():
    rows = [
        {
            "feature": f"f{i}",
            "rank_ic": 1.0 - i * 0.1,
            "best_lag": 1,
            "ic_sign": "+",
            "n": 100,
        }
        for i in range(5)
    ]
    selected = trim_columns(
        rows, top_n=2, pool_columns=None, always_include_columns=None
    )
    assert [r["feature"] for r in selected] == ["f0", "f1"]


def test_build_compute_request_mixed():
    req = build_compute_request(["pos_col", "neg_col"], ["atr_f"])
    assert req == ["atr_f", "pos_col", "neg_col"]


def test_run_ic_prune_column_writeback(tmp_path: Path) -> None:
    import yaml

    (tmp_path / "deps.yaml").write_text(
        "features:\n"
        "  pos_f:\n    output_columns: [pos_col]\n"
        "  neg_f:\n    output_columns: [neg_col, neg_extra]\n"
        "  atr_f:\n    output_columns: [atr]\n"
    )
    pq = tmp_path / "features.parquet"
    _synthetic_holdout_df().to_parquet(pq)
    feat_yaml = tmp_path / "features.yaml"
    feat_yaml.write_text(
        "name: demo\nfeature_pipeline:\n  exclude_columns: [atr]\n  requested_features: []\n"
    )
    archetype_yaml = tmp_path / "archetypes/model_features.yaml"
    out = tmp_path / "out"
    paths = run_ic_prune(
        parquet=pq,
        output_dir=out,
        min_ic=0.05,
        min_n=100,
        write_features_yaml=feat_yaml,
        writeback_mode="columns",
        top_n_columns=1,
        always_include=["atr_f"],
        write_model_features_yaml=archetype_yaml,
        project_root=tmp_path,
        feature_deps_path=tmp_path / "deps.yaml",
    )
    payload = __import__("json").loads((out / "ic_prune_holdout.json").read_text())
    assert payload["writeback_mode"] == "columns"
    assert payload["selected_columns"] == [payload["columns"][0]["feature"]]
    doc = yaml.safe_load(feat_yaml.read_text())
    assert doc["feature_pipeline"]["requested_features"][0] == "atr_f"
    assert (
        doc["feature_pipeline"]["requested_features"][1]
        == payload["selected_columns"][0]
    )
    arch = yaml.safe_load(archetype_yaml.read_text())
    assert len(arch["columns"]) == 1
    assert arch["columns"][0]["feature"] == payload["selected_columns"][0]
    assert paths["model_features_yaml"] == archetype_yaml


def test_build_monotone_payload_columns():
    rows = [
        {"feature": "a", "rank_ic": 0.2},
        {"feature": "b", "rank_ic": -0.1},
    ]
    payload = build_monotone_payload_columns(rows)
    assert payload["monotone_constraints"] == [1, -1]
