from pathlib import Path


def test_export_features_yaml_writes_empty_yaml_when_no_qualified_factors(
    tmp_path: Path,
):
    from src.time_series_model.diagnostics.factor_ts_eval import export_features_yaml

    out_dir = tmp_path / "out"
    out_path = out_dir / "features_pool_b.yaml"

    p = export_features_yaml(
        qualified_factors={"positive": [], "negative": []},
        strategy_name="trend_following",
        symbol="BTCUSDT",
        output_dir=out_dir,
        strategy_config_path=None,
        error_factors=None,
        output_path=out_path,
        invert_features=[],
    )

    assert p == out_path
    assert out_path.exists()
    txt = out_path.read_text(encoding="utf-8")
    assert "requested_features" in txt
