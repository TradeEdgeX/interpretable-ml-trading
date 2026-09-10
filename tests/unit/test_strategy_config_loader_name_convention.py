def test_strategy_config_loader_name_is_dir_name_even_if_yaml_declares_other(
    tmp_path, capsys
):
    from src.time_series_model.strategy_config.loader import StrategyConfigLoader

    cfg_dir = tmp_path / "my_strategy_dir"
    cfg_dir.mkdir(parents=True)

    # Deliberately mismatch YAML name vs directory name.
    (cfg_dir / "features.yaml").write_text(
        "\n".join(
            [
                "name: declared_other_name",
                "feature_pipeline:",
                "  requested_features: []",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (cfg_dir / "labels.yaml").write_text(
        "\n".join(
            [
                "target_column: y",
                "generator:",
                "  module: src.time_series_model.strategies.labels.bpc_label",
                "  function: compute_bpc_label",
                "  params: {}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (cfg_dir / "model.yaml").write_text(
        "\n".join(
            [
                "trainer:",
                "  module: src.time_series_model.models.lightgbm_trainer",
                "  function: train_lightgbm_model",
                "  params: {}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = StrategyConfigLoader(cfg_dir).load()
    out = capsys.readouterr().out

    assert cfg.name == "my_strategy_dir"
    assert "Strategy name mismatch" in out
    assert cfg.meta.get("declared_name") == "declared_other_name"


def test_strategy_config_loader_strict_name_match_raises(tmp_path):
    from src.time_series_model.strategy_config.loader import StrategyConfigLoader

    cfg_dir = tmp_path / "my_strategy_dir"
    cfg_dir.mkdir(parents=True)

    (cfg_dir / "features.yaml").write_text(
        "name: declared_other_name\nfeature_pipeline:\n  requested_features: []\n",
        encoding="utf-8",
    )
    (cfg_dir / "labels.yaml").write_text(
        "\n".join(
            [
                "target_column: y",
                "generator:",
                "  module: src.time_series_model.strategies.labels.bpc_label",
                "  function: compute_bpc_label",
                "  params: {}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (cfg_dir / "model.yaml").write_text(
        "\n".join(
            [
                "trainer:",
                "  module: src.time_series_model.models.lightgbm_trainer",
                "  function: train_lightgbm_model",
                "  params: {}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        StrategyConfigLoader(cfg_dir, strict_name_match=True).load()
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Strategy name mismatch" in str(e)
