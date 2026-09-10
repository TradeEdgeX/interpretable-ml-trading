def test_groups_source_auto_uses_global_config_when_present(monkeypatch, tmp_path):
    from src.time_series_model.diagnostics import feature_group_search as fgs

    # Create a fake repo root with config/feature_groups.yaml
    repo = tmp_path / "repo"
    (repo / "config").mkdir(parents=True)
    (repo / "config" / "feature_groups.yaml").write_text(
        "groups:\n  g1:\n    - a_f\n", encoding="utf-8"
    )

    monkeypatch.chdir(repo)

    groups, src, auto = fgs._load_groups_with_source(
        strategy_dir_name="some_strategy",
        groups_json=None,
        groups_yaml=None,
    )
    assert groups == {"g1": ["a_f"]}
    assert src == "groups_yaml:auto:config/feature_groups.yaml"
    assert auto is True


def test_groups_source_falls_back_to_default_groups_when_no_files(
    monkeypatch, tmp_path
):
    from src.time_series_model.diagnostics import feature_group_search as fgs

    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    monkeypatch.chdir(repo)

    groups, src, auto = fgs._load_groups_with_source(
        strategy_dir_name="some_strategy",
        groups_json=None,
        groups_yaml=None,
    )
    assert isinstance(groups, dict) and len(groups) > 0
    assert src == "default_groups"
    assert auto is False


def test_groups_source_auto_prefers_strategy_specific_yaml_over_global(
    monkeypatch, tmp_path
):
    from src.time_series_model.diagnostics import feature_group_search as fgs

    repo = tmp_path / "repo"
    (repo / "config").mkdir(parents=True)
    (repo / "config" / "feature_groups.yaml").write_text(
        "groups:\n  g_global:\n    - a_f\n", encoding="utf-8"
    )
    (repo / "config" / "feature_groups_trend_following_semantic.yaml").write_text(
        "groups:\n  g_strategy:\n    - b_f\n", encoding="utf-8"
    )
    monkeypatch.chdir(repo)

    groups, src, auto = fgs._load_groups_with_source(
        strategy_dir_name="trend_following",
        groups_json=None,
        groups_yaml=None,
    )
    assert groups == {"g_strategy": ["b_f"]}
    assert src == "groups_yaml:auto:config/feature_groups_trend_following_semantic.yaml"
    assert auto is True
