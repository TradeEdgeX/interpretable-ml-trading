"""Unit tests for the public research CLI."""

from click.testing import CliRunner

from cli.main import cli, get_project_root


class TestCLI:
    def test_cli_help(self):
        result = CliRunner().invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "features" in result.output
        assert "data" in result.output
        assert "feature-store" in result.output
        assert "research" in result.output
        assert "train" not in result.output
        assert "nnmultihead" not in result.output
        assert "lab" not in result.output
        assert "console" not in result.output

    def test_cli_version(self):
        result = CliRunner().invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestFeaturesCommands:
    def test_features_help(self):
        result = CliRunner().invoke(cli, ["features", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "count" in result.output

    def test_features_list_help(self):
        result = CliRunner().invoke(cli, ["features", "list", "--help"])
        assert result.exit_code == 0
        assert "--all" in result.output
        assert "--category" in result.output
        assert "--search" in result.output


class TestDataCommands:
    def test_data_help(self):
        result = CliRunner().invoke(cli, ["data", "--help"])
        assert result.exit_code == 0
        assert "download" in result.output
        assert "convert" in result.output
        assert "pipeline" in result.output

    def test_data_download_help(self):
        result = CliRunner().invoke(cli, ["data", "download", "--help"])
        assert result.exit_code == 0
        assert "--symbols" in result.output
        assert "--start-year" in result.output


class TestResearchCommands:
    def test_research_harness_ma_cross_runs(self):
        result = CliRunner().invoke(cli, ["research", "harness", "ma_cross"])
        assert result.exit_code == 0
        assert "event_backtest" in result.output

    def test_features_list_search_runs(self):
        result = CliRunner().invoke(cli, ["features", "list", "--search", "ema_50"])
        assert result.exit_code == 0
        assert "ema" in result.output.lower()

    def test_research_help(self):
        result = CliRunner().invoke(cli, ["research", "--help"])
        assert result.exit_code == 0
        for name in (
            "index",
            "harness",
            "close",
            "run",
            "init",
            "replay",
            "scorecard",
        ):
            assert name in result.output


class TestFeatureStoreCommands:
    def test_feature_store_help(self):
        result = CliRunner().invoke(cli, ["feature-store", "--help"])
        assert result.exit_code == 0
        assert "build" in result.output


class TestProjectRoot:
    def test_get_project_root(self):
        root = get_project_root()
        assert root.exists()
        assert (root / "setup.py").exists() or (root / "pyproject.toml").exists()
