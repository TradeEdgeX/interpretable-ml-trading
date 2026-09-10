import pandas as pd
import yaml

from src.features.registry import register_feature
from src.features.loader.strategy_feature_loader import StrategyFeatureLoader


@register_feature("test_compute_atr_ratio_from_series", category="test")
def _test_compute_atr_ratio_from_series(
    *, atr: pd.Series, close: pd.Series
) -> pd.DataFrame:
    atr_s = pd.to_numeric(atr, errors="coerce").fillna(0.0).astype(float)
    cl = pd.to_numeric(close, errors="coerce").replace(0, pd.NA).astype(float)
    return (atr_s / cl).fillna(0.0).rename("atr_ratio").to_frame()


@register_feature("test_compute_atr_from_series", category="test")
def _test_compute_atr_from_series(
    *, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.DataFrame:
    # Very cheap ATR proxy (not a real ATR); good enough for output-trim tests.
    h = pd.to_numeric(high, errors="coerce").astype(float)
    l = pd.to_numeric(low, errors="coerce").astype(float)
    return (h - l).fillna(0.0).rename("atr").to_frame()


@register_feature("test_compute_semantic_scores_from_series", category="test")
def _test_compute_semantic_scores_from_series(*, close: pd.Series) -> pd.DataFrame:
    cl = pd.to_numeric(close, errors="coerce").fillna(0.0).astype(float)
    # Two output columns to simulate a semantic block.
    return pd.DataFrame(
        {
            "scene_a": (cl * 0.0 + 0.1),
            "scene_b": (cl * 0.0 + 0.9),
        },
        index=close.index,
    )


def _write_feature_deps(tmp_path) -> str:
    deps = {
        "features": {
            "atr_f": {
                "module": "test",
                "compute_func": "test_compute_atr_from_series",
                "dependencies": [],
                "required_columns": ["high", "low", "close"],
                "output_columns": ["atr"],
                "pass_full_df": False,
                "column_mappings": {"high": "high", "low": "low", "close": "close"},
            },
            "atr_ratio_f": {
                "module": "test",
                "compute_func": "test_compute_atr_ratio_from_series",
                "dependencies": ["atr_f"],
                "required_columns": ["atr", "close"],
                "output_columns": ["atr_ratio"],
                "pass_full_df": False,
                "column_mappings": {"atr": "atr", "close": "close"},
            },
            "semantic_scores_f": {
                "module": "test",
                "compute_func": "test_compute_semantic_scores_from_series",
                "dependencies": [],
                "required_columns": ["close"],
                "output_columns": ["scene_a", "scene_b"],
                "pass_full_df": False,
                "column_mappings": {"close": "close"},
            },
        }
    }
    p = tmp_path / "feature_deps_min.yaml"
    p.write_text(yaml.safe_dump(deps, sort_keys=False), encoding="utf-8")
    return str(p)


def test_loader_returns_only_requested_outputs_not_dependency_outputs(tmp_path):
    feature_deps_path = _write_feature_deps(tmp_path)
    loader = StrategyFeatureLoader(
        feature_deps_path=feature_deps_path,
        cache_dir=None,
        use_disk_cache=False,
        use_memory_cache=False,
        use_monthly_cache=False,
        normalization_contract_mode="warn",
    )

    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "open": [1, 1, 1, 1, 1],
            "high": [2, 2, 2, 2, 2],
            "low": [1, 1, 1, 1, 1],
            "close": [2, 2, 2, 2, 2],
            "volume": [10, 10, 10, 10, 10],
        },
        index=idx,
    )

    out = loader.load_features_from_requested(df, ["atr_ratio_f"], fit=True)
    assert "atr_ratio" in out.columns
    # Dependency output should NOT leak into final output columns
    assert "atr" not in out.columns
    # Base input columns should still be present
    assert "close" in out.columns
    assert "high" in out.columns


def test_loader_keeps_singleton_output_column_only(tmp_path):
    feature_deps_path = _write_feature_deps(tmp_path)
    loader = StrategyFeatureLoader(
        feature_deps_path=feature_deps_path,
        cache_dir=None,
        use_disk_cache=False,
        use_memory_cache=False,
        use_monthly_cache=False,
        normalization_contract_mode="warn",
    )

    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "open": [1, 1, 1, 1, 1],
            "high": [2, 2, 2, 2, 2],
            "low": [1, 1, 1, 1, 1],
            "close": [2, 2, 2, 2, 2],
            "volume": [10, 10, 10, 10, 10],
        },
        index=idx,
    )

    # Request only a single output column name (singleton mode).
    out = loader.load_features_from_requested(df, ["scene_a"], fit=True)
    assert "scene_a" in out.columns
    assert "scene_b" not in out.columns
