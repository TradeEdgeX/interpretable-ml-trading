import pandas as pd

from src.features.loader.strategy_feature_loader import StrategyFeatureLoader


def test_load_features_auto_materializes_feature_store(tmp_path) -> None:
    # Minimal OHLCV frame for a single month
    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "open": [1, 2, 3, 4, 5],
            "high": [2, 3, 4, 5, 6],
            "low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5],
            "volume": [10, 11, 12, 13, 14],
            "_symbol": ["AAA"] * 5,
            "symbol": ["AAA"] * 5,
        },
        index=idx,
    )

    loader = StrategyFeatureLoader()
    out = loader.load_features_from_requested(
        df,
        requested_features=["atr_f"],  # exists in feature_dependencies.yaml
        fit=True,
        feature_store_dir=str(tmp_path),
        feature_store_layer="unit_test_layer",
        feature_store_symbol="AAA",
        feature_store_timeframe="1D",
    )
    assert "atr" in out.columns

    # Monthly partition should be materialized
    expected = tmp_path / "unit_test_layer" / "AAA" / "1D" / "2025-01.parquet"
    assert expected.exists()


def test_feature_store_stale_when_cache_version_changes(tmp_path) -> None:
    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "open": [1, 2, 3, 4, 5],
            "high": [2, 3, 4, 5, 6],
            "low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5],
            "volume": [10, 11, 12, 13, 14],
            "_symbol": ["AAA"] * 5,
            "symbol": ["AAA"] * 5,
        },
        index=idx,
    )
    loader = StrategyFeatureLoader()
    # First run materializes with current cache_version (e.g., v6)
    _ = loader.load_features_from_requested(
        df,
        requested_features=["atr_f"],
        fit=True,
        feature_store_dir=str(tmp_path),
        feature_store_layer="unit_test_layer",
        feature_store_symbol="AAA",
        feature_store_timeframe="1D",
    )

    # Simulate cache version bump
    loader.computer.cache_version = "v999"
    _ = loader.load_features_from_requested(
        df,
        requested_features=["atr_f"],
        fit=True,
        feature_store_dir=str(tmp_path),
        feature_store_layer="unit_test_layer",
        feature_store_symbol="AAA",
        feature_store_timeframe="1D",
    )

    meta_path = tmp_path / "unit_test_layer" / "AAA" / "1D" / "2025-01.meta.json"
    meta = meta_path.read_text(encoding="utf-8")
    assert "v999" in meta
