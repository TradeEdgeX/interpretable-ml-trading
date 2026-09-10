import pandas as pd

from src.features.loader.strategy_feature_loader import StrategyFeatureLoader


def test_feature_store_materialize_multi_symbol(tmp_path) -> None:
    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    base = {
        "open": [1, 2, 3, 4, 5],
        "high": [2, 3, 4, 5, 6],
        "low": [0.5, 1.5, 2.5, 3.5, 4.5],
        "close": [1.5, 2.5, 3.5, 4.5, 5.5],
        "volume": [10, 11, 12, 13, 14],
    }

    loader = StrategyFeatureLoader()
    for sym in ("AAA", "BBB"):
        df = pd.DataFrame(
            {**base, "_symbol": [sym] * 5, "symbol": [sym] * 5},
            index=idx,
        )
        out = loader.load_features_from_requested(
            df,
            requested_features=["atr_f"],
            fit=True,
            feature_store_dir=str(tmp_path),
            feature_store_layer="unit_test_layer",
            feature_store_symbol=sym,
            feature_store_timeframe="1D",
        )
        assert "atr" in out.columns

    # Both symbols should have their own monthly partition.
    for sym in ("AAA", "BBB"):
        expected = tmp_path / "unit_test_layer" / sym / "1D" / "2025-01.parquet"
        assert expected.exists()
