import pandas as pd


def test_align_to_base_index_handles_tz_mismatch():
    """
    Regression test:
    - monthly caches were historically stored with tz-naive DatetimeIndex
    - newer loaders may produce tz-aware (UTC) indices
    - a plain reindex() would yield ALL-NaN (no timestamp matches)
    """
    from src.features.loader.feature_computer import FeatureComputer

    base_index = pd.date_range("2023-01-01", periods=5, freq="4H", tz="UTC")
    cached_index = pd.date_range("2023-01-01", periods=5, freq="4H")  # tz-naive

    cached = pd.DataFrame({"macd": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=cached_index)

    fc = FeatureComputer(cache_dir=None, use_disk_cache=False, use_monthly_cache=False)
    aligned = fc._align_to_base_index("macd_f", cached, base_index)

    assert int(aligned["macd"].notna().sum()) == 5
