import pandas as pd


def test_feature_computer_is_sequential_only():
    from src.features.loader.feature_computer import FeatureComputer

    comp = FeatureComputer(cache_dir=None, use_disk_cache=False, use_memory_cache=False)
    assert comp.executor is None
    assert comp.parallel_backend == "sequential"
    assert comp.max_workers == 1


def test_monthly_cache_key_changes_with_df_signature():
    from src.features.loader.feature_computer import FeatureComputer

    comp = FeatureComputer(cache_dir="cache/features", use_monthly_cache=True)
    params = {"window": 20}
    feature_info = {"output_columns": ["rsi"]}

    # Same feature/month/params but different df_sig must produce different keys
    k1 = comp._get_monthly_cache_key(
        "rsi_f", "2024-01", params, feature_info, df_sig="SIG_A"
    )
    k2 = comp._get_monthly_cache_key(
        "rsi_f", "2024-01", params, feature_info, df_sig="SIG_B"
    )
    assert k1 != k2


def test_df_signature_is_stable_for_same_df():
    from src.features.loader.feature_computer import FeatureComputer

    comp = FeatureComputer(cache_dir=None, use_disk_cache=False, use_memory_cache=False)
    idx = pd.date_range("2024-01-01", periods=5, freq="1h")
    df = pd.DataFrame(
        {"close": [1, 2, 3, 4, 5], "volume": [10, 11, 12, 13, 14]}, index=idx
    )
    s1 = comp._get_df_signature(df)
    s2 = comp._get_df_signature(df)
    assert s1 == s2
