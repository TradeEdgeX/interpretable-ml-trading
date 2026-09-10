import pandas as pd

from src.features.loader.feature_computer import FeatureComputer


def test_monthly_cache_hits_even_with_warmup_window(tmp_path) -> None:
    # Two windows:
    # - window1: includes previous month warmup + Jan
    # - window2: includes MORE warmup + same Jan
    # Expect: Jan month cache should hit on the 2nd run.
    idx_dec = pd.date_range("2024-12-25", periods=3, freq="D")
    idx_jan = pd.date_range("2025-01-01", periods=5, freq="D")

    def _df(idx):
        return pd.DataFrame(
            {
                "open": range(len(idx)),
                "high": range(len(idx)),
                "low": range(len(idx)),
                "close": range(len(idx)),
                "volume": [10] * len(idx),
                "_symbol": ["AAA"] * len(idx),
                "symbol": ["AAA"] * len(idx),
            },
            index=idx,
        )

    df_window1 = pd.concat([_df(idx_dec), _df(idx_jan)]).sort_index()
    idx_dec_more = pd.date_range("2024-12-20", periods=6, freq="D")
    df_window2 = pd.concat([_df(idx_dec_more), _df(idx_jan)]).sort_index()

    fc = FeatureComputer(
        cache_dir=str(tmp_path), use_monthly_cache=True, use_memory_cache=False
    )

    features_cfg = {
        "atr_f": {
            "compute_func": "compute_atr_from_series",
            "dependencies": [],
            "required_columns": ["high", "low", "close"],
            "output_columns": ["atr"],
            "column_mappings": {"high": "high", "low": "low", "close": "close"},
            "compute_params": {"period": 14},
            "pass_full_df": False,
        }
    }

    _ = fc.compute_features_parallel(df_window1, features_cfg, ["atr_f"], fit=True)
    _ = fc.compute_features_parallel(df_window2, features_cfg, ["atr_f"], fit=True)
    stats = fc.drain_debug_stats()

    # At least one month should be a cache hit on the second run.
    assert stats["cache_hits"]["monthly"] >= 1
