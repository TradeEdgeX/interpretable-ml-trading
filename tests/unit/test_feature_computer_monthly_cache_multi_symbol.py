import glob

import pandas as pd

from src.features.loader.feature_computer import FeatureComputer


def test_monthly_cache_separates_symbols(tmp_path) -> None:
    fc = FeatureComputer(
        cache_dir=str(tmp_path),
        use_monthly_cache=True,
        use_memory_cache=False,
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

    idx = pd.date_range("2025-01-01", periods=5, freq="D")

    def _df(sym: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open": [1, 2, 3, 4, 5],
                "high": [2, 3, 4, 5, 6],
                "low": [0.5, 1.5, 2.5, 3.5, 4.5],
                "close": [1.5, 2.5, 3.5, 4.5, 5.5],
                "volume": [10, 11, 12, 13, 14],
                "_symbol": [sym] * 5,
                "symbol": [sym] * 5,
            },
            index=idx,
        )

    # First compute AAA (writes cache)
    _ = fc.compute_features_parallel(_df("AAA"), features_cfg, ["atr_f"], fit=True)
    fc.drain_debug_stats()

    # Compute BBB (should NOT hit AAA cache)
    _ = fc.compute_features_parallel(_df("BBB"), features_cfg, ["atr_f"], fit=True)
    stats_bbb = fc.drain_debug_stats()
    assert stats_bbb["cache_hits"]["monthly"] == 0

    # Compute AAA again (should hit AAA cache)
    _ = fc.compute_features_parallel(_df("AAA"), features_cfg, ["atr_f"], fit=True)
    stats_aaa2 = fc.drain_debug_stats()
    assert stats_aaa2["cache_hits"]["monthly"] >= 1

    # Cache keys should include sym=AAA and sym=BBB
    pkl = glob.glob(str(tmp_path / "monthly" / "*.pkl"))
    joined = "\n".join(pkl)
    assert "sym=AAA" in joined
    assert "sym=BBB" in joined
