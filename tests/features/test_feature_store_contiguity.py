import numpy as np
import pandas as pd

from src.features.loader.strategy_feature_loader import StrategyFeatureLoader


def _make_two_month_uptrend() -> pd.DataFrame:
    idx = pd.date_range("2024-01-10", "2024-02-20", freq="4H")
    close = np.linspace(100, 200, len(idx))
    df = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1000, 1200, len(idx)),
            "_symbol": "TEST",
            "symbol": "TEST",
        },
        index=idx,
    )
    return df


def test_monthly_cache_reset_vs_contiguous(tmp_path):
    df = _make_two_month_uptrend()

    loader_monthly = StrategyFeatureLoader(
        cache_dir=str(tmp_path / "monthly_cache"),
        use_disk_cache=True,
        use_memory_cache=False,
        use_monthly_cache=True,
    )
    loader_full = StrategyFeatureLoader(
        cache_dir=str(tmp_path / "full_cache"),
        use_disk_cache=False,
        use_memory_cache=False,
        use_monthly_cache=False,
    )

    df_monthly = loader_monthly.load_features_from_requested(
        df, requested_features=["rsi"], fit=True
    )
    df_full = loader_full.load_features_from_requested(
        df, requested_features=["rsi"], fit=True
    )

    feb_start = df.index[df.index.month == 2][0]
    rsi_monthly = float(df_monthly.loc[feb_start, "rsi"])
    rsi_full = float(df_full.loc[feb_start, "rsi"])

    # Monthly cache computes per-month and resets rolling state at boundary.
    assert rsi_monthly <= 55.0
    # Contiguous compute should preserve trend across months.
    assert rsi_full >= 80.0
