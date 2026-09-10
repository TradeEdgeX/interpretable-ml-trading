import numpy as np
import pandas as pd

from src.features.loader.feature_wrappers import compute_footprint_features


def test_footprint_basic_outputs_are_unitless_after_normalization():
    # Two 1-hour bars with simple OHLCV
    idx = pd.date_range("2024-01-01", periods=2, freq="H")
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000.0, 1200.0],
        },
        index=idx,
    )

    # Provide explicit open/close boundaries for footprint slicing
    df["open_time"] = df.index
    df["close_time"] = df.index + pd.Timedelta(hours=1)

    # Create ticks within each bar
    tick_times = pd.to_datetime(
        [
            idx[0] + pd.Timedelta(minutes=5),
            idx[0] + pd.Timedelta(minutes=10),
            idx[0] + pd.Timedelta(minutes=20),
            idx[1] + pd.Timedelta(minutes=5),
            idx[1] + pd.Timedelta(minutes=15),
            idx[1] + pd.Timedelta(minutes=30),
        ]
    )
    ticks = pd.DataFrame(
        {
            "price": [100.5, 101.2, 101.2, 101.8, 102.2, 102.9],
            "volume": [10.0, 15.0, 8.0, 12.0, 20.0, 5.0],
            "side": [1, 1, -1, -1, 1, -1],
        },
        index=tick_times,
    )

    out = compute_footprint_features(
        df,
        ticks=ticks,
        open_col="open_time",
        close_col="close_time",
        price_bin_method="fixed_bins",
        price_bin_target_bins=10,
        value_area_pct=0.7,
    )

    # Price-level columns should be ATR-distance (not raw price magnitude)
    price_cols = [
        "fp_poc",
        "fp_hvn",
        "fp_lvn",
        "fp_vah",
        "fp_val",
        "fp_max_imbalance_price",
        "fp_exhaustion_price",
    ]
    for c in price_cols:
        assert c in out.columns
        vals = pd.to_numeric(out[c], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        assert np.isfinite(vals.to_numpy()).all()
        # Should be in "few ATRs" range for typical bars
        assert vals.abs().max() < 50

    # Flow-like columns should be finite
    for c in [
        "fp_delta_poc",
        "fp_max_imbalance_ratio",
        "fp_volume_skew",
        "fp_delta_skew",
        "fp_exhaustion_zscore",
    ]:
        assert c in out.columns
        vals = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        assert np.isfinite(vals.to_numpy()).all()

    # Binary-ish
    assert "fp_delta_divergence" in out.columns
    div = pd.to_numeric(out["fp_delta_divergence"], errors="coerce").fillna(0.0)
    assert ((div >= 0.0) & (div <= 1.0)).all()
