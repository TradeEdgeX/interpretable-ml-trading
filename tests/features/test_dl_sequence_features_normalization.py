import numpy as np
import pandas as pd

from src.features.loader.dl_feature_wrappers import (
    compute_dl_sequence_features_from_series,
)


def _make_ohlcv(idx: pd.DatetimeIndex, price_scale: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # similar returns profile, different absolute price scale
    rets = rng.normal(0, 0.002, len(idx))
    close = pd.Series(price_scale * (1.0 + rets).cumprod(), index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) * (1.0 + 0.0005)
    low = pd.concat([open_, close], axis=1).min(axis=1) * (1.0 - 0.0005)
    volume = pd.Series(np.abs(rng.normal(1000, 50, len(idx))), index=idx)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_dl_sequence_embeddings_are_bounded_and_cross_asset_scale_stable():
    idx = pd.date_range("2024-01-01", periods=260, freq="5min")
    df_a = _make_ohlcv(idx, price_scale=100.0, seed=0)
    df_b = _make_ohlcv(idx, price_scale=50000.0, seed=1)

    params = dict(
        backend="auto",
        seq_length=64,
        d_model=16,
        use_fp16=False,
        device="cpu",
        feature_columns=["open", "high", "low", "close", "volume"],
        prefix="dl_seq",
        output_normalization="tanh",
    )

    out_a = compute_dl_sequence_features_from_series(
        open=df_a["open"],
        high=df_a["high"],
        low=df_a["low"],
        close=df_a["close"],
        volume=df_a["volume"],
        **params,
    )
    out_b = compute_dl_sequence_features_from_series(
        open=df_b["open"],
        high=df_b["high"],
        low=df_b["low"],
        close=df_b["close"],
        volume=df_b["volume"],
        **params,
    )

    cols = [f"dl_seq_f{i}" for i in range(params["d_model"])]
    a = out_a[cols].to_numpy()
    b = out_b[cols].to_numpy()

    assert np.isfinite(a).all()
    assert np.isfinite(b).all()

    # bounded embeddings (stable/unitless)
    assert np.nanmax(np.abs(a)) <= 1.0000001
    assert np.nanmax(np.abs(b)) <= 1.0000001

    # Scale stability across assets: std per-dim should be same order of magnitude.
    # Skip degenerate all-zero case.
    std_a = np.nanstd(a, axis=0)
    std_b = np.nanstd(b, axis=0)
    if float(np.nanmax(std_a)) > 1e-9 and float(np.nanmax(std_b)) > 1e-9:
        ratio = np.maximum(std_a / (std_b + 1e-12), std_b / (std_a + 1e-12))
        # Loose bound: we're ensuring "not wildly different" across assets.
        assert float(np.nanmedian(ratio)) < 10.0
