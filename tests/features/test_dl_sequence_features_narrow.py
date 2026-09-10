import numpy as np
import pandas as pd

from src.features.loader.dl_feature_wrappers import (
    compute_dl_sequence_features,
    compute_dl_sequence_features_from_series,
)


def test_dl_sequence_features_from_series_matches_df_entrypoint():
    idx = pd.date_range("2024-01-01", periods=200, freq="5min")
    rng = np.random.default_rng(0)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.2, len(idx))), index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.1
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.1
    volume = pd.Series(np.abs(rng.normal(1000, 50, len(idx))), index=idx)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )

    params = dict(
        backend="auto",
        seq_length=32,
        d_model=16,
        use_fp16=False,
        device="cpu",
        feature_columns=["open", "high", "low", "close", "volume"],
        prefix="dl_seq",
    )

    df_out = compute_dl_sequence_features(df.copy(), **params)
    s_out = compute_dl_sequence_features_from_series(
        open=open_, high=high, low=low, close=close, volume=volume, **params
    )

    expected_cols = [f"dl_seq_f{i}" for i in range(params["d_model"])]
    assert all(c in df_out.columns for c in expected_cols)
    assert all(c in s_out.columns for c in expected_cols)
    for c in expected_cols:
        assert df_out[c].equals(s_out[c])
