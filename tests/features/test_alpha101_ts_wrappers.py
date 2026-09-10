import pandas as pd

from src.features.registry import get_feature_func


def test_alpha101_ts_core_registered_and_runs(sample_data):
    df = sample_data.copy()
    assert isinstance(df.index, pd.DatetimeIndex)

    # Some df-level features expect a symbol column (even if unused here)
    df["_symbol"] = "BTCUSDT"
    df["symbol"] = "BTCUSDT"

    func = get_feature_func("compute_alpha101_ts_core_from_df")
    out = func(df)

    assert isinstance(out, pd.DataFrame)
    expected = {
        "alpha101_001_ts",
        "alpha101_022_ts",
        "alpha101_043_ts",
        "alpha101_066_ts",
    }
    assert expected.issubset(set(out.columns))
    # Should align index and be numeric
    assert len(out) == len(df)
    for c in expected:
        assert pd.api.types.is_numeric_dtype(out[c])


def test_alpha101_ts_no_future_leak(sample_data):
    df = sample_data.copy()
    df["_symbol"] = "BTCUSDT"
    df["symbol"] = "BTCUSDT"
    func = get_feature_func("compute_alpha101_ts_core_from_df")

    out1 = func(df)
    cutoff = df.index[int(len(df) * 0.7)]

    df2 = df.copy()
    mask = df2.index >= cutoff
    # perturb only future values
    df2.loc[mask, "close"] = df2.loc[mask, "close"] * 3.0
    df2.loc[mask, "volume"] = df2.loc[mask, "volume"] * 5.0

    out2 = func(df2)
    pd.testing.assert_frame_equal(
        out1.loc[out1.index < cutoff],
        out2.loc[out2.index < cutoff],
        check_exact=False,
        atol=1e-12,
        rtol=0,
    )


def test_alpha101_ts_streaming_batch_consistency(sample_data):
    df = sample_data.copy()
    df["_symbol"] = "BTCUSDT"
    df["symbol"] = "BTCUSDT"
    func = get_feature_func("compute_alpha101_ts_core_from_df")

    batch = func(df).sort_index()
    split = df.index[int(len(df) * 0.6)]
    overlap = 80  # bars overlap for rolling windows

    df1 = df.loc[df.index < split].copy()
    df2 = df.iloc[max(0, df.index.get_loc(split) - overlap) :].copy()

    out2 = func(df2).sort_index()
    out2_post = out2.loc[out2.index >= split]
    batch_post = batch.loc[batch.index >= split]
    # Rolling corr/std operations can introduce tiny floating point drift when computed in chunks.
    pd.testing.assert_frame_equal(
        batch_post, out2_post, check_exact=False, atol=1e-6, rtol=1e-9
    )
