import numpy as np
import pandas as pd

from src.features.time_series.baseline_features import (
    compute_acceleration_3,
    compute_acceleration_3_from_series,
    compute_volume_anomaly,
    compute_volume_anomaly_from_series,
    compute_trend_r2_20,
    compute_trend_r2_20_from_series,
    compute_trend_r2_50,
    compute_trend_r2_50_from_series,
    compute_slope_consistency_score,
    compute_slope_consistency_score_from_series,
    compute_volatility_reversal_score,
    compute_volatility_reversal_score_from_series,
    compute_atr,
    compute_atr_from_series,
    compute_atr_percentile,
    compute_atr_percentile_from_series,
    compute_trend_volatility_alignment,
    compute_trend_volatility_alignment_from_series,
    compute_compression_to_breakout_prob,
    compute_compression_to_breakout_prob_from_series,
    compute_roc_5_from_series,
    compute_trend_confidence_from_series,
)


def test_remaining_baseline_series_entrypoints_match_df_versions():
    idx = pd.date_range("2024-01-01", periods=600, freq="5min")
    rng = np.random.default_rng(0)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.2, len(idx))), index=idx)
    high = close + np.abs(rng.normal(0.1, 0.05, len(idx)))
    low = close - np.abs(rng.normal(0.1, 0.05, len(idx)))
    volume = pd.Series(np.abs(rng.normal(1000, 200, len(idx))), index=idx)

    df = pd.DataFrame(
        {"close": close, "high": high, "low": low, "volume": volume}, index=idx
    )

    # acceleration_3
    df_a = compute_acceleration_3(df.copy(), feature_shift=0)
    s_a = compute_acceleration_3_from_series(close=close, feature_shift=0)[
        "acceleration_3"
    ]
    assert np.allclose(df_a["acceleration_3"].values, s_a.values, equal_nan=True)

    # volume_anomaly
    df_v = compute_volume_anomaly(df.copy())
    s_v = compute_volume_anomaly_from_series(volume=volume)["volume_anomaly"]
    assert np.allclose(df_v["volume_anomaly"].values, s_v.values, equal_nan=True)

    # trend r2
    df_r2_20 = compute_trend_r2_20(df.copy(), feature_shift=0)
    s_r2_20 = compute_trend_r2_20_from_series(close=close, feature_shift=0)[
        "trend_r2_20"
    ]
    assert np.allclose(df_r2_20["trend_r2_20"].values, s_r2_20.values, equal_nan=True)

    df_r2_50 = compute_trend_r2_50(df.copy(), feature_shift=0)
    s_r2_50 = compute_trend_r2_50_from_series(close=close, feature_shift=0)[
        "trend_r2_50"
    ]
    assert np.allclose(df_r2_50["trend_r2_50"].values, s_r2_50.values, equal_nan=True)

    # slope consistency
    df_sc = compute_slope_consistency_score(df.copy())
    s_sc = compute_slope_consistency_score_from_series(close=close)[
        "slope_consistency_score"
    ]
    assert np.allclose(
        df_sc["slope_consistency_score"].values, s_sc.values, equal_nan=True
    )

    # volatility reversal score
    df_vrs = compute_volatility_reversal_score(df.copy())
    s_vrs = compute_volatility_reversal_score_from_series(
        high=high, low=low, close=close
    )["volatility_reversal_score"]
    assert np.allclose(
        df_vrs["volatility_reversal_score"].values, s_vrs.values, equal_nan=True
    )

    # atr_percentile
    df_ap = compute_atr_percentile(df.copy(), window=288, shift=1)
    s_ap = compute_atr_percentile_from_series(
        high=high, low=low, close=close, window=288, shift=1
    )["atr_percentile"]
    assert np.allclose(df_ap["atr_percentile"].values, s_ap.values, equal_nan=True)

    # trend_volatility_alignment (depends on roc_5 + atr_percentile internally)
    df_tva = compute_trend_volatility_alignment(
        df.copy(), feature_shift=0, atr_percentile_window=288
    )
    s_tva = compute_trend_volatility_alignment_from_series(
        close=close, high=high, low=low, feature_shift=0, atr_percentile_window=288
    )["trend_volatility_alignment"]
    assert np.allclose(
        df_tva["trend_volatility_alignment"].values, s_tva.values, equal_nan=True
    )

    # compression_to_breakout_prob (simple product); use roc_5 series
    roc_5 = compute_roc_5_from_series(close=close)
    compression_duration = pd.Series(
        np.maximum(0, rng.normal(5, 2, len(idx))), index=idx
    )
    df_cb = pd.DataFrame(
        {"compression_duration": compression_duration, "roc_5": roc_5}, index=idx
    )
    df_cb2 = compute_compression_to_breakout_prob(df_cb.copy())
    s_cb = compute_compression_to_breakout_prob_from_series(
        compression_duration=compression_duration, roc_5=roc_5
    )["compression_to_breakout_prob"]
    if "compression_to_breakout_prob" in df_cb2.columns:
        assert np.allclose(
            df_cb2["compression_to_breakout_prob"].values, s_cb.values, equal_nan=True
        )
    else:
        # legacy returns df unchanged if missing deps; here we provided them, so it should exist
        raise AssertionError(
            "legacy compute_compression_to_breakout_prob did not create output column"
        )


def test_compute_trend_confidence_from_series_matches_sign_bundle():
    idx = pd.date_range("2024-01-01", periods=80, freq="1h")
    rng = np.random.default_rng(7)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.3, len(idx))), index=idx)
    horizons = (3, 5, 10)
    rets = [close.pct_change(int(h)) for h in horizons]
    signs = pd.concat([np.sign(r) for r in rets], axis=1).fillna(0.0)
    signs.columns = list(range(len(horizons)))
    mean_signs = signs.mean(axis=1)
    expected = signs.abs().mean(axis=1) * mean_signs.abs()
    out = compute_trend_confidence_from_series(close=close, horizons=horizons)
    assert list(out.columns) == [
        "trend_confidence",
        "trend_direction_raw",
        "trend_direction",
    ]
    assert np.allclose(out["trend_confidence"].values, expected.values, equal_nan=True)


def test_compute_atr_from_series():
    """
    测试修复：compute_atr_from_series 函数

    Bug修复：添加了 compute_atr_from_series 函数，返回 DataFrame 格式
    确保 narrow-IO 模式下能正确工作

    注意：ATR 现在返回归一化值 (atr / close)，典型范围 [0.001, 0.1]
    """
    idx = pd.date_range("2024-01-01", periods=100, freq="5min")
    rng = np.random.default_rng(42)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.2, len(idx))), index=idx)
    high = close + np.abs(rng.normal(0.1, 0.05, len(idx)))
    low = close - np.abs(rng.normal(0.1, 0.05, len(idx)))

    # 测试 compute_atr_from_series 返回 DataFrame
    result_df = compute_atr_from_series(high=high, low=low, close=close, period=14)

    # 验证返回类型
    assert isinstance(result_df, pd.DataFrame), "应该返回 DataFrame"
    assert "atr" in result_df.columns, "应该包含 'atr' 列"
    assert len(result_df) == len(idx), "长度应该匹配输入"

    # 验证归一化后的 ATR (atr / close)
    # 原始 ATR 除以 close 后，应该是一个小的比率
    atr_values = result_df["atr"].dropna()
    assert len(atr_values) > 0, "应该有有效的 ATR 值"
    assert (atr_values >= 0).all(), "归一化 ATR 应该 >= 0"

    # 归一化后的典型范围检查
    # 对于正常波动的资产，atr/close 通常在 [0.001, 0.1] 范围内
    assert (
        atr_values.max() < 0.5
    ), f"归一化 ATR 应该 < 0.5，实际 max={atr_values.max():.4f}"
    assert (
        atr_values.mean() < 0.1
    ), f"归一化 ATR 均值应该 < 0.1，实际 mean={atr_values.mean():.4f}"

    # 验证与原始 compute_atr 的归一化关系
    raw_atr = compute_atr(high, low, close, period=14)
    expected_norm = (raw_atr / close).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 检查归一化正确性
    valid_mask = ~np.isnan(result_df["atr"].values) & ~np.isnan(expected_norm.values)
    if valid_mask.sum() > 0:
        assert np.allclose(
            result_df["atr"].values[valid_mask],
            expected_norm.values[valid_mask],
            rtol=1e-6,
        ), "归一化 ATR 应该等于 raw_atr / close"
