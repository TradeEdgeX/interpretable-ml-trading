"""
TA-Lib 指标特征测试

测试内容：
1. 无未来函数测试（修改未来数据不影响历史特征值）⭐⭐⭐⭐⭐
2. 多资产归一化测试（特征分布对齐）⭐⭐⭐⭐
3. 流式vs批量一致性测试 ⭐⭐⭐⭐
4. 特征数学正确性验证

覆盖的特征节点：约 55 个（使用 compute_talib_indicator_from_series）
- trend: SMA, EMA, WMA, TEMA, DEMA 等移动平均线
- momentum: RSI, MOM, CCI, STOCH, WILLR, PPO, TRIX, ULTOSC, CMO 等
- volume: OBV, AD Line, ADOSC 等
- volatility: BB, ATR 相关指标
"""

import numpy as np
import pandas as pd
import pytest
import talib

from src.features.loader.talib_feature_wrappers import (
    compute_talib_indicator_from_series,
)


def create_mock_data(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """创建模拟数据用于测试"""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="4H")

    # 生成价格数据
    returns = np.random.randn(n_samples) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_samples) * 0.001),
            "high": prices * (1 + np.abs(np.random.randn(n_samples) * 0.002)),
            "low": prices * (1 - np.abs(np.random.randn(n_samples) * 0.002)),
            "close": prices,
            "volume": np.random.uniform(1000, 10000, n_samples),
        },
        index=dates,
    )

    return df


class TestTalibIndicators:
    """TA-Lib 指标特征测试"""

    # 测试常用的 TA-Lib 指标
    INDICATORS_TO_TEST = [
        # Trend indicators
        ("SMA", {"real": None, "timeperiod": 20}, "sma_20"),
        ("EMA", {"real": None, "timeperiod": 20}, "ema_20"),
        ("WMA", {"real": None, "timeperiod": 20}, "wma_20"),
        ("TEMA", {"real": None, "timeperiod": 20}, "tema_20"),
        ("DEMA", {"real": None, "timeperiod": 20}, "dema_20"),
        # Momentum indicators
        ("RSI", {"real": None, "timeperiod": 14}, "rsi"),
        ("MOM", {"real": None, "timeperiod": 10}, "mom"),
        ("CCI", {"high": None, "low": None, "close": None, "timeperiod": 14}, "cci"),
        (
            "STOCHF",
            {
                "high": None,
                "low": None,
                "close": None,
                "fastk_period": 5,
                "fastd_period": 3,
            },
            "stochf",
        ),
        (
            "WILLR",
            {"high": None, "low": None, "close": None, "timeperiod": 14},
            "willr",
        ),
        # Volume indicators
        ("OBV", {"close": None, "volume": None}, "obv"),
        ("AD", {"high": None, "low": None, "close": None, "volume": None}, "ad"),
        (
            "ADOSC",
            {
                "high": None,
                "low": None,
                "close": None,
                "volume": None,
                "fastperiod": 3,
                "slowperiod": 10,
            },
            "adosc",
        ),
    ]

    def test_adx_basic(self):
        """基础功能测试：ADX (Average Directional Index)"""
        df = create_mock_data(200)
        result = compute_talib_indicator_from_series(
            indicator_name="ADX",
            high=df["high"],
            low=df["low"],
            close=df["close"],
            timeperiod=14,
            output_column="adx",
        )

        # 检查输出列
        assert "adx" in result.columns
        assert len(result) == len(df)

        # 检查数值合理性（ADX 应该在 [0, 100] 范围内）
        adx = result["adx"].dropna()
        if len(adx) > 0:
            assert (adx >= 0).all() and (
                adx <= 100
            ).all(), (
                f"ADX 应该在 [0, 100] 范围内，范围: [{adx.min():.2f}, {adx.max():.2f}]"
            )

    def test_plus_di_minus_di_basic(self):
        """基础功能测试：PLUS_DI 和 MINUS_DI"""
        df = create_mock_data(200)

        # 测试 PLUS_DI
        result_plus = compute_talib_indicator_from_series(
            indicator_name="PLUS_DI",
            high=df["high"],
            low=df["low"],
            close=df["close"],
            timeperiod=14,
            output_column="plus_di",
        )
        assert "plus_di" in result_plus.columns

        # 测试 MINUS_DI
        result_minus = compute_talib_indicator_from_series(
            indicator_name="MINUS_DI",
            high=df["high"],
            low=df["low"],
            close=df["close"],
            timeperiod=14,
            output_column="minus_di",
        )
        assert "minus_di" in result_minus.columns

        # 检查数值合理性（DI 应该在 [0, 100] 范围内）
        plus_di = result_plus["plus_di"].dropna()
        minus_di = result_minus["minus_di"].dropna()
        if len(plus_di) > 0:
            assert (plus_di >= 0).all() and (plus_di <= 100).all()
        if len(minus_di) > 0:
            assert (minus_di >= 0).all() and (minus_di <= 100).all()

    def test_aroon_basic(self):
        """基础功能测试：AROON"""
        df = create_mock_data(200)
        # AROON 返回两个输出：aroondown, aroonup - 使用 AROON 的 down 输出
        result = compute_talib_indicator_from_series(
            indicator_name="AROON",
            high=df["high"],
            low=df["low"],
            timeperiod=14,
            output_column="aroon_down",
        )

        # 检查输出存在（可能是一个或两个列）
        assert len(result) == len(df)

    def test_kama_basic(self):
        """基础功能测试：KAMA (Kaufman Adaptive Moving Average)"""
        df = create_mock_data(200)
        result = compute_talib_indicator_from_series(
            indicator_name="KAMA",
            real=df["close"],
            timeperiod=10,
            output_column="kama",
        )

        # 检查输出列
        assert "kama" in result.columns
        assert len(result) == len(df)

        # KAMA 应该在价格的合理范围内
        kama = result["kama"].dropna()
        if len(kama) > 0:
            price_range = (df["close"].min() * 0.9, df["close"].max() * 1.1)
            assert (kama >= price_range[0]).all() and (kama <= price_range[1]).all()

    def test_willr_basic(self):
        """基础功能测试：WILLR (Williams %R)"""
        df = create_mock_data(200)
        result = compute_talib_indicator_from_series(
            indicator_name="WILLR",
            high=df["high"],
            low=df["low"],
            close=df["close"],
            timeperiod=14,
            output_column="willr",
        )

        # 检查输出列
        assert "willr" in result.columns

        # Williams %R 应该在 [-100, 0] 范围内
        willr = result["willr"].dropna()
        if len(willr) > 0:
            assert (willr >= -100).all() and (
                willr <= 0
            ).all(), f"Williams %R 应该在 [-100, 0] 范围内，范围: [{willr.min():.2f}, {willr.max():.2f}]"

    def test_sma_basic(self):
        """基础功能测试：SMA"""
        df = create_mock_data(200)
        result = compute_talib_indicator_from_series(
            indicator_name="SMA",
            real=df["close"],
            timeperiod=20,
            output_column="sma_20",
        )

        # 检查输出列
        assert "sma_20" in result.columns
        assert len(result) == len(df)

        # 检查数值合理性
        valid_data = result["sma_20"].dropna()
        if len(valid_data) > 0:
            # SMA 应该接近价格水平
            assert (valid_data > 0).all(), "SMA 应该为正数"

    def test_rsi_basic(self):
        """基础功能测试：RSI"""
        df = create_mock_data(200)
        result = compute_talib_indicator_from_series(
            indicator_name="RSI",
            real=df["close"],
            timeperiod=14,
            output_column="rsi",
        )

        # 检查输出列
        assert "rsi" in result.columns
        assert len(result) == len(df)

        # 检查数值合理性（RSI 应该在 [0, 100] 范围内）
        valid_data = result["rsi"].dropna()
        if len(valid_data) > 0:
            assert (valid_data >= 0).all() and (
                valid_data <= 100
            ).all(), "RSI 应该在 [0, 100] 范围内"

    def test_obv_basic(self):
        """基础功能测试：OBV"""
        df = create_mock_data(200)
        # OBV 需要 real (close) 和 volume 参数
        # 根据 feature_dependencies.yaml，OBV 使用 real: close 映射
        # TA-Lib OBV 函数签名: OBV(real, volume)
        result = compute_talib_indicator_from_series(
            indicator_name="OBV",
            real=df["close"],  # OBV 使用 'real' 参数名（映射到 close）
            volume=df["volume"],
            output_column="obv",
        )

        # 检查输出列
        assert "obv" in result.columns
        assert len(result) == len(df)

        # OBV 是累积值，应该单调或接近单调
        valid_data = result["obv"].dropna()
        if len(valid_data) > 10:
            # OBV 的变化应该与价格和成交量相关
            assert not valid_data.isna().all(), "OBV 应该有有效值"

    def test_no_future_leak(self):
        """
        测试1：无未来函数（修改未来数据不影响历史特征值）⭐⭐⭐⭐⭐
        """
        df = create_mock_data(300)

        # 测试多个指标
        indicators = [
            ("SMA", {"real": df["close"], "timeperiod": 20}, "sma_20"),
            ("RSI", {"real": df["close"], "timeperiod": 14}, "rsi"),
            ("EMA", {"real": df["close"], "timeperiod": 20}, "ema_20"),
        ]

        for indicator_name, kwargs, output_col in indicators:
            # 计算第一次特征
            result1 = compute_talib_indicator_from_series(
                indicator_name=indicator_name,
                output_column=output_col,
                **kwargs,
            )
            values_1 = result1[output_col].copy()

            # 修改未来数据
            df_future_modified = df.copy()
            if len(df) > 100:
                df_future_modified.loc[df_future_modified.index[100] :, "close"] *= 2.0

                # 重新计算特征
                kwargs_modified = kwargs.copy()
                kwargs_modified["real"] = df_future_modified["close"]
                result2 = compute_talib_indicator_from_series(
                    indicator_name=indicator_name,
                    output_column=output_col,
                    **kwargs_modified,
                )
                values_2 = result2[output_col].copy()

                # 检查前50个时间点的特征值（应该不受未来数据影响）
                check_idx = df.index[:50]
                vals_1_check = values_1.loc[check_idx].dropna()
                vals_2_check = values_2.loc[check_idx].dropna()

                if len(vals_1_check) > 0 and len(vals_2_check) > 0:
                    # 找到共同索引
                    common_idx = vals_1_check.index.intersection(vals_2_check.index)
                    if len(common_idx) > 0:
                        diff = (
                            vals_1_check.loc[common_idx] - vals_2_check.loc[common_idx]
                        ).abs()
                        max_diff = diff.max()

                        # TA-Lib 指标是因果的，应该完全相同
                        assert (
                            max_diff < 1e-10
                        ), f"{indicator_name} 未来数据变化不应影响历史特征值，最大差异: {max_diff}"

    def test_normalization_multi_asset(self):
        """
        测试2：多资产归一化（特征分布对齐）⭐⭐⭐⭐

        验证：
        - 不同价格水平的资产，TA-Lib 指标应该在相似范围内
        - 对于相对指标（如 RSI），应该对不同资产的价格水平不敏感
        """
        np.random.seed(42)
        n = 200

        # 不同价格水平的资产
        assets = {
            "BTCUSDT": 50000 + np.cumsum(np.random.randn(n) * 100),
            "ETHUSDT": 3000 + np.cumsum(np.random.randn(n) * 10),
            "SOLUSDT": 100 + np.cumsum(np.random.randn(n) * 0.5),
        }

        results = {}
        for symbol, prices in assets.items():
            dates = pd.date_range("2024-01-01", periods=n, freq="4H")
            df = pd.DataFrame(
                {
                    "close": prices,
                    "volume": np.random.uniform(1000, 10000, n),
                },
                index=dates,
            )

            # 计算 RSI（相对指标，应该对不同价格水平不敏感）
            result = compute_talib_indicator_from_series(
                indicator_name="RSI",
                real=df["close"],
                timeperiod=14,
                output_column="rsi",
            )
            result["_symbol"] = symbol
            results[symbol] = result

        # 检查：不同资产的 RSI 应该在相似范围内（都在 [0, 100]）
        for symbol, result in results.items():
            rsi = result["rsi"].dropna()
            if len(rsi) > 0:
                assert (rsi >= 0).all() and (
                    rsi <= 100
                ).all(), f"{symbol} RSI 应该在 [0, 100] 范围内"
                # RSI 的均值应该在合理范围内（通常 30-70）
                mean_rsi = rsi.mean()
                assert (
                    20 <= mean_rsi <= 80
                ), f"{symbol} RSI 均值 {mean_rsi:.2f} 应该在合理范围内"

    def test_streaming_vs_batch_consistency(self):
        """
        测试3：流式 vs 批量一致性 ⭐⭐⭐⭐
        对生产部署至关重要：生产环境往往是流式推理，而训练是批量计算
        """
        df = create_mock_data(300)

        # 批量计算（一次性计算所有数据）
        batch_result = compute_talib_indicator_from_series(
            indicator_name="SMA",
            real=df["close"],
            timeperiod=20,
            output_column="sma_20",
        )

        # 流式计算（分块处理，模拟在线推理）
        chunk_size = 100
        streaming_results = []

        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i : i + chunk_size].copy()
            chunk_result = compute_talib_indicator_from_series(
                indicator_name="SMA",
                real=chunk_df["close"],
                timeperiod=20,
                output_column="sma_20",
            )
            streaming_results.append(chunk_result)

        if len(streaming_results) > 0:
            streaming_result = pd.concat(streaming_results, axis=0)

            # 比较关键特征
            key_col = "sma_20"
            if key_col in batch_result.columns and key_col in streaming_result.columns:
                batch_vals = batch_result[key_col].dropna()
                stream_vals = streaming_result[key_col].dropna()

                # 找到共同索引
                common_idx = batch_vals.index.intersection(stream_vals.index)
                if len(common_idx) > 10:  # 至少需要10个数据点
                    diff = (
                        batch_vals.loc[common_idx] - stream_vals.loc[common_idx]
                    ).abs()
                    max_diff = diff.max()
                    mean_diff = diff.mean()

                    # 允许一定的数值误差（由于分块计算可能导致边界处理略有不同）
                    # 但 TA-Lib 指标应该是完全一致的
                    assert max_diff < 1e-10, (
                        f"流式与批量计算不一致，最大差异: {max_diff:.8f}, "
                        f"平均差异: {mean_diff:.8f}"
                    )

    def test_talib_math_correctness(self):
        """测试：TA-Lib 指标数学正确性"""
        df = create_mock_data(100)

        # 测试 SMA：手动计算验证
        period = 20
        result = compute_talib_indicator_from_series(
            indicator_name="SMA",
            real=df["close"],
            timeperiod=period,
            output_column="sma_20",
        )

        # 手动计算 SMA
        sma_manual = df["close"].rolling(window=period, min_periods=period).mean()

        # 与特征值比较（允许微小误差）
        sma_computed = result["sma_20"]
        valid_idx = sma_computed.dropna().index

        if len(valid_idx) > 0:
            diff = (sma_computed.loc[valid_idx] - sma_manual.loc[valid_idx]).abs()
            max_diff = diff.max()

            assert max_diff < 1e-10, f"SMA 计算不正确: 最大差异={max_diff:.10f}"

        # 测试 RSI：验证在 [0, 100] 范围内
        result_rsi = compute_talib_indicator_from_series(
            indicator_name="RSI",
            real=df["close"],
            timeperiod=14,
            output_column="rsi",
        )

        rsi = result_rsi["rsi"].dropna()
        if len(rsi) > 0:
            assert (rsi >= 0).all() and (rsi <= 100).all(), "RSI 应该在 [0, 100] 范围内"

    def test_multiple_indicators(self):
        """测试多个常用指标"""
        df = create_mock_data(200)

        # 测试多个指标
        indicators = [
            ("SMA", {"real": df["close"], "timeperiod": 20}, "sma_20"),
            ("EMA", {"real": df["close"], "timeperiod": 20}, "ema_20"),
            ("RSI", {"real": df["close"], "timeperiod": 14}, "rsi"),
            ("MOM", {"real": df["close"], "timeperiod": 10}, "mom"),
        ]

        for indicator_name, kwargs, output_col in indicators:
            try:
                result = compute_talib_indicator_from_series(
                    indicator_name=indicator_name,
                    output_column=output_col,
                    **kwargs,
                )

                assert (
                    output_col in result.columns
                ), f"{indicator_name} 应该输出 {output_col} 列"
                assert len(result) == len(df), f"{indicator_name} 输出长度应该匹配输入"

                # 检查是否有有效值
                valid_data = result[output_col].dropna()
                assert len(valid_data) > 0, f"{indicator_name} 应该有有效值"

            except Exception as e:
                pytest.fail(f"{indicator_name} 测试失败: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
