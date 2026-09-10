"""
多资产归一化可比性测试

验证归一化特征在不同资产间具有可比性：
1. 相同市场状态下，不同资产的归一化特征值应该在相似范围内
2. 归一化特征不应该因为绝对价格差异而产生数量级差异
3. 特征分布应该符合预期的归一化范围

运行测试:
    pytest tests/features/test_multi_asset_normalization.py -v
"""

import pytest
import pandas as pd
import numpy as np
import talib
from typing import Dict, Tuple


def generate_multi_asset_data(
    n_bars: int = 500, seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """生成多资产模拟数据，模拟 BTC/ETH/SOL 等不同价格量级的资产"""
    np.random.seed(seed)

    assets = {
        # 资产名: (初始价格, 日波动率, 日均成交量)
        "BTCUSDT": (50000, 0.03, 1e9),
        "ETHUSDT": (3000, 0.04, 5e8),
        "SOLUSDT": (100, 0.05, 1e8),
        "DOGEUSDT": (0.1, 0.06, 5e7),
    }

    result = {}
    for symbol, (price, vol, volume) in assets.items():
        # 生成价格序列（几何布朗运动）
        returns = np.random.normal(0, vol / np.sqrt(24 * 6), n_bars)  # 4小时bar
        close = price * np.cumprod(1 + returns)

        # 生成 OHLC
        high = close * (1 + np.abs(np.random.normal(0, vol / 2, n_bars)))
        low = close * (1 - np.abs(np.random.normal(0, vol / 2, n_bars)))
        open_ = close * (1 + np.random.normal(0, vol / 3, n_bars))

        # 确保 OHLC 关系正确
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))

        # 生成成交量
        vol_data = volume * (1 + np.random.normal(0, 0.3, n_bars))
        vol_data = np.abs(vol_data)

        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": vol_data,
            }
        )
        df.index = pd.date_range("2024-01-01", periods=n_bars, freq="4h")
        result[symbol] = df

    return result


def check_feature_range(
    feature_values: Dict[str, pd.Series],
    expected_range: Tuple[float, float],
    tolerance: float = 0.1,
    feature_name: str = "",
) -> Tuple[bool, str]:
    """检查特征值是否在预期范围内"""
    min_val, max_val = expected_range

    for symbol, values in feature_values.items():
        valid_values = values.dropna()
        if len(valid_values) == 0:
            continue

        actual_min = valid_values.min()
        actual_max = valid_values.max()

        # 允许一定的容差
        if actual_min < min_val - tolerance or actual_max > max_val + tolerance:
            return (
                False,
                f"{feature_name} for {symbol}: [{actual_min:.4f}, {actual_max:.4f}] not in [{min_val}, {max_val}]",
            )

    return True, ""


def check_cross_asset_comparability(
    feature_values: Dict[str, pd.Series],
    max_std_ratio: float = 5.0,
    feature_name: str = "",
) -> Tuple[bool, str]:
    """
    检查跨资产可比性：
    不同资产的特征标准差不应该差距太大
    """
    stds = {}
    for symbol, values in feature_values.items():
        valid_values = values.dropna()
        if len(valid_values) > 10:
            stds[symbol] = valid_values.std()

    if len(stds) < 2:
        return True, ""

    std_values = list(stds.values())
    max_std = max(std_values)
    min_std = min(std_values)

    if min_std > 0 and max_std / min_std > max_std_ratio:
        return (
            False,
            f"{feature_name}: std ratio {max_std/min_std:.2f} > {max_std_ratio} ({stds})",
        )

    return True, ""


class TestNormalizedFeatures:
    """测试归一化特征的跨资产可比性"""

    @pytest.fixture
    def multi_asset_data(self):
        """多资产测试数据"""
        return generate_multi_asset_data()

    def test_sma_position_comparability(self, multi_asset_data):
        """测试 SMA 200 Position 跨资产可比性"""
        from src.features.time_series.baseline_features import (
            compute_sma_position_from_series,
        )

        feature_values = {}
        for symbol, df in multi_asset_data.items():
            # 先计算 SMA 200
            sma_200 = df["close"].rolling(200, min_periods=1).mean()
            result = compute_sma_position_from_series(
                close=df["close"], sma_200=sma_200
            )
            feature_values[symbol] = result["sma_200_position"]

        # 检查范围: 通常在 [-0.5, 0.5] 左右（±50%偏离 SMA）
        ok, msg = check_feature_range(
            feature_values, (-1.0, 1.0), tolerance=0.5, feature_name="sma_200_position"
        )
        assert ok, msg

        # 检查跨资产可比性
        ok, msg = check_cross_asset_comparability(
            feature_values, max_std_ratio=5.0, feature_name="sma_200_position"
        )
        assert ok, msg

    def test_volume_ratio_comparability(self, multi_asset_data):
        """测试 Volume Ratio 跨资产可比性"""
        from src.features.time_series.baseline_features import (
            compute_volume_ratio_from_series,
        )

        feature_values = {}
        for symbol, df in multi_asset_data.items():
            result = compute_volume_ratio_from_series(volume=df["volume"])
            feature_values[symbol] = result["volume_ratio"]

        # 检查范围: 通常在 [0, 5] 左右（成交量相对均值的倍数）
        ok, msg = check_feature_range(
            feature_values, (0, 10), tolerance=1.0, feature_name="volume_ratio"
        )
        assert ok, msg

        # 检查跨资产可比性
        ok, msg = check_cross_asset_comparability(
            feature_values, max_std_ratio=5.0, feature_name="volume_ratio"
        )
        assert ok, msg

    def test_rsi_comparability(self, multi_asset_data):
        """测试 RSI 跨资产可比性（天然归一化）"""
        feature_values = {}
        for symbol, df in multi_asset_data.items():
            rsi = talib.RSI(df["close"].values, timeperiod=14)
            feature_values[symbol] = pd.Series(rsi, index=df.index)

        # RSI 范围: [0, 100]
        ok, msg = check_feature_range(
            feature_values, (0, 100), tolerance=1.0, feature_name="rsi"
        )
        assert ok, msg

        # 检查跨资产可比性
        ok, msg = check_cross_asset_comparability(
            feature_values, max_std_ratio=3.0, feature_name="rsi"
        )
        assert ok, msg

    def test_bb_width_normalized_comparability(self, multi_asset_data):
        """测试 BB Width Normalized 跨资产可比性"""
        from src.features.time_series.baseline_features import (
            compute_bb_width_features_from_series,
        )

        feature_values = {}
        for symbol, df in multi_asset_data.items():
            result = compute_bb_width_features_from_series(
                close=df["close"], high=df["high"], low=df["low"]
            )
            feature_values[symbol] = result["bb_width_normalized"]

        # BB Width Normalized 是 rolling percentile，范围 [0, 1]，但可能有轻微超出
        ok, msg = check_feature_range(
            feature_values, (0, 2), tolerance=0.5, feature_name="bb_width_normalized"
        )
        assert ok, msg

        # 检查跨资产可比性
        ok, msg = check_cross_asset_comparability(
            feature_values, max_std_ratio=5.0, feature_name="bb_width_normalized"
        )
        assert ok, msg

    def test_bb_position_comparability(self, multi_asset_data):
        """测试 BB Position 跨资产可比性"""
        from src.features.time_series.baseline_features import (
            compute_bb_width_features_from_series,
        )

        feature_values = {}
        for symbol, df in multi_asset_data.items():
            result = compute_bb_width_features_from_series(
                close=df["close"], high=df["high"], low=df["low"]
            )
            feature_values[symbol] = result["bb_position"]

        # BB Position 范围: [-1, 1]（可以超出）
        ok, msg = check_feature_range(
            feature_values, (-2, 2), tolerance=1.0, feature_name="bb_position"
        )
        assert ok, msg

        # 检查跨资产可比性
        ok, msg = check_cross_asset_comparability(
            feature_values, max_std_ratio=3.0, feature_name="bb_position"
        )
        assert ok, msg

    def test_trend_r2_comparability(self, multi_asset_data):
        """测试 Trend R2 跨资产可比性"""
        from src.features.time_series.baseline_features import (
            compute_trend_r2_20_from_series,
        )

        feature_values = {}
        for symbol, df in multi_asset_data.items():
            result = compute_trend_r2_20_from_series(close=df["close"])
            feature_values[symbol] = result["trend_r2_20"]

        # R2 范围: [0, 1]
        ok, msg = check_feature_range(
            feature_values, (0, 1), tolerance=0.1, feature_name="trend_r2_20"
        )
        assert ok, msg

        # 检查跨资产可比性
        ok, msg = check_cross_asset_comparability(
            feature_values, max_std_ratio=3.0, feature_name="trend_r2_20"
        )
        assert ok, msg

    def test_atr_percentile_comparability(self, multi_asset_data):
        """测试 ATR Percentile 跨资产可比性"""
        from src.features.time_series.baseline_features import (
            compute_atr_percentile_from_series,
        )

        feature_values = {}
        for symbol, df in multi_asset_data.items():
            result = compute_atr_percentile_from_series(
                high=df["high"], low=df["low"], close=df["close"]
            )
            feature_values[symbol] = result["atr_percentile"]

        # ATR Percentile 范围: [0, 1]
        ok, msg = check_feature_range(
            feature_values, (0, 1), tolerance=0.1, feature_name="atr_percentile"
        )
        assert ok, msg

        # 检查跨资产可比性
        ok, msg = check_cross_asset_comparability(
            feature_values, max_std_ratio=3.0, feature_name="atr_percentile"
        )
        assert ok, msg


class TestUnnormalizedFeatureIssues:
    """测试未归一化特征的跨资产问题（验证问题确实存在）"""

    @pytest.fixture
    def multi_asset_data(self):
        """多资产测试数据"""
        return generate_multi_asset_data()

    def test_raw_sma_not_comparable(self, multi_asset_data):
        """验证原始 SMA 不可跨资产比较"""
        feature_values = {}
        for symbol, df in multi_asset_data.items():
            sma = talib.SMA(df["close"].values, timeperiod=20)
            feature_values[symbol] = pd.Series(sma, index=df.index)

        # 计算均值比例
        means = {s: v.dropna().mean() for s, v in feature_values.items()}
        max_mean = max(means.values())
        min_mean = min(means.values())

        # 原始 SMA 的均值差异应该非常大（因为价格量级不同）
        mean_ratio = max_mean / min_mean
        assert (
            mean_ratio > 1000
        ), f"Raw SMA mean ratio {mean_ratio:.0f} should be > 1000 (BTC vs DOGE)"

    def test_raw_atr_not_comparable(self, multi_asset_data):
        """验证原始 ATR 不可跨资产比较"""
        feature_values = {}
        for symbol, df in multi_asset_data.items():
            atr = talib.ATR(
                df["high"].values, df["low"].values, df["close"].values, timeperiod=14
            )
            feature_values[symbol] = pd.Series(atr, index=df.index)

        # 计算均值比例
        means = {s: v.dropna().mean() for s, v in feature_values.items()}
        max_mean = max(means.values())
        min_mean = min(means.values())

        # 原始 ATR 的均值差异应该非常大
        mean_ratio = max_mean / min_mean
        assert (
            mean_ratio > 1000
        ), f"Raw ATR mean ratio {mean_ratio:.0f} should be > 1000"

    def test_raw_obv_not_comparable(self, multi_asset_data):
        """验证原始 OBV 不可跨资产比较"""
        feature_values = {}
        for symbol, df in multi_asset_data.items():
            obv = talib.OBV(df["close"].values, df["volume"].values)
            feature_values[symbol] = pd.Series(obv, index=df.index)

        # 计算标准差比例
        stds = {s: v.dropna().std() for s, v in feature_values.items()}
        max_std = max(stds.values())
        min_std = min(stds.values())

        # 原始 OBV 的标准差差异应该很大
        std_ratio = max_std / min_std
        assert std_ratio > 5, f"Raw OBV std ratio {std_ratio:.0f} should be > 5"

    def test_raw_macd_not_comparable(self, multi_asset_data):
        """验证原始 MACD 不可跨资产比较"""
        feature_values = {}
        for symbol, df in multi_asset_data.items():
            macd, _, _ = talib.MACD(df["close"].values)
            feature_values[symbol] = pd.Series(macd, index=df.index)

        # 计算标准差比例
        stds = {s: v.dropna().std() for s, v in feature_values.items()}
        max_std = max(stds.values())
        min_std = min(stds.values())

        # 原始 MACD 的标准差差异应该很大
        std_ratio = max_std / min_std
        assert std_ratio > 100, f"Raw MACD std ratio {std_ratio:.0f} should be > 100"


class TestNormalizationMethods:
    """测试归一化方法的有效性"""

    @pytest.fixture
    def multi_asset_data(self):
        """多资产测试数据"""
        return generate_multi_asset_data()

    def test_price_position_normalization(self, multi_asset_data):
        """测试价格位置归一化: (close - sma) / close"""

        def compute_position(df, window=20):
            sma = df["close"].rolling(window).mean()
            position = (df["close"] - sma) / df["close"]
            return position

        feature_values = {s: compute_position(df) for s, df in multi_asset_data.items()}

        # 位置归一化后，所有资产的分布应该相似
        ok, msg = check_cross_asset_comparability(
            feature_values, max_std_ratio=3.0, feature_name="price_position"
        )
        assert ok, msg

    def test_atr_normalization(self, multi_asset_data):
        """测试 ATR 归一化: atr / close"""

        def compute_atr_normalized(df, window=14):
            atr = talib.ATR(
                df["high"].values,
                df["low"].values,
                df["close"].values,
                timeperiod=window,
            )
            atr_norm = atr / df["close"].values
            return pd.Series(atr_norm, index=df.index)

        feature_values = {
            s: compute_atr_normalized(df) for s, df in multi_asset_data.items()
        }

        # ATR 归一化后，所有资产的分布应该相似
        ok, msg = check_cross_asset_comparability(
            feature_values, max_std_ratio=3.0, feature_name="atr_normalized"
        )
        assert ok, msg

    def test_volume_ratio_normalization(self, multi_asset_data):
        """测试成交量比率归一化: volume / rolling_mean"""

        def compute_vol_ratio(df, window=20):
            mean_vol = df["volume"].rolling(window).mean()
            ratio = df["volume"] / (mean_vol + 1e-8)
            return ratio

        feature_values = {
            s: compute_vol_ratio(df) for s, df in multi_asset_data.items()
        }

        # 成交量比率归一化后，所有资产的分布应该相似
        ok, msg = check_cross_asset_comparability(
            feature_values, max_std_ratio=3.0, feature_name="volume_ratio"
        )
        assert ok, msg

    def test_macd_atr_normalization(self, multi_asset_data):
        """测试 MACD / ATR 归一化"""

        def compute_macd_normalized(df):
            macd, _, _ = talib.MACD(df["close"].values)
            atr = talib.ATR(
                df["high"].values, df["low"].values, df["close"].values, timeperiod=14
            )
            macd_norm = macd / (atr + 1e-8)
            return pd.Series(macd_norm, index=df.index)

        feature_values = {
            s: compute_macd_normalized(df) for s, df in multi_asset_data.items()
        }

        # MACD/ATR 归一化后，所有资产的分布应该相似
        ok, msg = check_cross_asset_comparability(
            feature_values, max_std_ratio=5.0, feature_name="macd_atr_normalized"
        )
        assert ok, msg

    def test_obv_change_normalization(self, multi_asset_data):
        """测试 OBV 变化率归一化 (使用 pct_change 而不是 ratio)"""

        def compute_obv_change(df, window=20):
            obv = talib.OBV(df["close"].values, df["volume"].values)
            obv_series = pd.Series(obv, index=df.index)
            # 使用变化率而不是绝对值比率
            obv_change = obv_series.diff() / (obv_series.rolling(window).std() + 1e-8)
            return obv_change

        feature_values = {
            s: compute_obv_change(df) for s, df in multi_asset_data.items()
        }

        # OBV 标准化变化后，所有资产的分布应该相似
        ok, msg = check_cross_asset_comparability(
            feature_values, max_std_ratio=5.0, feature_name="obv_change_normalized"
        )
        assert ok, msg


class TestTalibNormalizeModes:
    """测试 talib_feature_wrappers 的 normalize_mode 参数"""

    @pytest.fixture
    def multi_asset_data(self):
        """多资产测试数据"""
        return generate_multi_asset_data()

    def test_sma_position_mode(self, multi_asset_data):
        """测试 SMA position 归一化模式"""
        from src.features.loader.talib_feature_wrappers import (
            compute_talib_indicator_from_series,
        )

        feature_values = {}
        for symbol, df in multi_asset_data.items():
            result = compute_talib_indicator_from_series(
                indicator_name="SMA",
                timeperiod=20,
                output_column="sma_20_position",
                normalize_mode="position",
                real=df["close"],
            )
            feature_values[symbol] = result["sma_20_position"]

        # 验证跨资产可比性
        ok, msg = check_cross_asset_comparability(
            feature_values, max_std_ratio=3.0, feature_name="sma_20_position"
        )
        assert ok, msg

        # 验证范围 [-0.5, 0.5]（正常市场下不应偏离太远）
        for symbol, vals in feature_values.items():
            vals_clean = vals.dropna()
            assert vals_clean.min() > -0.5, f"{symbol}: min too low"
            assert vals_clean.max() < 0.5, f"{symbol}: max too high"

    def test_ema_position_mode(self, multi_asset_data):
        """测试 EMA position 归一化模式"""
        from src.features.loader.talib_feature_wrappers import (
            compute_talib_indicator_from_series,
        )

        feature_values = {}
        for symbol, df in multi_asset_data.items():
            result = compute_talib_indicator_from_series(
                indicator_name="EMA",
                timeperiod=20,
                output_column="ema_20_position",
                normalize_mode="position",
                real=df["close"],
            )
            feature_values[symbol] = result["ema_20_position"]

        ok, msg = check_cross_asset_comparability(
            feature_values, max_std_ratio=3.0, feature_name="ema_20_position"
        )
        assert ok, msg

    def test_obv_change_ratio_mode(self, multi_asset_data):
        """测试 OBV change_ratio 归一化模式"""
        from src.features.loader.talib_feature_wrappers import (
            compute_talib_indicator_from_series,
        )

        feature_values = {}
        for symbol, df in multi_asset_data.items():
            result = compute_talib_indicator_from_series(
                indicator_name="OBV",
                output_column="obv_normalized",
                normalize_mode="change_ratio",
                real=df["close"],
                volume=df["volume"],
            )
            feature_values[symbol] = result["obv_normalized"]

        # OBV change_ratio 应该近似标准正态分布
        ok, msg = check_cross_asset_comparability(
            feature_values, max_std_ratio=5.0, feature_name="obv_normalized"
        )
        assert ok, msg

        # 验证标准差接近 1
        for symbol, vals in feature_values.items():
            vals_clean = vals.dropna()
            std = vals_clean.std()
            assert 0.5 < std < 2.0, f"{symbol}: std {std:.2f} not near 1.0"

    def test_all_ma_features_comparable(self, multi_asset_data):
        """测试所有 MA 特征归一化后跨资产可比"""
        from src.features.loader.talib_feature_wrappers import (
            compute_talib_indicator_from_series,
        )

        # 长周期 MA 允许更大的方差比率（因为趋势跟踪有差异）
        ma_configs = [
            ("SMA", 20, 3.0),
            ("SMA", 50, 4.0),
            ("SMA", 200, 5.0),
            ("EMA", 20, 3.0),
            ("EMA", 50, 4.0),
            ("TEMA", 20, 3.0),
            ("KAMA", 20, 3.0),
        ]

        for indicator, period, max_std_ratio in ma_configs:
            feature_values = {}
            for symbol, df in multi_asset_data.items():
                result = compute_talib_indicator_from_series(
                    indicator_name=indicator,
                    timeperiod=period,
                    output_column=f"{indicator.lower()}_{period}_position",
                    normalize_mode="position",
                    real=df["close"],
                )
                feature_values[symbol] = result[
                    f"{indicator.lower()}_{period}_position"
                ]

            ok, msg = check_cross_asset_comparability(
                feature_values,
                max_std_ratio=max_std_ratio,
                feature_name=f"{indicator}_{period}_position",
            )
            assert ok, f"{indicator}_{period}: {msg}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
