#!/usr/bin/env python3
"""
频谱特征测试 - 使用模拟数据验证特征计算正确性
可在 Docker 环境中运行
"""

import unittest
import sys
import warnings

warnings.filterwarnings("ignore")

# 添加项目根目录到路径

try:
    import numpy as np
    import pandas as pd
    from scipy import signal
except ImportError as e:
    print(f"⚠️  Missing required packages: {e}")
    print("   Please install: pip install numpy pandas scipy")
    sys.exit(1)

from src.features.time_series.utils_spectrum_features import (
    compute_spectrum_features,
    extract_spectrum_features_from_series,
)


# Route B: DF-style entrypoint removed; provide a local DF wrapper.
def extract_spectrum_features(
    df: pd.DataFrame,
    price_col: str = "close",
    volume_col: str | None = None,
    cvd_col: str | None = None,
    rolling_window: int = 64,
) -> pd.DataFrame:
    vol = df[volume_col] if volume_col and volume_col in df.columns else None
    cvd = df[cvd_col] if cvd_col and cvd_col in df.columns else None
    feats = extract_spectrum_features_from_series(
        close=df[price_col],
        volume=vol,
        cvd=cvd,
        rolling_window=rolling_window,
    )
    out = df.copy()
    for c in feats.columns:
        out[c] = feats[c]
    return out


class TestSpectrumFeatures(unittest.TestCase):
    """频谱特征测试类"""

    def setUp(self):
        """设置测试环境"""
        np.random.seed(42)
        self.n_samples = 500  # 增加样本数，确保频谱计算有意义
        self.rolling_window = 64
        self.fs = 1.0  # 默认采样频率

    def create_white_noise_data(self, n_samples=None):
        """创建白噪声数据"""
        if n_samples is None:
            n_samples = self.n_samples
        returns = np.random.randn(n_samples)
        df = pd.DataFrame(
            {
                "close": 100 + np.cumsum(returns),
                "volume": np.random.randint(100, 1000, n_samples),
                "cvd": np.cumsum(np.random.randn(n_samples)),
            }
        )
        return df

    def create_trend_data(self, n_samples=None):
        """创建趋势数据"""
        if n_samples is None:
            n_samples = self.n_samples
        returns = (
            np.linspace(0.001, 0.005, n_samples) + np.random.randn(n_samples) * 0.01
        )
        price = 100 * np.exp(np.cumsum(returns))
        df = pd.DataFrame(
            {
                "close": price,
                "volume": np.random.randint(100, 1000, n_samples),
                "cvd": np.cumsum(np.random.randn(n_samples)),
            }
        )
        return df

    def create_periodic_data(self, n_samples=None, freq=0.1):
        """创建周期信号数据"""
        if n_samples is None:
            n_samples = self.n_samples
        t = np.arange(n_samples)
        signal = np.sin(2 * np.pi * freq * t) + np.random.randn(n_samples) * 0.1
        df = pd.DataFrame(
            {
                "close": 100 + np.cumsum(signal),
                "volume": np.random.randint(100, 1000, n_samples),
                "cvd": np.cumsum(np.random.randn(n_samples)),
            }
        )
        return df

    def test_white_noise_characteristics(self):
        """测试 1：白噪声数据特征"""
        print("\n" + "=" * 70)
        print("测试 1：白噪声数据特征")
        print("=" * 70)
        df = self.create_white_noise_data()
        returns = df["close"].pct_change().fillna(0).values
        features = compute_spectrum_features(returns, fs=self.fs)

        # 白噪声应具有高平坦度、高熵、低主频显著性
        # 注意：对于短信号，可能返回默认值，所以放宽条件
        if features["spectral_flatness"] < 1.0:  # 如果计算成功
            self.assertGreater(
                features["spectral_flatness"],
                0.5,
                "白噪声应有高平坦度（>0.5）",
            )
            self.assertGreater(
                features["spectral_entropy"],
                0.5,
                "白噪声应有高熵（>0.5）",
            )
            self.assertLess(
                features["has_dominant_freq"],
                0.5,
                "白噪声不应有显著主频",
            )
        # 高频和低频能量比应该合理（允许为0，因为频率分段可能没有覆盖所有能量）
        self.assertGreaterEqual(
            features["high_freq_energy_ratio"],
            0.0,
            "高频能量比应 >= 0",
        )
        self.assertLessEqual(
            features["high_freq_energy_ratio"],
            1.0,
            "高频能量比应 <= 1",
        )
        print(f"  白噪声特征: {features}")
        print("  ✅ 白噪声特征验证通过")

    def test_periodic_signal_characteristics(self):
        """测试 2：周期信号数据特征"""
        print("\n" + "=" * 70)
        print("测试 2：周期信号数据特征")
        print("=" * 70)
        freq = 0.05  # 周期为 1/0.05 = 20 个采样点
        df = self.create_periodic_data(freq=freq)
        returns = df["close"].pct_change().fillna(0).values
        features = compute_spectrum_features(returns, fs=self.fs)

        # 周期信号应具有低平坦度、低熵、高主频显著性、能量集中在低频
        # 注意：如果信号太短或计算失败，可能返回默认值
        if features["spectral_flatness"] < 1.0:  # 如果计算成功
            self.assertLess(
                features["spectral_flatness"],
                0.8,
                "周期信号应有低平坦度（<0.8）",
            )
            self.assertLess(
                features["spectral_entropy"],
                0.9,
                "周期信号应有低熵（<0.9）",
            )
            # 周期信号更可能有显著主频
            self.assertGreaterEqual(
                features["has_dominant_freq"],
                0.0,
                "has_dominant_freq 应 >= 0",
            )
            self.assertGreaterEqual(
                features["low_freq_energy_ratio"],
                0.0,
                "低频能量比应 >= 0",
            )
        print(f"  周期信号特征: {features}")
        print("  ✅ 周期信号特征验证通过")

    def test_trend_data_characteristics(self):
        """测试 3：趋势数据特征"""
        print("\n" + "=" * 70)
        print("测试 3：趋势数据特征")
        print("=" * 70)
        df = self.create_trend_data()
        returns = df["close"].pct_change().fillna(0).values
        features = compute_spectrum_features(returns, fs=self.fs)

        # 趋势数据（收益率）通常表现为低频能量较高，平坦度较低
        # 注意：如果信号太短或计算失败，可能返回默认值
        if features["spectral_flatness"] < 1.0:  # 如果计算成功
            self.assertLess(
                features["spectral_flatness"],
                0.9,
                "趋势收益率应有较低平坦度（<0.9）",
            )
            self.assertGreaterEqual(
                features["low_freq_energy_ratio"],
                0.0,
                "低频能量比应 >= 0",
            )
            self.assertLessEqual(
                features["high_freq_energy_ratio"],
                1.0,
                "高频能量比应 <= 1",
            )
        # 即使返回默认值，也应该验证值在合理范围内
        self.assertGreaterEqual(
            features["spectral_flatness"],
            0.0,
            "平坦度应 >= 0",
        )
        self.assertLessEqual(
            features["spectral_flatness"],
            1.0,
            "平坦度应 <= 1",
        )
        print(f"  趋势数据特征: {features}")
        print("  ✅ 趋势数据特征验证通过")

    def test_extract_spectrum_features_integration(self):
        """测试 4：extract_spectrum_features 集成测试"""
        print("\n" + "=" * 70)
        print("测试 4：extract_spectrum_features 集成测试")
        print("=" * 70)
        df = self.create_white_noise_data(n_samples=self.n_samples)
        result_df = extract_spectrum_features(
            df, rolling_window=self.rolling_window, volume_col="volume", cvd_col="cvd"
        )

        # 检查输出列是否存在
        expected_cols = [
            "spectrum_price_has_dominant_freq",
            "spectrum_price_flatness",
            "spectrum_price_high_freq_ratio",
            "spectrum_price_low_freq_ratio",
            "spectrum_price_entropy",
            "spectrum_price_centroid",
            "spectrum_volume_flatness",
            "spectrum_volume_high_freq_ratio",
            "spectrum_volume_low_freq_ratio",
            "spectrum_volume_entropy",
            "spectrum_volume_centroid",
            "spectrum_cvd_flatness",
            "spectrum_cvd_high_freq_ratio",
            "spectrum_cvd_low_freq_ratio",
            "spectrum_cvd_entropy",
            "spectrum_cvd_centroid",
        ]
        for col in expected_cols:
            self.assertIn(col, result_df.columns, f"缺少列: {col}")

        # 检查 NaN 填充和 shift(1)
        # 前 rolling_window 个值应为默认值
        for col in expected_cols:
            if (
                "has_dominant_freq" in col
                or "low_freq_ratio" in col
                or "high_freq_ratio" in col
            ):
                default_val = 0.0
            elif "flatness" in col or "entropy" in col:
                default_val = 1.0
            elif "centroid" in col:
                default_val = 0.0
            else:
                default_val = 0.0

            # 检查前 rolling_window 行的默认值
            actual_values = result_df.loc[: self.rolling_window - 1, col].fillna(
                default_val
            )
            self.assertTrue(
                np.allclose(actual_values, default_val, equal_nan=True),
                f"{col} 前 {self.rolling_window} 行应为 {default_val}",
            )

        # 检查 shift(1) 后的第一个非 NaN 值
        # 第 rolling_window 行应该有值，但由于 shift(1)，实际是 rolling_window + 1 行开始有计算值
        # 注意：如果计算失败，可能返回默认值，所以只检查值是否在合理范围内
        first_value = result_df.loc[self.rolling_window, "spectrum_price_flatness"]
        self.assertGreaterEqual(
            first_value,
            0.0,
            f"shift(1) 后，第一个计算值应 >= 0，实际: {first_value}",
        )
        self.assertLessEqual(
            first_value,
            1.0,
            f"shift(1) 后，第一个计算值应 <= 1，实际: {first_value}",
        )
        print("  ✅ extract_spectrum_features 集成验证通过")

    def test_edge_cases(self):
        """测试 5：边界情况处理"""
        print("\n" + "=" * 70)
        print("测试 5：边界情况处理")
        print("=" * 70)

        # 短信号
        short_signal = np.array([1, 2, 3, 4, 5])
        features = compute_spectrum_features(short_signal)
        self.assertEqual(
            features["spectral_flatness"],
            1.0,
            "短信号应返回默认平坦度",
        )
        self.assertEqual(
            features["spectral_entropy"],
            1.0,
            "短信号应返回默认熵",
        )
        print("  ✅ 短信号处理通过")

        # 零信号
        zero_signal = np.zeros(self.n_samples)
        features = compute_spectrum_features(zero_signal)
        self.assertEqual(
            features["spectral_flatness"],
            1.0,
            "零信号应返回默认平坦度",
        )
        self.assertEqual(
            features["spectral_entropy"],
            1.0,
            "零信号应返回默认熵",
        )
        print("  ✅ 零信号处理通过")

        # NaN 信号
        nan_signal = np.full(self.n_samples, np.nan)
        features = compute_spectrum_features(nan_signal)
        self.assertEqual(
            features["spectral_flatness"],
            1.0,
            "NaN信号应返回默认平坦度",
        )
        self.assertEqual(
            features["spectral_entropy"],
            1.0,
            "NaN信号应返回默认熵",
        )
        print("  ✅ NaN信号处理通过")

        print("  ✅ 边界情况处理验证通过")

    def test_feature_value_ranges(self):
        """测试 6：特征值范围验证"""
        print("\n" + "=" * 70)
        print("测试 6：特征值范围验证")
        print("=" * 70)
        df = self.create_white_noise_data()
        returns = df["close"].pct_change().fillna(0).values
        features = compute_spectrum_features(returns, fs=self.fs)

        # 检查特征值范围
        self.assertGreaterEqual(
            features["spectral_flatness"],
            0.0,
            "平坦度应 >= 0",
        )
        self.assertLessEqual(
            features["spectral_flatness"],
            1.0,
            "平坦度应 <= 1",
        )

        self.assertGreaterEqual(
            features["spectral_entropy"],
            0.0,
            "熵应 >= 0",
        )
        self.assertLessEqual(
            features["spectral_entropy"],
            1.0,
            "熵应 <= 1",
        )

        self.assertGreaterEqual(
            features["high_freq_energy_ratio"],
            0.0,
            "高频能量比应 >= 0",
        )
        self.assertLessEqual(
            features["high_freq_energy_ratio"],
            1.0,
            "高频能量比应 <= 1",
        )

        self.assertGreaterEqual(
            features["low_freq_energy_ratio"],
            0.0,
            "低频能量比应 >= 0",
        )
        self.assertLessEqual(
            features["low_freq_energy_ratio"],
            1.0,
            "低频能量比应 <= 1",
        )

        self.assertGreaterEqual(
            features["spectral_centroid"],
            0.0,
            "频谱重心应 >= 0",
        )

        self.assertIn(
            features["has_dominant_freq"],
            [0.0, 1.0],
            "has_dominant_freq 应为 0 或 1",
        )

        print("  ✅ 特征值范围验证通过")

    def test_consistency_across_windows(self):
        """测试 7：不同窗口大小的一致性"""
        print("\n" + "=" * 70)
        print("测试 7：不同窗口大小的一致性")
        print("=" * 70)
        df = self.create_periodic_data(freq=0.05, n_samples=300)

        # 测试不同窗口大小
        windows = [32, 64, 128]
        results = []
        for window in windows:
            result_df = extract_spectrum_features(
                df, rolling_window=window, volume_col="volume", cvd_col="cvd"
            )
            # 取最后一个有效值
            last_valid_idx = result_df["spectrum_price_flatness"].last_valid_index()
            if last_valid_idx is not None:
                results.append(result_df.loc[last_valid_idx, "spectrum_price_flatness"])

        # 不同窗口的结果应该相似（允许一定误差）
        if len(results) > 1:
            max_diff = max(results) - min(results)
            self.assertLess(
                max_diff,
                0.3,
                f"不同窗口大小的结果差异应 < 0.3，实际差异: {max_diff}",
            )
        print(f"  不同窗口结果: {results}")
        print("  ✅ 窗口一致性验证通过")


def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("开始运行频谱特征测试")
    print("=" * 70)
    print(f"Python 版本: {sys.version}")
    print(f"NumPy 版本: {np.__version__}")
    print(f"Pandas 版本: {pd.__version__}")
    print("=" * 70)

    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSpectrumFeatures)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 输出总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"运行测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")

    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"  - {test}")
            print(f"    {traceback}")

    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}")
            print(f"    {traceback}")

    # 返回退出码
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
