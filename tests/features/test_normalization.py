"""
特征归一化工具函数测试

测试内容：
1. 基础归一化功能（zscore, minmax, robust_minmax）
2. 滚动归一化功能（防止未来信息泄露）
3. 多资产分组归一化
4. NaN值处理
5. 边界情况处理（常数序列、空序列等）
6. 批量归一化功能
"""

import unittest
import pandas as pd
import numpy as np

# 添加项目根目录到路径

from src.features.time_series.utils_normalization import (
    normalize_series,
    normalize_by_group,
    normalize_dataframe,
)


class TestNormalizeSeries(unittest.TestCase):
    """测试 normalize_series 函数"""

    def test_zscore_basic(self):
        """测试基本的zscore归一化"""
        x = np.array([1, 2, 3, 4, 5])
        result = normalize_series(x, method="zscore")

        # 检查均值接近0
        self.assertAlmostEqual(np.mean(result[~np.isnan(result)]), 0, places=5)
        # 检查标准差接近1
        self.assertAlmostEqual(np.std(result[~np.isnan(result)]), 1, places=5)

    def test_zscore_with_nan(self):
        """测试zscore归一化处理NaN值"""
        x = np.array([1, 2, np.nan, 4, 5])
        result = normalize_series(x, method="zscore")

        # NaN应该被保留
        self.assertTrue(np.isnan(result[2]))
        # 其他值应该被归一化
        self.assertFalse(np.isnan(result[0]))
        self.assertFalse(np.isnan(result[1]))

    def test_minmax_basic(self):
        """测试基本的minmax归一化"""
        x = np.array([10, 20, 30, 40, 50])
        result = normalize_series(x, method="minmax")

        # 最小值应该接近0
        self.assertAlmostEqual(np.min(result[~np.isnan(result)]), 0, places=5)
        # 最大值应该接近1
        self.assertAlmostEqual(np.max(result[~np.isnan(result)]), 1, places=5)
        # 所有值应该在[0, 1]范围内
        self.assertTrue(
            np.all((result[~np.isnan(result)] >= 0) & (result[~np.isnan(result)] <= 1))
        )

    def test_robust_minmax(self):
        """测试robust minmax归一化（对异常值鲁棒）"""
        # 创建包含异常值的数据
        x = np.array([1, 2, 3, 4, 5, 100])  # 100是异常值
        result = normalize_series(x, method="robust_minmax")

        # 所有值应该在[0, 1]范围内
        self.assertTrue(
            np.all((result[~np.isnan(result)] >= 0) & (result[~np.isnan(result)] <= 1))
        )

    def test_constant_sequence(self):
        """测试常数序列（std=0或max=min）"""
        x = np.array([5, 5, 5, 5, 5])
        result = normalize_series(x, method="zscore")

        # 常数序列应该返回全零
        self.assertTrue(np.allclose(result, 0))

    def test_all_nan(self):
        """测试全部为NaN的情况"""
        x = np.array([np.nan, np.nan, np.nan])
        result = normalize_series(x, method="zscore")

        # 应该返回全零
        self.assertTrue(np.allclose(result, 0))


class TestNormalizeByGroup(unittest.TestCase):
    """测试 normalize_by_group 函数"""

    def setUp(self):
        """创建测试数据"""
        np.random.seed(42)
        n_samples = 100

        # 创建多资产数据
        self.df_multi = pd.DataFrame(
            {
                "_symbol": ["BTC"] * 50 + ["ETH"] * 50,
                "feature": np.concatenate(
                    [
                        np.random.randn(50).cumsum() + 50000,  # BTC价格水平
                        np.random.randn(50).cumsum() + 3000,  # ETH价格水平
                    ]
                ),
            }
        )

        # 创建单资产数据
        self.df_single = pd.DataFrame(
            {
                "feature": np.random.randn(n_samples).cumsum() + 100,
            }
        )

    def test_global_zscore_multi_asset(self):
        """测试多资产全局zscore归一化"""
        result = normalize_by_group(
            self.df_multi,
            value_col="feature",
            group_col="_symbol",
            method="zscore",
            window=None,  # 全局归一化
        )

        # 检查每个资产的归一化结果
        btc_result = result[self.df_multi["_symbol"] == "BTC"]
        eth_result = result[self.df_multi["_symbol"] == "ETH"]

        # 每个资产内部应该均值接近0，标准差接近1
        self.assertAlmostEqual(btc_result.mean(), 0, places=2)
        self.assertAlmostEqual(eth_result.mean(), 0, places=2)
        self.assertAlmostEqual(btc_result.std(), 1, places=1)
        self.assertAlmostEqual(eth_result.std(), 1, places=1)

    def test_rolling_zscore_multi_asset(self):
        """测试多资产滚动zscore归一化"""
        window = 20
        result = normalize_by_group(
            self.df_multi,
            value_col="feature",
            group_col="_symbol",
            method="zscore",
            window=window,
        )

        # 结果应该不为空
        self.assertEqual(len(result), len(self.df_multi))
        # 前window-1行可能为NaN（如果fillna=False），但fillna=True时应该被填充
        self.assertFalse(result.isna().any())

    def test_rolling_vs_global_difference(self):
        """测试滚动归一化与全局归一化的差异"""
        # 全局归一化
        global_result = normalize_by_group(
            self.df_single,
            value_col="feature",
            method="zscore",
            window=None,
        )

        # 滚动归一化
        rolling_result = normalize_by_group(
            self.df_single,
            value_col="feature",
            method="zscore",
            window=20,
        )

        # 两者应该不同（滚动归一化是动态的）
        self.assertFalse(np.allclose(global_result.values, rolling_result.values))

    def test_rolling_minmax(self):
        """测试滚动minmax归一化"""
        result = normalize_by_group(
            self.df_single,
            value_col="feature",
            method="minmax",
            window=20,
        )

        # 所有值应该在[0, 1]范围内（因为clip=True）
        self.assertTrue(np.all((result >= 0) & (result <= 1)))

    def test_rolling_robust_minmax(self):
        """测试滚动robust minmax归一化"""
        # 添加异常值
        df_with_outlier = self.df_single.copy()
        df_with_outlier.loc[50, "feature"] = 100000  # 异常值

        result = normalize_by_group(
            df_with_outlier,
            value_col="feature",
            method="robust_minmax",
            window=20,
        )

        # 所有值应该在[0, 1]范围内
        self.assertTrue(np.all((result >= 0) & (result <= 1)))

    def test_clip_parameter(self):
        """测试clip参数"""
        # 创建极端值数据
        df_extreme = pd.DataFrame(
            {
                "feature": np.random.randn(100) * 10,  # 标准差为10
            }
        )

        # 不裁剪
        result_no_clip = normalize_by_group(
            df_extreme,
            value_col="feature",
            method="zscore",
            window=None,
            clip=False,
        )

        # 裁剪
        result_clip = normalize_by_group(
            df_extreme,
            value_col="feature",
            method="zscore",
            window=None,
            clip=True,
        )

        # 裁剪后的值应该在[-3, 3]范围内
        self.assertTrue(np.all((result_clip >= -3) & (result_clip <= 3)))
        # 不裁剪的结果可能超出[-3, 3]
        self.assertTrue(np.any((result_no_clip < -3) | (result_no_clip > 3)))

    def test_fillna_parameter(self):
        """测试fillna参数"""
        df_with_nan = self.df_single.copy()
        df_with_nan.loc[10:15, "feature"] = np.nan

        # fillna=True
        result_filled = normalize_by_group(
            df_with_nan,
            value_col="feature",
            method="zscore",
            window=None,
            fillna=True,
        )

        # fillna=False
        result_not_filled = normalize_by_group(
            df_with_nan,
            value_col="feature",
            method="zscore",
            window=None,
            fillna=False,
        )

        # fillna=True时不应该有NaN
        self.assertFalse(result_filled.isna().any())
        # fillna=False时应该有NaN
        self.assertTrue(result_not_filled.isna().any())

    def test_single_asset_no_group(self):
        """测试单资产（无分组列）"""
        result = normalize_by_group(
            self.df_single,
            value_col="feature",
            method="zscore",
            window=None,
        )

        # 应该成功归一化
        self.assertEqual(len(result), len(self.df_single))
        self.assertAlmostEqual(result.mean(), 0, places=3)
        self.assertAlmostEqual(result.std(), 1, places=1)


class TestNormalizeDataframe(unittest.TestCase):
    """测试 normalize_dataframe 函数"""

    def setUp(self):
        """创建测试数据"""
        np.random.seed(42)
        n_samples = 100

        self.df = pd.DataFrame(
            {
                "_symbol": ["BTC"] * 50 + ["ETH"] * 50,
                "feature1": np.random.randn(n_samples).cumsum() + 100,
                "feature2": np.random.randn(n_samples).cumsum() + 200,
                "feature3": np.random.randn(n_samples).cumsum() + 300,
            }
        )

    def test_batch_normalization(self):
        """测试批量归一化"""
        result = normalize_dataframe(
            self.df,
            value_cols=["feature1", "feature2", "feature3"],
            group_col="_symbol",
            method="zscore",
            window=None,
        )

        # 应该创建归一化后的列
        self.assertIn("feature1", result.columns)
        self.assertIn("feature2", result.columns)
        self.assertIn("feature3", result.columns)

        # 每个特征应该被归一化
        for col in ["feature1", "feature2", "feature3"]:
            btc_result = result[result["_symbol"] == "BTC"][col]
            self.assertAlmostEqual(btc_result.mean(), 0, places=3)

    def test_suffix_parameter(self):
        """测试suffix参数"""
        result = normalize_dataframe(
            self.df,
            value_cols=["feature1", "feature2"],
            method="zscore",
            suffix="_zscore",
        )

        # 应该创建带后缀的列
        self.assertIn("feature1_zscore", result.columns)
        self.assertIn("feature2_zscore", result.columns)
        # 原始列应该保留
        self.assertIn("feature1", result.columns)
        self.assertIn("feature2", result.columns)

    def test_batch_rolling_normalization(self):
        """测试批量滚动归一化"""
        result = normalize_dataframe(
            self.df,
            value_cols=["feature1", "feature2"],
            group_col="_symbol",
            method="zscore",
            window=20,
            suffix="_rolling_zscore",
        )

        # 应该创建滚动归一化的列
        self.assertIn("feature1_rolling_zscore", result.columns)
        self.assertIn("feature2_rolling_zscore", result.columns)
        # 不应该有NaN（fillna=True）
        self.assertFalse(result["feature1_rolling_zscore"].isna().any())


class TestGlobalNormalizationWarning(unittest.TestCase):
    """测试全局归一化警告"""

    def test_global_normalization_warning(self):
        """测试全局归一化会触发警告"""
        import warnings

        df = pd.DataFrame(
            {
                "feature": np.random.randn(100).cumsum() + 100,
            }
        )

        # 应该触发警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            normalize_by_group(
                df,
                value_col="feature",
                method="zscore",
                window=None,  # 全局归一化
                warn_global=True,
            )
            # 应该有一个警告
            self.assertGreater(len(w), 0)
            self.assertTrue(
                any("未来信息泄露" in str(warning.message) for warning in w)
            )

    def test_global_normalization_no_warning(self):
        """测试可以关闭警告"""
        import warnings

        df = pd.DataFrame(
            {
                "feature": np.random.randn(100).cumsum() + 100,
            }
        )

        # 不应该触发警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            normalize_by_group(
                df,
                value_col="feature",
                method="zscore",
                window=None,
                warn_global=False,  # 关闭警告
            )
            # 不应该有警告
            self.assertEqual(len(w), 0)


class TestGlobalNormalizationWarning(unittest.TestCase):
    """测试全局归一化警告"""

    def test_global_normalization_warning(self):
        """测试全局归一化会触发警告"""
        import warnings

        df = pd.DataFrame(
            {
                "feature": np.random.randn(100).cumsum() + 100,
            }
        )

        # 应该触发警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            normalize_by_group(
                df,
                value_col="feature",
                method="zscore",
                window=None,  # 全局归一化
                warn_global=True,
            )
            # 应该有一个警告
            self.assertGreater(len(w), 0)
            self.assertTrue(
                any("未来信息泄露" in str(warning.message) for warning in w)
            )

    def test_global_normalization_no_warning(self):
        """测试可以关闭警告"""
        import warnings

        df = pd.DataFrame(
            {
                "feature": np.random.randn(100).cumsum() + 100,
            }
        )

        # 不应该触发警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            normalize_by_group(
                df,
                value_col="feature",
                method="zscore",
                window=None,
                warn_global=False,  # 关闭警告
            )
            # 不应该有警告
            self.assertEqual(len(w), 0)


class TestRollingNormalizationUseCase(unittest.TestCase):
    """测试滚动归一化的实际用例"""

    def test_rolling_prevents_lookahead_bias(self):
        """
        演示滚动归一化如何防止未来信息泄露

        这是滚动归一化的核心价值：
        - 全局归一化：使用整个数据集（包括未来数据）计算统计量 → 有未来信息泄露
        - 滚动归一化：只使用历史窗口数据计算统计量 → 无未来信息泄露
        """
        # 创建有明显趋势的数据
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "price": np.arange(100) + np.random.randn(100) * 0.1,  # 上升趋势
            }
        )

        # 全局归一化（错误方式 - 有未来信息泄露）
        global_norm = normalize_by_group(
            df,
            value_col="price",
            method="zscore",
            window=None,
        )

        # 滚动归一化（正确方式 - 无未来信息泄露）
        rolling_norm = normalize_by_group(
            df,
            value_col="price",
            method="zscore",
            window=20,  # 只用过去20天的数据
        )

        # 在时间点t，滚动归一化只使用t-19到t的数据
        # 而全局归一化使用了整个数据集（包括t+1到t+100的数据）

        # 验证：前20个数据点，两种方法的结果应该不同
        self.assertFalse(
            np.allclose(
                global_norm.iloc[:20].values, rolling_norm.iloc[:20].values, atol=0.1
            )
        )

        # 验证：滚动归一化的前几个点可能接近0（窗口不足）
        # 但整体趋势应该被保留

    def test_rolling_adapts_to_regime_changes(self):
        """
        演示滚动归一化如何适应市场状态变化

        滚动归一化能够：
        - 适应波动率变化（高波动期和低波动期）
        - 适应趋势变化（上升趋势和下降趋势）
        """
        # 创建有状态变化的数据
        dates = pd.date_range("2020-01-01", periods=200, freq="D")

        # 前100天：低波动
        period1 = np.random.randn(100) * 1 + 100
        # 后100天：高波动
        period2 = np.random.randn(100) * 5 + 100

        df = pd.DataFrame(
            {
                "date": dates,
                "price": np.concatenate([period1, period2]),
            }
        )

        # 滚动归一化（window=30）
        rolling_norm = normalize_by_group(
            df,
            value_col="price",
            method="zscore",
            window=30,
        )

        # 验证：两个时期的归一化结果都应该在合理范围内
        # 因为滚动归一化会适应每个时期的波动率
        period1_norm = rolling_norm.iloc[30:100]  # 跳过前30个（窗口不足）
        period2_norm = rolling_norm.iloc[130:200]  # 跳过前30个（窗口不足）

        # 两个时期的归一化结果都应该均值接近0，标准差接近1
        self.assertAlmostEqual(period1_norm.mean(), 0, places=0)
        self.assertAlmostEqual(period2_norm.mean(), 0, places=0)
        self.assertAlmostEqual(period1_norm.std(), 1, places=1)
        self.assertAlmostEqual(period2_norm.std(), 1, places=0)


if __name__ == "__main__":
    unittest.main()
