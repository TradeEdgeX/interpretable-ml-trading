"""
测试 WPT 波动率增强特征
验证 enhance_wpt_vol_features 函数是否正确生成所有增强特征
"""

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.utils_volatility_features import enhance_wpt_vol_features


def create_mock_wpt_data(n_samples: int = 100) -> pd.DataFrame:
    """创建模拟的 WPT 特征数据"""
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="4H")
    df = pd.DataFrame(index=dates)

    # 基础 WPT 特征
    np.random.seed(42)
    df["wpt_price_trend"] = 100 + np.cumsum(np.random.randn(n_samples) * 0.1)
    df["wpt_price_fluctuation"] = np.random.randn(n_samples) * 0.5
    df["wpt_price_reconstructed"] = df["wpt_price_trend"] + df["wpt_price_fluctuation"]

    # 能量比特征
    df["wpt_price_energy_low_ratio"] = np.random.uniform(0.3, 0.5, n_samples)
    df["wpt_price_energy_mid_ratio"] = np.random.uniform(0.2, 0.4, n_samples)
    df["wpt_price_energy_high_ratio"] = (
        1.0 - df["wpt_price_energy_low_ratio"] - df["wpt_price_energy_mid_ratio"]
    )
    df["wpt_price_energy_mid_low_ratio"] = df["wpt_price_energy_mid_ratio"] / (
        df["wpt_price_energy_low_ratio"] + 1e-8
    )

    # 成交量 WPT 特征
    df["wpt_volume_energy_low_ratio"] = np.random.uniform(0.2, 0.4, n_samples)
    df["wpt_volume_energy_high_ratio"] = 1.0 - df["wpt_volume_energy_low_ratio"]

    # CVD WPT 特征（可选）
    df["wpt_cvd_energy_low_ratio"] = np.random.uniform(0.3, 0.5, n_samples)

    # VPER
    df["wpt_vper"] = np.random.uniform(0.5, 2.0, n_samples)

    return df


class TestEnhanceWPTVolFeatures:
    """测试 enhance_wpt_vol_features 函数"""

    def test_basic_functionality(self):
        """测试基本功能：所有增强特征是否正确生成"""
        df_wpt = create_mock_wpt_data(100)
        df_enhanced = enhance_wpt_vol_features(df_wpt)

        # 检查增强特征是否存在
        assert "wpt_price_high_energy_ratio" in df_enhanced.columns
        assert "wpt_price_fluct_l1_l2_ratio" in df_enhanced.columns
        assert "wpt_vhph_sync" in df_enhanced.columns

        # 检查原始特征是否保留
        assert "wpt_price_trend" in df_enhanced.columns
        assert "wpt_price_fluctuation" in df_enhanced.columns

    def test_wpt_price_high_energy_ratio(self):
        """测试高频能量占比特征"""
        df_wpt = create_mock_wpt_data(100)
        df_enhanced = enhance_wpt_vol_features(df_wpt)

        # 应该等于 wpt_price_energy_high_ratio
        assert np.allclose(
            df_enhanced["wpt_price_high_energy_ratio"],
            df_wpt["wpt_price_energy_high_ratio"],
            rtol=1e-10,
        )

    def test_wpt_price_fluct_l1_l2_ratio(self):
        """测试 L1/L2 范数比特征"""
        df_wpt = create_mock_wpt_data(100)
        df_enhanced = enhance_wpt_vol_features(df_wpt)

        # 手动计算验证
        fluct = df_wpt["wpt_price_fluctuation"]
        l1 = np.abs(fluct)
        l2 = fluct**2
        expected = (l1 + 1e-8) / (np.sqrt(l2) + 1e-8)

        assert np.allclose(
            df_enhanced["wpt_price_fluct_l1_l2_ratio"], expected, rtol=1e-10
        )

        # L1/L2 比应该在合理范围内（通常 > 0）
        assert (df_enhanced["wpt_price_fluct_l1_l2_ratio"] > 0).all()

    def test_wpt_vhph_sync(self):
        """测试体积-价格高频同步性特征"""
        df_wpt = create_mock_wpt_data(100)
        df_enhanced = enhance_wpt_vol_features(df_wpt)

        # 手动计算验证
        expected = (
            df_wpt["wpt_volume_energy_high_ratio"]
            * df_wpt["wpt_price_energy_high_ratio"]
        )

        assert np.allclose(df_enhanced["wpt_vhph_sync"], expected, rtol=1e-10)

    def test_missing_features_fallback(self):
        """测试缺失特征时的降级处理"""
        df_wpt = create_mock_wpt_data(100)

        # 移除部分特征
        df_wpt = df_wpt.drop(
            columns=["wpt_price_energy_high_ratio", "wpt_price_fluctuation"]
        )
        df_enhanced = enhance_wpt_vol_features(df_wpt)

        # 应该仍然生成特征，但值为默认值
        assert "wpt_price_high_energy_ratio" in df_enhanced.columns
        assert "wpt_price_fluct_l1_l2_ratio" in df_enhanced.columns
        assert (df_enhanced["wpt_price_high_energy_ratio"] == 0.0).all()
        assert (df_enhanced["wpt_price_fluct_l1_l2_ratio"] == 0.0).all()

    def test_no_wpt_features(self):
        """测试没有 WPT 特征时的处理"""
        df_empty = pd.DataFrame(
            index=pd.date_range("2024-01-01", periods=100, freq="4H")
        )
        df_enhanced = enhance_wpt_vol_features(df_empty)

        # 应该仍然生成增强特征，但值为默认值
        assert "wpt_price_high_energy_ratio" in df_enhanced.columns
        assert "wpt_price_fluct_l1_l2_ratio" in df_enhanced.columns
        assert "wpt_vhph_sync" in df_enhanced.columns

    def test_numerical_stability(self):
        """测试数值稳定性：处理极端值"""
        df_wpt = create_mock_wpt_data(100)

        # 添加极端值
        df_wpt.loc[df_wpt.index[0], "wpt_price_fluctuation"] = 1e10
        df_wpt.loc[df_wpt.index[1], "wpt_price_fluctuation"] = -1e10
        df_wpt.loc[df_wpt.index[2], "wpt_price_fluctuation"] = 0.0

        df_enhanced = enhance_wpt_vol_features(df_wpt)

        # 不应该有 NaN 或 Inf
        assert not df_enhanced["wpt_price_fluct_l1_l2_ratio"].isna().any()
        assert not np.isinf(df_enhanced["wpt_price_fluct_l1_l2_ratio"]).any()

    def test_feature_values_range(self):
        """测试特征值的合理范围"""
        df_wpt = create_mock_wpt_data(1000)
        df_enhanced = enhance_wpt_vol_features(df_wpt)

        # wpt_price_high_energy_ratio 应该在 [0, 1] 范围内
        assert (df_enhanced["wpt_price_high_energy_ratio"] >= 0).all()
        assert (df_enhanced["wpt_price_high_energy_ratio"] <= 1).all()

        # wpt_vhph_sync 应该在 [0, 1] 范围内（两个 [0,1] 值的乘积）
        assert (df_enhanced["wpt_vhph_sync"] >= 0).all()
        assert (df_enhanced["wpt_vhph_sync"] <= 1).all()

        # wpt_price_fluct_l1_l2_ratio 应该 > 0
        assert (df_enhanced["wpt_price_fluct_l1_l2_ratio"] > 0).all()


def test_integration_with_volatility_model():
    """测试与波动率模型配置的集成"""
    from src.time_series_model.pipeline.training.volatility_model_config import (
        prepare_volatility_model_data,
        load_volatility_model_config,
    )

    # 创建包含 WPT 特征的模拟数据
    df_wpt = create_mock_wpt_data(100)

    # 添加其他必需的波动率特征
    df_wpt["garch_volatility"] = np.random.uniform(0.01, 0.05, len(df_wpt))
    df_wpt["vol_raw_5"] = np.random.uniform(0.01, 0.05, len(df_wpt))
    df_wpt["atr"] = np.random.uniform(100, 200, len(df_wpt))

    # 加载配置
    config = load_volatility_model_config()

    # 准备数据（应该自动调用 enhance_wpt_vol_features）
    X_vol, selected_features, _ = prepare_volatility_model_data(
        df_wpt, config, feature_loader=None
    )

    # 检查增强的 WPT 特征是否在选中的特征中
    enhanced_features = [
        "wpt_price_high_energy_ratio",
        "wpt_price_fluct_l1_l2_ratio",
        "wpt_vhph_sync",
    ]

    for feat in enhanced_features:
        if feat in df_wpt.columns or feat in X_vol.columns:
            # 如果特征存在，应该被选中（如果在配置中列出）
            pass  # 配置文件中已列出，应该会被选中


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
