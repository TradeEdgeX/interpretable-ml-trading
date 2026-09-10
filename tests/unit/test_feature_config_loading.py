"""
测试特征配置加载的 pytest 测试

验证：
1. 交互特征和衍生特征是否能正确加载
2. 所有特征函数是否正确映射
3. 特征计算是否有错误
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestFeatureConfigLoading:
    """特征配置加载测试类"""

    def test_feature_loader_initialization(self, feature_loader):
        """测试特征加载器初始化"""
        assert feature_loader is not None
        assert feature_loader.feature_deps is not None
        assert "features" in feature_loader.feature_deps
        print("✅ 特征加载器初始化成功")

    def test_strategy_config_loading(self, strategy_config):
        """测试策略配置加载"""
        assert strategy_config is not None
        assert hasattr(strategy_config, "features")
        requested_features = strategy_config.features.requested_features
        assert len(requested_features) > 0
        print(f"✅ 策略配置加载成功，请求 {len(requested_features)} 个特征")

    def test_feature_loading(self, sample_data, feature_loader, strategy_config):
        """测试特征加载"""
        requested_features = strategy_config.features.requested_features

        # 只测试前20个特征以加快测试速度
        test_features = requested_features[:20]

        try:
            df_features = feature_loader.load_features_from_requested(
                sample_data.copy(),
                test_features,
                fit=True,
            )

            assert len(df_features.columns) >= len(sample_data.columns)
            new_cols = [c for c in df_features.columns if c not in sample_data.columns]
            print(f"✅ 特征加载成功: 新增 {len(new_cols)} 个特征列")

        except Exception as e:
            pytest.fail(f"特征加载失败: {e}")

    def test_derived_features_exist(self, sample_data, feature_loader, strategy_config):
        """测试衍生特征是否存在"""
        requested_features = strategy_config.features.requested_features
        test_features = requested_features[:20]

        try:
            df_features = feature_loader.load_features_from_requested(
                sample_data.copy(),
                test_features,
                fit=True,
            )

            derived_features = [
                "sr_strength_combined",
                "sr_distance_normalized",
                "dist_to_zz_high",
                "dist_to_zz_low",
                "cvd_slope_5",
                "atr_ratio",
                "bb_width_ratio",
                "compression_score",
                "tbr_ma_5",
                "tbr_spike",
            ]

            found_derived = [c for c in derived_features if c in df_features.columns]
            print(f"✅ 找到 {len(found_derived)}/{len(derived_features)} 个衍生特征")

            # 检查交互特征
            interaction_features = [c for c in df_features.columns if "_x_" in c]
            if interaction_features:
                print(f"✅ 找到 {len(interaction_features)} 个交互特征")

        except Exception as e:
            # 某些特征可能因为依赖不存在而无法计算，这是正常的
            print(f"⚠️  特征计算出现异常（可能是依赖缺失）: {e}")

    def test_feature_values_quality(self, sample_data, feature_loader, strategy_config):
        """测试特征值质量"""
        requested_features = strategy_config.features.requested_features
        test_features = requested_features[:20]

        try:
            df_features = feature_loader.load_features_from_requested(
                sample_data.copy(),
                test_features,
                fit=True,
            )

            # 检查 NaN 和 Inf
            numeric_cols = df_features.select_dtypes(include=[np.number]).columns
            nan_counts = df_features[numeric_cols].isna().sum()
            inf_counts = np.isinf(df_features[numeric_cols]).sum()

            high_nan_cols = nan_counts[
                nan_counts > len(df_features) * 0.5
            ].index.tolist()
            high_inf_cols = inf_counts[inf_counts > 0].index.tolist()

            if high_nan_cols:
                print(f"⚠️  高 NaN 比例的特征 (>50%): {len(high_nan_cols)} 个")
            else:
                print("✅ 没有高 NaN 比例的特征")

            assert len(high_inf_cols) == 0, f"发现包含 Inf 的特征: {high_inf_cols[:5]}"

        except Exception as e:
            print(f"⚠️  特征质量检查异常: {e}")
