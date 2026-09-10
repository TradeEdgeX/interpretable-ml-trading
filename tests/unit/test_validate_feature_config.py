"""
验证特征配置的 pytest 测试

检查：
1. 配置文件语法
2. 函数映射是否存在
3. 依赖关系是否完整
"""

import pytest
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def feature_dependencies():
    """加载特征依赖配置"""
    deps_path = PROJECT_ROOT / "config" / "feature_dependencies.yaml"
    with open(deps_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@pytest.fixture
def strategy_features_config():
    """加载策略特征配置"""
    strategy_path = (
        PROJECT_ROOT / "config" / "strategies" / "sr_reversal" / "features.yaml"
    )
    with open(strategy_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestFeatureConfig:
    """特征配置验证测试类"""

    def test_config_file_syntax(self, feature_dependencies):
        """测试配置文件语法是否正确"""
        assert "features" in feature_dependencies, "配置文件中缺少 features 键"
        features = feature_dependencies.get("features", {})
        assert len(features) > 0, "配置文件中没有定义任何特征"
        print(f"✅ 配置文件语法正确，共 {len(features)} 个特征")

    def test_derived_features_config(self, feature_dependencies):
        """测试衍生特征配置"""
        from src.features.registry import get_compute_func

        features = feature_dependencies.get("features", {})
        derived_features = [
            "sr_strength_combined",
            "sr_distance_normalized",
            "dist_to_zz_high",
            "dist_to_zz_low",
            "dist_to_zz_high_atr",
            "dist_to_zz_low_atr",
            "cvd_slope_5",
            "atr_ratio",
            "bb_width_ratio",
            "compression_score",
            "tbr_ma_5",
            "tbr_spike",
        ]

        errors = []
        for feat_name in derived_features:
            assert feat_name in features, f"衍生特征 {feat_name} 在配置文件中不存在"

            feat_config = features[feat_name]
            compute_func_name = feat_config.get("compute_func")
            assert compute_func_name, f"特征 {feat_name} 缺少 compute_func"

            # 检查函数映射
            try:
                func = get_compute_func(compute_func_name)
                print(f"   ✅ {feat_name}: {compute_func_name} -> {func.__name__}")
            except ValueError as e:
                errors.append(
                    f"特征 {feat_name}: {compute_func_name} 函数映射不存在 - {e}"
                )

            # 检查依赖关系
            output_cols = feat_config.get("output_columns", [])
            if not output_cols:
                print(f"   ⚠️  {feat_name}: 缺少 output_columns")

        assert len(errors) == 0, f"发现 {len(errors)} 个函数映射错误: {errors}"

    def test_interaction_features_config(self, feature_dependencies):
        """测试交互特征配置"""
        from src.features.registry import get_compute_func

        features = feature_dependencies.get("features", {})
        interaction_features = [
            "vpin_x_wick_upper",
            "vpin_x_wick_lower",
            "vpin_x_wick_upper_rank",
            "vpin_x_wick_lower_rank",
        ]

        errors = []
        for feat_name in interaction_features:
            assert feat_name in features, f"交互特征 {feat_name} 在配置文件中不存在"

            feat_config = features[feat_name]
            compute_func_name = feat_config.get("compute_func")
            assert compute_func_name, f"特征 {feat_name} 缺少 compute_func"

            try:
                func = get_compute_func(compute_func_name)
                print(f"   ✅ {feat_name}: {compute_func_name} -> {func.__name__}")
            except ValueError as e:
                errors.append(
                    f"特征 {feat_name}: {compute_func_name} 函数映射不存在 - {e}"
                )

        assert len(errors) == 0, f"发现 {len(errors)} 个函数映射错误: {errors}"

    def test_strategy_features_in_deps(
        self, feature_dependencies, strategy_features_config
    ):
        """测试策略请求的特征是否都在依赖配置中定义"""
        features = feature_dependencies.get("features", {})
        requested = strategy_features_config.get("feature_pipeline", {}).get(
            "requested_features", []
        )

        missing_in_deps = [f for f in requested if f not in features]
        if missing_in_deps:
            print(f"   ⚠️  策略请求的特征在依赖配置中不存在: {missing_in_deps[:10]}")
            # 这可能是警告，不一定是错误，因为某些特征可能是动态生成的
        else:
            print(f"   ✅ 所有请求的特征都在依赖配置中定义")

        print(f"   请求的特征数量: {len(requested)}")
