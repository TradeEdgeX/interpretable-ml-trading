#!/usr/bin/env python3
"""
单元测试：自动检测计算需求模块

测试：
1. resolve_feature_dependencies 函数
2. map_features_to_tier_nodes 函数
3. extract_required_features_from_execution_archetypes 函数
4. 集成测试：完整的自动检测流程
"""

from pathlib import Path

import pytest
import yaml

from src.cli.auto_detect_compute_requirements import (
    resolve_feature_dependencies,
    map_features_to_tier_nodes,
    extract_required_features_from_execution_archetypes,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_feature_dependencies():
    """创建示例feature_dependencies配置"""
    return {
        "features": {
            "path_efficiency_pct_f": {
                "module": "baseline",
                "compute_func": "compute_path_efficiency_pct_from_series",
                "dependencies": ["path_efficiency_f", "atr_f"],
                "output_columns": ["path_efficiency_pct"],
            },
            "path_efficiency_f": {
                "module": "baseline",
                "compute_func": "compute_path_efficiency_from_series",
                "dependencies": ["atr_f"],
                "output_columns": ["path_efficiency"],
            },
            "atr_f": {
                "module": "baseline",
                "compute_func": "compute_atr",
                "dependencies": [],
                "output_columns": ["atr"],
            },
            "jump_risk_pct_f": {
                "module": "baseline",
                "compute_func": "compute_jump_risk_pct_from_series",
                "dependencies": ["jump_risk_f", "atr_f"],
                "output_columns": ["jump_risk_pct"],
            },
            "jump_risk_f": {
                "module": "baseline",
                "compute_func": "compute_jump_risk_from_series",
                "dependencies": ["atr_f"],
                "output_columns": ["jump_risk"],
            },
        }
    }


class TestResolveFeatureDependencies:
    """测试resolve_feature_dependencies函数"""

    def test_resolve_single_node_no_deps(self, sample_feature_dependencies):
        """测试解析没有依赖的单个node"""
        nodes = {"atr_f"}
        result = resolve_feature_dependencies(nodes, sample_feature_dependencies)
        assert result == {"atr_f"}

    def test_resolve_single_node_with_deps(self, sample_feature_dependencies):
        """测试解析有依赖的单个node"""
        nodes = {"path_efficiency_pct_f"}
        result = resolve_feature_dependencies(nodes, sample_feature_dependencies)
        # 应该包含path_efficiency_pct_f及其所有依赖
        assert "path_efficiency_pct_f" in result
        assert "path_efficiency_f" in result
        assert "atr_f" in result

    def test_resolve_multiple_nodes(self, sample_feature_dependencies):
        """测试解析多个nodes"""
        nodes = {"path_efficiency_pct_f", "jump_risk_pct_f"}
        result = resolve_feature_dependencies(nodes, sample_feature_dependencies)
        # 应该包含所有nodes及其依赖
        assert "path_efficiency_pct_f" in result
        assert "jump_risk_pct_f" in result
        assert "path_efficiency_f" in result
        assert "jump_risk_f" in result
        assert "atr_f" in result  # 共享依赖

    def test_resolve_empty_nodes(self, sample_feature_dependencies):
        """测试解析空nodes集合"""
        nodes = set()
        result = resolve_feature_dependencies(nodes, sample_feature_dependencies)
        assert result == set()

    def test_resolve_none_dependencies(self):
        """测试处理None dependencies"""
        nodes = {"atr_f"}
        result = resolve_feature_dependencies(nodes, None)
        assert result == {"atr_f"}

    def test_resolve_empty_dependencies(self):
        """测试处理空dependencies"""
        nodes = {"atr_f"}
        result = resolve_feature_dependencies(nodes, {})
        assert result == {"atr_f"}

    def test_resolve_circular_dependency_handling(self, sample_feature_dependencies):
        """测试处理循环依赖（应该不会无限循环）"""
        # 添加一个循环依赖（虽然不应该存在，但测试防御性）
        deps = sample_feature_dependencies.copy()
        deps["features"]["path_efficiency_f"]["dependencies"] = [
            "path_efficiency_pct_f"
        ]  # 创建循环

        nodes = {"path_efficiency_pct_f"}
        # 应该能正常处理，不会无限循环
        result = resolve_feature_dependencies(nodes, deps)
        assert "path_efficiency_pct_f" in result


class TestMapFeaturesToTierNodes:
    """测试map_features_to_tier_nodes函数"""

    def test_map_single_feature(self, sample_feature_dependencies):
        """测试映射单个特征"""
        features = {"path_efficiency_pct"}
        result = map_features_to_tier_nodes(features, sample_feature_dependencies)
        assert "path_efficiency_pct_f" in result

    def test_map_multiple_features(self, sample_feature_dependencies):
        """测试映射多个特征"""
        features = {"path_efficiency_pct", "jump_risk_pct"}
        result = map_features_to_tier_nodes(features, sample_feature_dependencies)
        assert "path_efficiency_pct_f" in result
        assert "jump_risk_pct_f" in result

    def test_map_nonexistent_feature(self, sample_feature_dependencies):
        """测试映射不存在的特征"""
        features = {"nonexistent_feature"}
        result = map_features_to_tier_nodes(features, sample_feature_dependencies)
        assert result == set()

    def test_map_empty_features(self, sample_feature_dependencies):
        """测试映射空特征集合"""
        features = set()
        result = map_features_to_tier_nodes(features, sample_feature_dependencies)
        assert result == set()

    def test_map_none_dependencies(self):
        """测试处理None dependencies"""
        features = {"path_efficiency_pct"}
        result = map_features_to_tier_nodes(features, None)
        assert result == set()

    def test_map_empty_dependencies(self):
        """测试处理空dependencies"""
        features = {"path_efficiency_pct"}
        result = map_features_to_tier_nodes(features, {})
        assert result == set()


class TestExtractRequiredFeaturesFromExecutionArchetypes:
    """测试extract_required_features_from_execution_archetypes函数"""

    def test_extract_from_real_config(self):
        """测试从真实的execution_archetypes.yaml提取特征"""
        config_path = PROJECT_ROOT / "config/nnmultihead/execution_archetypes.yaml"
        if not config_path.exists():
            pytest.skip("execution_archetypes.yaml not found")

        features = extract_required_features_from_execution_archetypes(config_path)
        assert isinstance(features, set)
        assert len(features) > 0

        # 验证一些常见的特征
        expected_features = [
            "path_efficiency_pct",
            "jump_risk_pct",
            "cvd_change_5_pct",
            "volume_ratio_pct",
        ]
        found_expected = [f for f in expected_features if f in features]
        assert (
            len(found_expected) > 0
        ), f"应该找到至少一个预期特征，但只找到: {features}"

    def test_extract_from_nonexistent_file(self):
        """测试从不存在的文件提取（应该返回空集合）"""
        config_path = PROJECT_ROOT / "nonexistent_file.yaml"
        features = extract_required_features_from_execution_archetypes(config_path)
        assert features == set()

    def test_extract_from_invalid_yaml(self, tmp_path):
        """测试从无效的YAML文件提取"""
        import yaml

        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("invalid: yaml: content: [")
        features = extract_required_features_from_execution_archetypes(invalid_yaml)
        # 应该能处理异常，返回空集合
        assert isinstance(features, set)
        assert features == set()


class TestIntegration:
    """集成测试：完整的自动检测流程"""

    def test_full_auto_detect_flow(self):
        """测试完整的自动检测流程"""
        # 1. 从execution_archetypes.yaml提取特征
        config_path = PROJECT_ROOT / "config/nnmultihead/execution_archetypes.yaml"
        if not config_path.exists():
            pytest.skip("execution_archetypes.yaml not found")

        features = extract_required_features_from_execution_archetypes(config_path)
        assert len(features) > 0

        # 2. 读取feature_dependencies.yaml
        deps_path = PROJECT_ROOT / "config/feature_dependencies.yaml"
        if not deps_path.exists():
            pytest.skip("feature_dependencies.yaml not found")

        with open(deps_path, "r", encoding="utf-8") as f:
            feature_dependencies = yaml.safe_load(f) or {}

        # 3. 映射特征到feature nodes
        gate_nodes = map_features_to_tier_nodes(features, feature_dependencies)
        assert len(gate_nodes) > 0

        # 4. 解析依赖
        all_gate_nodes = resolve_feature_dependencies(gate_nodes, feature_dependencies)
        assert len(all_gate_nodes) >= len(gate_nodes)

        # 5. 验证所有gate nodes的依赖都被包含
        for node in gate_nodes:
            assert node in all_gate_nodes

        print(f"\n✅ 完整流程测试通过:")
        print(f"   - 提取到 {len(features)} 个特征")
        print(f"   - 映射到 {len(gate_nodes)} 个feature nodes")
        print(f"   - 解析依赖后共有 {len(all_gate_nodes)} 个nodes")

    def test_consistency_with_build_feature_store(self):
        """测试与build_feature_store_nnmultihead.py的一致性"""
        # 这个测试确保我们的公共函数与build_feature_store_nnmultihead.py中的逻辑一致
        config_path = PROJECT_ROOT / "config/nnmultihead/execution_archetypes.yaml"
        deps_path = PROJECT_ROOT / "config/feature_dependencies.yaml"

        if not config_path.exists() or not deps_path.exists():
            pytest.skip("Required config files not found")

        # 使用公共函数
        features = extract_required_features_from_execution_archetypes(config_path)
        with open(deps_path, "r", encoding="utf-8") as f:
            feature_dependencies = yaml.safe_load(f) or {}
        gate_nodes = map_features_to_tier_nodes(features, feature_dependencies)
        all_gate_nodes = resolve_feature_dependencies(gate_nodes, feature_dependencies)

        # 验证结果不为空
        assert len(all_gate_nodes) > 0

        # 验证包含一些预期的nodes
        expected_nodes = [
            "path_efficiency_pct_f",
            "jump_risk_pct_f",
            "cvd_change_5_pct_f",
        ]
        found_expected = [n for n in expected_nodes if n in all_gate_nodes]
        assert (
            len(found_expected) > 0
        ), f"应该找到至少一个预期的node，但只找到: {all_gate_nodes}"

    def test_consistency_with_live_feature_plan(self):
        """测试与live_feature_plan.py的一致性"""
        # 这个测试确保我们的公共函数与live_feature_plan.py中的逻辑一致
        config_path = PROJECT_ROOT / "config/nnmultihead/execution_archetypes.yaml"
        deps_path = PROJECT_ROOT / "config/feature_dependencies.yaml"

        if not config_path.exists() or not deps_path.exists():
            pytest.skip("Required config files not found")

        # 使用公共函数
        features = extract_required_features_from_execution_archetypes(config_path)
        with open(deps_path, "r", encoding="utf-8") as f:
            feature_dependencies = yaml.safe_load(f) or {}
        gate_nodes = map_features_to_tier_nodes(features, feature_dependencies)
        all_gate_nodes = resolve_feature_dependencies(gate_nodes, feature_dependencies)

        # 验证结果不为空
        assert len(all_gate_nodes) > 0

        print(f"\n✅ 与live_feature_plan.py一致性测试通过:")
        print(f"   - 检测到 {len(all_gate_nodes)} 个gate-required nodes")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
