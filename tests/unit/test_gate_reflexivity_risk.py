#!/usr/bin/env python3
"""
单元测试：Gate规则中的反身性风险控制

测试：
1. SHD hard veto
2. LFI soft veto
3. OFCI soft veto
4. 仓位调整函数
5. 快速hard veto检查
"""

import pytest

from src.time_series_model.nnmultihead.gate_reflexivity_risk import (
    gate_reflexivity_risk,
    apply_reflexivity_position_scaling,
    check_reflexivity_hard_veto,
)


class TestSHDHardVeto:
    """测试SHD hard veto"""

    def test_shd_hard_veto(self):
        """测试SHD hard veto（shd_p > 0.9）"""
        features = {
            "shd_p": 0.95,
            "ofci_p": 0.5,
            "lfi_p": 0.5,
        }

        allow, multiplier, reason = gate_reflexivity_risk(features)

        assert allow is False
        assert multiplier == 0.0
        assert "strategy_homogeneity" in reason

    def test_shd_below_threshold(self):
        """测试SHD低于阈值时不应该veto"""
        features = {
            "shd_p": 0.8,
            "ofci_p": 0.5,
            "lfi_p": 0.5,
        }

        allow, multiplier, reason = gate_reflexivity_risk(features)

        assert allow is True
        assert multiplier == 1.0  # 没有soft veto

    def test_shd_exactly_at_threshold(self):
        """测试SHD刚好在阈值"""
        features = {
            "shd_p": 0.9,
            "ofci_p": 0.5,
            "lfi_p": 0.5,
        }

        allow, multiplier, reason = gate_reflexivity_risk(features)

        # 0.9不应该触发（应该是>0.9）
        assert allow is True


class TestLFISoftVeto:
    """测试LFI soft veto"""

    def test_lfi_soft_veto(self):
        """测试LFI soft veto（lfi_p > 0.9，position *= 0.3）"""
        features = {
            "shd_p": 0.5,  # 低于阈值，不触发hard veto
            "ofci_p": 0.5,
            "lfi_p": 0.95,
        }

        allow, multiplier, reason = gate_reflexivity_risk(features)

        assert allow is True
        assert abs(multiplier - 0.3) < 1e-6
        assert "fragile_liquidity" in reason

    def test_lfi_below_threshold(self):
        """测试LFI低于阈值时不应该soft veto"""
        features = {
            "shd_p": 0.5,
            "ofci_p": 0.5,
            "lfi_p": 0.8,
        }

        allow, multiplier, reason = gate_reflexivity_risk(features)

        assert allow is True
        assert multiplier == 1.0


class TestOFCISoftVeto:
    """测试OFCI soft veto"""

    def test_ofci_soft_veto(self):
        """测试OFCI soft veto（ofci_p > 0.9，position *= 0.6）"""
        features = {
            "shd_p": 0.5,  # 低于阈值
            "lfi_p": 0.5,  # 低于阈值
            "ofci_p": 0.95,
        }

        allow, multiplier, reason = gate_reflexivity_risk(features)

        assert allow is True
        assert abs(multiplier - 0.6) < 1e-6
        assert "high_consensus" in reason

    def test_ofci_below_threshold(self):
        """测试OFCI低于阈值时不应该soft veto"""
        features = {
            "shd_p": 0.5,
            "lfi_p": 0.5,
            "ofci_p": 0.8,
        }

        allow, multiplier, reason = gate_reflexivity_risk(features)

        assert allow is True
        assert multiplier == 1.0


class TestPriorityOrder:
    """测试优先级顺序"""

    def test_shd_priority_over_lfi(self):
        """测试SHD优先级高于LFI"""
        features = {
            "shd_p": 0.95,  # 应该触发hard veto
            "lfi_p": 0.95,  # 虽然也高，但SHD优先
            "ofci_p": 0.5,
        }

        allow, multiplier, reason = gate_reflexivity_risk(features)

        # SHD应该优先，触发hard veto
        assert allow is False
        assert "strategy_homogeneity" in reason

    def test_lfi_priority_over_ofci(self):
        """测试LFI优先级高于OFCI"""
        features = {
            "shd_p": 0.5,  # 低于阈值
            "lfi_p": 0.95,  # 应该触发soft veto
            "ofci_p": 0.95,  # 虽然也高，但LFI优先
        }

        allow, multiplier, reason = gate_reflexivity_risk(features)

        # LFI应该优先
        assert allow is True
        assert abs(multiplier - 0.3) < 1e-6
        assert "fragile_liquidity" in reason


class TestApplyReflexivityPositionScaling:
    """测试仓位调整函数"""

    def test_apply_reflexivity_position_scaling_hard_veto(self):
        """测试hard veto时返回0仓位"""
        features = {
            "shd_p": 0.95,
            "ofci_p": 0.5,
            "lfi_p": 0.5,
        }

        adjusted_position, reason = apply_reflexivity_position_scaling(
            base_position=100.0,
            features=features,
        )

        assert adjusted_position == 0.0
        assert "strategy_homogeneity" in reason

    def test_apply_reflexivity_position_scaling_soft_veto(self):
        """测试soft veto时应用仓位倍数"""
        features = {
            "shd_p": 0.5,
            "lfi_p": 0.95,  # 触发soft veto
            "ofci_p": 0.5,
        }

        adjusted_position, reason = apply_reflexivity_position_scaling(
            base_position=100.0,
            features=features,
        )

        assert abs(adjusted_position - 30.0) < 1e-6  # 100 * 0.3
        assert "fragile_liquidity" in reason

    def test_apply_reflexivity_position_scaling_no_veto(self):
        """测试没有veto时保持原仓位"""
        features = {
            "shd_p": 0.5,
            "lfi_p": 0.5,
            "ofci_p": 0.5,
        }

        adjusted_position, reason = apply_reflexivity_position_scaling(
            base_position=100.0,
            features=features,
        )

        assert adjusted_position == 100.0
        assert "reflexivity_risk_acceptable" in reason


class TestCheckReflexivityHardVeto:
    """测试快速hard veto检查"""

    def test_check_reflexivity_hard_veto_triggered(self):
        """测试触发hard veto"""
        features = {
            "shd_p": 0.95,
        }

        should_veto, reason = check_reflexivity_hard_veto(features)

        assert should_veto is True
        assert reason is not None
        assert "strategy_homogeneity" in reason

    def test_check_reflexivity_hard_veto_not_triggered(self):
        """测试不触发hard veto"""
        features = {
            "shd_p": 0.8,
        }

        should_veto, reason = check_reflexivity_hard_veto(features)

        assert should_veto is False
        assert reason is None

    def test_check_reflexivity_hard_veto_missing_key(self):
        """测试缺少shd_p键"""
        features = {}

        should_veto, reason = check_reflexivity_hard_veto(features)

        assert should_veto is False
        assert reason is None


class TestEdgeCases:
    """测试边界情况"""

    def test_missing_features(self):
        """测试缺少某些特征"""
        features = {
            "shd_p": 0.5,
            # 缺少ofci_p和lfi_p
        }

        allow, multiplier, reason = gate_reflexivity_risk(features)

        # 应该使用默认值0.0，不触发任何veto
        assert allow is True
        assert multiplier == 1.0

    def test_all_features_high(self):
        """测试所有特征都很高（SHD应该优先）"""
        features = {
            "shd_p": 0.95,
            "lfi_p": 0.95,
            "ofci_p": 0.95,
        }

        allow, multiplier, reason = gate_reflexivity_risk(features)

        # SHD应该优先，触发hard veto
        assert allow is False

    def test_zero_values(self):
        """测试零值"""
        features = {
            "shd_p": 0.0,
            "lfi_p": 0.0,
            "ofci_p": 0.0,
        }

        allow, multiplier, reason = gate_reflexivity_risk(features)

        assert allow is True
        assert multiplier == 1.0
