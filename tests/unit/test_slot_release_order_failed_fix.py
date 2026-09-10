"""
Bug fix 测试: 下单失败释放预留 slot (slot 泄漏修复)

修复背景:
    enforce_before_order() 在 place_order() 之前预留 slot。
    如果 place_order() 失败（API key/余额/权限），slot 永久泄漏，
    导致 slot 被占满后无法再开新仓。

修复内容:
    1. SLOT_RELEASE_REASONS 新增 "order_failed"
    2. order_flow_listener._execute_intent() 异常处理中调用 release_slot(reason="order_failed")
"""

import pytest


class TestSlotReleaseReasons:
    """验证 SLOT_RELEASE_REASONS 包含 order_failed"""

    def test_order_failed_in_release_reasons(self):
        """order_failed 必须在 SLOT_RELEASE_REASONS 中"""
        from src.time_series_model.core.constitution.constitution_executor import (
            SLOT_RELEASE_REASONS,
        )

        assert (
            "order_failed" in SLOT_RELEASE_REASONS
        ), "SLOT_RELEASE_REASONS 缺少 'order_failed'，下单失败时 slot 将泄漏"
        assert (
            "stale_sync_no_api" in SLOT_RELEASE_REASONS
        ), "SLOT_RELEASE_REASONS 缺少 'stale_sync_no_api'，api=None 定期同步无法清槽"

    def test_original_reasons_still_present(self):
        """原有的释放原因不应被删除"""
        from src.time_series_model.core.constitution.constitution_executor import (
            SLOT_RELEASE_REASONS,
        )

        for reason in ["position_closed", "stop_loss_hit", "take_profit_hit"]:
            assert reason in SLOT_RELEASE_REASONS, f"原有释放原因 '{reason}' 被意外删除"


class TestReleaseSlotOrderFailed:
    """验证 release_slot 使用 order_failed 原因能正确释放 slot"""

    @pytest.fixture
    def runtime_state(self):
        """创建包含一个活跃 slot 的 runtime state"""
        from src.time_series_model.core.constitution.runtime_state import (
            ConstitutionRuntimeState,
            SlotRecord,
            SlotsRuntimeState,
        )

        st = ConstitutionRuntimeState()
        # 预留一个 slot（模拟 enforce_before_order 的效果）
        st.slots.active["BTCUSDT:123"] = SlotRecord(
            position_id="BTCUSDT:123",
            symbol="BTCUSDT",
            archetype="compression_breakout",
        )
        return st

    @pytest.fixture
    def executor(self, tmp_path):
        """创建 ConstitutionExecutor 实例（使用临时 constitution.yaml）"""
        from src.time_series_model.core.constitution.constitution_executor import (
            ConstitutionExecutor,
        )

        # 写一个最小的 constitution.yaml
        yaml_path = tmp_path / "constitution.yaml"
        yaml_path.write_text(
            """
version: 1
name: test_constitution
kill_switch:
  enabled: false
  daily_loss_limit: 0.04
  weekly_loss_limit: 0.08
  monthly_loss_limit: 0.12
  max_dd: 0.20
  max_turnover_mean: 0.35
  max_cost_mean: 0.002
  kill_on_any_hard_violation: true
""",
            encoding="utf-8",
        )
        return ConstitutionExecutor(constitution_yaml=yaml_path)

    def test_release_slot_with_order_failed(self, executor, runtime_state):
        """order_failed 原因应成功释放 slot"""
        assert "BTCUSDT:123" in runtime_state.slots.active
        assert runtime_state.slots.active_count() == 1

        executor.release_slot(
            st=runtime_state,
            position_id="BTCUSDT:123",
            reason="order_failed",
        )

        assert "BTCUSDT:123" not in runtime_state.slots.active
        assert runtime_state.slots.active_count() == 0

    def test_release_slot_with_invalid_reason_noop(self, executor, runtime_state):
        """无效的 release reason 不应释放 slot"""
        executor.release_slot(
            st=runtime_state,
            position_id="BTCUSDT:123",
            reason="invalid_reason",
        )

        # slot 应该仍在
        assert "BTCUSDT:123" in runtime_state.slots.active
        assert runtime_state.slots.active_count() == 1

    def test_release_nonexistent_slot_noop(self, executor, runtime_state):
        """释放不存在的 slot 应为 no-op"""
        executor.release_slot(
            st=runtime_state,
            position_id="NONEXIST:999",
            reason="order_failed",
        )

        # 原有 slot 不受影响
        assert "BTCUSDT:123" in runtime_state.slots.active
        assert runtime_state.slots.active_count() == 1

    def test_release_slot_empty_pid_noop(self, executor, runtime_state):
        """空 position_id 应为 no-op"""
        executor.release_slot(
            st=runtime_state,
            position_id="  ",
            reason="order_failed",
        )

        assert runtime_state.slots.active_count() == 1

    def test_traditional_reasons_still_work(self, executor, runtime_state):
        """传统释放原因（position_closed 等）应继续正常工作"""
        executor.release_slot(
            st=runtime_state,
            position_id="BTCUSDT:123",
            reason="position_closed",
        )

        assert runtime_state.slots.active_count() == 0

    def test_multiple_slots_only_target_released(self, executor):
        """多个 slot 时只释放目标 slot"""
        from src.time_series_model.core.constitution.runtime_state import (
            ConstitutionRuntimeState,
            SlotRecord,
        )

        st = ConstitutionRuntimeState()
        st.slots.active["BTCUSDT:1"] = SlotRecord(
            position_id="BTCUSDT:1", symbol="BTCUSDT"
        )
        st.slots.active["ETHUSDT:2"] = SlotRecord(
            position_id="ETHUSDT:2", symbol="ETHUSDT"
        )

        executor.release_slot(
            st=st,
            position_id="BTCUSDT:1",
            reason="order_failed",
        )

        assert "BTCUSDT:1" not in st.slots.active
        assert "ETHUSDT:2" in st.slots.active
        assert st.slots.active_count() == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
