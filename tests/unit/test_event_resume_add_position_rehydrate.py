"""跨月 resume 后应重建宪法层 add_count（与 event_backtest.run 一致）。"""

from __future__ import annotations

from scripts.event_backtest import (
    PositionSimulator,
    _collect_open_parent_pids,
    _filter_add_position_dict_for_open_parents,
    _load_add_position_runtime_from_resume,
    _merge_add_position_runtime_with_open_legs,
    _prune_stale_add_position_records,
    _rehydrate_add_position_runtime_from_simulator,
)
from src.time_series_model.core.constitution.runtime_state import (
    AddPositionRecord,
    ConstitutionRuntimeState,
)


def test_rehydrate_sets_add_count_from_open_add_legs() -> None:
    sim = PositionSimulator()
    parent_pid = "parentpid111"
    sim._positions[parent_pid] = {
        "_is_add_position": False,
        "breakeven_locked": False,
        "symbol": "ADAUSDT",
        "side": "LONG",
    }
    for i in range(3):
        sim._positions[f"child{i}"] = {
            "_is_add_position": True,
            "_parent_pid": parent_pid,
        }
    st = ConstitutionRuntimeState()
    _rehydrate_add_position_runtime_from_simulator(sim, st)
    rec = st.add_position.positions.get(parent_pid)
    assert rec is not None
    assert rec.add_count == 3


def test_rehydrate_ignores_parent_without_add_legs() -> None:
    sim = PositionSimulator()
    sim._positions["solo"] = {"_is_add_position": False}
    st = ConstitutionRuntimeState()
    _rehydrate_add_position_runtime_from_simulator(sim, st)
    assert "solo" not in st.add_position.positions


def test_load_add_position_runtime_from_resume_state() -> None:
    st = ConstitutionRuntimeState()
    loaded = _load_add_position_runtime_from_resume(
        {
            "add_position_state": {
                "positions": {
                    "parent-a": {
                        "position_id": "parent-a",
                        "add_count": 5,
                        "locked_profit": True,
                        "current_r": 1.2,
                        "updated_at": "2026-01-01T00:00:00+00:00",
                    }
                }
            }
        },
        st,
    )
    assert loaded == 1
    rec = st.add_position.positions.get("parent-a")
    assert rec is not None
    assert rec.add_count == 5
    assert rec.locked_profit is True
    assert rec.current_r == 1.2


def test_restore_open_positions_inherits_structural_for_old_resume() -> None:
    sim = PositionSimulator()
    loaded = sim.restore_open_positions(
        [
            {
                "pid": "parent1",
                "position": {
                    "_is_add_position": False,
                    "structural_exit": "vwap1200",
                    "entry_time": "2026-01-01T00:00:00+00:00",
                },
            },
            {
                "pid": "child1",
                "position": {
                    "_is_add_position": True,
                    "_parent_pid": "parent1",
                    "entry_time": "2026-01-01T00:00:00+00:00",
                },
            },
        ]
    )
    assert loaded == 2
    assert sim._positions["child1"].get("structural_exit") == "vwap1200"


def test_prune_removes_stale_add_position_then_merge_repairs_parent() -> None:
    """模拟 resume：JSON 含已平仓父仓 pid；当前仅 parent-b 开仓 + 2 条子腿。"""
    sim = PositionSimulator()
    sim._positions["parent-b"] = {"_is_add_position": False}
    sim._positions["c1"] = {"_is_add_position": True, "_parent_pid": "parent-b"}
    sim._positions["c2"] = {"_is_add_position": True, "_parent_pid": "parent-b"}
    st = ConstitutionRuntimeState()
    _load_add_position_runtime_from_resume(
        {
            "add_position_state": {
                "positions": {
                    "ghost": {"add_count": 3, "position_id": "ghost"},
                    "parent-b": {"add_count": 0, "position_id": "parent-b"},
                }
            }
        },
        st,
    )
    assert "ghost" in st.add_position.positions
    open_p = _collect_open_parent_pids({"s": sim})
    assert open_p == {"parent-b"}
    _prune_stale_add_position_records(st, open_p)
    assert "ghost" not in st.add_position.positions
    _merge_add_position_runtime_with_open_legs(sim, st)
    assert st.add_position.positions["parent-b"].add_count == 2


def test_merge_does_not_downgrade_add_count_when_add_legs_closed() -> None:
    """已加仓次数 3、仅剩 1 条子腿时不得把 add_count 压成 1（否则错误释放加仓额度）。"""
    sim = PositionSimulator()
    sim._positions["p"] = {"_is_add_position": False}
    sim._positions["c0"] = {"_is_add_position": True, "_parent_pid": "p"}
    st = ConstitutionRuntimeState()
    st.add_position.positions["p"] = AddPositionRecord(position_id="p", add_count=3)
    _merge_add_position_runtime_with_open_legs(sim, st)
    assert st.add_position.positions["p"].add_count == 3


def test_filter_add_position_dict_keeps_only_open_parents() -> None:
    full = {
        "positions": {
            "a": {"position_id": "a", "add_count": 1},
            "b": {"position_id": "b", "add_count": 2},
        }
    }
    rows = [{"pid": "b", "position": {"_is_add_position": False}}]
    out = _filter_add_position_dict_for_open_parents(full, rows)
    assert set(out["positions"].keys()) == {"b"}
