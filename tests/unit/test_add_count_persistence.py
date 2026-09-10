"""Regression: add_count persistence and ladder cap enforcement."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.order_management.models import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    PositionStatus,
)
from src.order_management.storage import Storage
from src.time_series_model.core.constitution.constitution_executor import (
    ConstitutionExecutor,
)
from src.time_series_model.core.constitution.runtime_state import (
    AddPositionRecord,
    ConstitutionRuntimeState,
)
from src.time_series_model.core.constitution.violation import ConstitutionViolation


def _minimal_constitution(tmp_path) -> str:
    cy = tmp_path / "constitution.yaml"
    db = tmp_path / "om.db"
    cy.write_text(
        f"""
version: 1
name: "ADD_COUNT_TEST"
kill_switch:
  enabled: false
safety_state:
  persist_to: "{db.as_posix()}"
slots:
  enabled: true
  slot_count: 2
  risk_per_slot: 0.01
  slot_state_tracking:
    persist_to: "{db.as_posix()}"
resource_allocation:
  per_strategy_limits:
    tpc:
      allow_add_position: true
      max_add_times: 3
""",
        encoding="utf-8",
    )
    return str(cy)


@pytest.mark.unit
def test_add_position_state_roundtrip(tmp_path) -> None:
    cy = _minimal_constitution(tmp_path)
    ex = ConstitutionExecutor(constitution_yaml=cy)
    st = ex.load_runtime_state()
    st.add_position.positions["p1"] = AddPositionRecord(position_id="p1", add_count=2)
    ex.save_runtime_state(st)
    st2 = ex.load_runtime_state()
    assert st2.add_position.positions["p1"].add_count == 2


@pytest.mark.unit
def test_validate_add_position_respects_ladder_cap(tmp_path) -> None:
    cy = _minimal_constitution(tmp_path)
    ex = ConstitutionExecutor(constitution_yaml=cy)
    st = ConstitutionRuntimeState()
    st.add_position.positions["p1"] = AddPositionRecord(position_id="p1", add_count=2)
    with pytest.raises(ConstitutionViolation, match="max_add_times"):
        ex.validate_add_position(
            st=st,
            position_id="p1",
            archetype="tpc",
            current_r=1.0,
            max_add_times_cap=2,
        )


@pytest.mark.unit
def test_validate_add_position_respects_explicit_zero_max_add_times(tmp_path) -> None:
    """Regression (2026-08-05): ``int(strat_cfg.get("max_add_times", 1) or 1)``
    silently coerced an explicit ``max_add_times: 0`` (disable adds entirely)
    back to 1, since 0 is falsy in Python. First add attempt must be blocked."""
    cy = tmp_path / "constitution.yaml"
    db = tmp_path / "om.db"
    cy.write_text(
        f"""
version: 1
name: "ADD_COUNT_ZERO_TEST"
kill_switch:
  enabled: false
safety_state:
  persist_to: "{db.as_posix()}"
resource_allocation:
  per_strategy_limits:
    srb:
      allow_add_position: true
      max_add_times: 0
""",
        encoding="utf-8",
    )
    ex = ConstitutionExecutor(constitution_yaml=str(cy))
    st = ConstitutionRuntimeState()
    with pytest.raises(ConstitutionViolation, match="max_add_times"):
        ex.validate_add_position(
            st=st,
            position_id="p1",
            archetype="srb",
            current_r=1.0,
        )


@pytest.mark.unit
def test_count_filled_buy_orders_since(tmp_path) -> None:
    storage = Storage(str(tmp_path / "orders.db"))
    now = datetime.now(timezone.utc)
    for i, oid in enumerate(["o1", "o2", "o3"], start=1):
        storage.create_order(
            Order(
                order_id=oid,
                client_order_id=f"c{i}",
                symbol="HYPEUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=float(i),
                filled_quantity=float(i),
                status=OrderStatus.FILLED,
                created_at=now,
            )
        )
    n = storage.count_filled_buy_orders_since("HYPEUSDT", since=now)
    assert n == 3


@pytest.mark.unit
def test_rehydrate_add_count_from_orders(tmp_path) -> None:
    cy = _minimal_constitution(tmp_path)
    ex = ConstitutionExecutor(constitution_yaml=cy)
    db = tmp_path / "om.db"
    storage = Storage(str(db))
    entry = datetime(2026, 6, 30, tzinfo=timezone.utc)
    storage.create_position(
        Position(
            position_id="HYPEUSDT:1",
            symbol="HYPEUSDT",
            side=PositionSide.LONG,
            entry_time=entry,
            entry_price=68.0,
            initial_size=26.0,
            current_size=153.0,
            total_cost=68.0 * 153.0,
            status=PositionStatus.OPEN,
            strategy_id="tpc",
            add_count=0,
        )
    )
    for oid in ("a", "b", "c"):
        storage.create_order(
            Order(
                order_id=oid,
                client_order_id=oid,
                symbol="HYPEUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=10.0,
                filled_quantity=10.0,
                status=OrderStatus.FILLED,
                created_at=entry,
            )
        )
    st = ex.load_runtime_state()
    assert st.add_position.positions["HYPEUSDT:1"].add_count == 2
