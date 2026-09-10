"""加仓腿应继承父仓 structural_exit（与 event_backtest.try_add_position 行为一致）。"""

from __future__ import annotations

from datetime import datetime, timezone

from src.time_series_model.core.trade_intent import TradeIntent
from src.time_series_model.live.position_logic import build_position_dict


def _apply_add_structural_inherit(pos: dict, parent_pos: dict) -> None:
    """Mirror scripts/event_backtest.py try_add_position post-build_position_dict logic."""
    _p_se = parent_pos.get("structural_exit")
    if _p_se and not pos.get("structural_exit"):
        pos["structural_exit"] = str(_p_se)


def test_float_ladder_style_intent_has_no_structural_until_inherit() -> None:
    now = datetime.now(timezone.utc)
    parent_intent = TradeIntent(
        action="LONG",
        symbol="ADAUSDT",
        archetype="bpc-long-120t",
        execution_profile={
            "rr_constraints": {
                "stop_loss_r": 4.0,
                "max_holding_bars": 0,
                "structural_exit": "vwap1200",
            },
            "bpc_position_config": {"breakeven_enabled": False},
        },
    )
    parent = build_position_dict(
        parent_intent, 1.0, 0.1, bar_minutes=120, entry_time=now
    )
    assert parent.get("structural_exit") == "vwap1200"

    ladder_intent = TradeIntent(
        action="LONG",
        symbol="ADAUSDT",
        archetype="bpc-long-120t",
        add_position=True,
        execution_profile={
            "add_position": {"trigger": {"type": "float_r_ladder_only"}},
        },
    )
    child = build_position_dict(
        ladder_intent, 1.1, 0.1, bar_minutes=120, entry_time=now
    )
    assert "structural_exit" not in child

    _apply_add_structural_inherit(child, parent)
    assert child.get("structural_exit") == "vwap1200"


def test_pcm_style_intent_unchanged_when_already_has_structural() -> None:
    now = datetime.now(timezone.utc)
    parent = build_position_dict(
        TradeIntent(
            action="LONG",
            symbol="BTCUSDT",
            archetype="bpc-long-120t",
            execution_profile={
                "rr_constraints": {
                    "stop_loss_r": 4.0,
                    "max_holding_bars": 0,
                    "structural_exit": "vwap1200",
                },
            },
        ),
        100.0,
        1.0,
        bar_minutes=120,
        entry_time=now,
    )
    child = build_position_dict(
        TradeIntent(
            action="LONG",
            symbol="BTCUSDT",
            archetype="bpc-long-120t",
            add_position=True,
            execution_profile={
                "rr_constraints": {"structural_exit": "vwap1200", "stop_loss_r": 4.0},
            },
        ),
        101.0,
        1.0,
        bar_minutes=120,
        entry_time=now,
    )
    assert child.get("structural_exit") == "vwap1200"
    _apply_add_structural_inherit(child, parent)
    assert child.get("structural_exit") == "vwap1200"
