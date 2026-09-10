from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.time_series_model.core.constitution.constitution_executor import (
    ConstitutionExecutor,
)
from src.time_series_model.core.constitution.period_loss_policy import (
    safety_limits_from_cfg,
)
from src.time_series_model.core.constitution.safety_runtime import (
    SafetyRuntimeState,
    evaluate_safety_state,
    load_safety_state,
    save_safety_state,
)
from src.time_series_model.core.constitution.runtime_state import (
    ConstitutionRuntimeState,
)
from src.time_series_model.core.constitution.state import ConstitutionState
from src.time_series_model.core.constitution.violation import ConstitutionViolation
from src.time_series_model.ops.state_snapshot import (
    SystemStateSnapshot,
    write_state_snapshot,
)


@dataclass(frozen=True)
class LiveEnforcementResult:
    ok: bool
    reason: str
    snapshot_path: Optional[str] = None


def enforce_before_order(
    *,
    executor: ConstitutionExecutor,
    runtime_state: ConstitutionRuntimeState,
    position_id: str,
    symbol: str,
    archetype: str,
    execution_strategy: Optional[str] = None,
    execution_tags: Optional[list[str]] = None,
    execution_evidence: Optional[dict[str, bool]] = None,
    equity: Optional[float] = None,
    drawdown: Optional[float] = None,
    daily_loss: float = 0.0,
    weekly_loss: float = 0.0,
    monthly_loss: float = 0.0,
    daily_cost_mean: Optional[float] = None,
    daily_turnover_mean: Optional[float] = None,
    hard_violation: bool = False,
    data_bad: bool = False,
    evt_risk_flag: Optional[bool] = None,
    snapshot_out: Optional[str | Path] = None,
    snapshot_extra: Optional[Dict[str, Any]] = None,
    pcm_budget: Optional[Dict[str, Any]] = None,
    reserve_slot: bool = True,
) -> LiveEnforcementResult:
    """
    Minimal live adapter hook:
    - validate kill-switch style drawdown constraints
    - reserve a slot (hard cap) unless ``reserve_slot`` is False (add legs)
    - persist runtime state (caller should call executor.save_runtime_state)
    - emit a SystemStateSnapshot for attribution

    ``reserve_slot=False`` is used for add-position legs so they do not consume
    a separate global slot — matching event_backtest, where ``try_add_position``
    never touches ``slots.active`` (only opens are slot-capped at the PCM layer).
    """
    st = ConstitutionState(
        task_id=None,
        timestamp=None,
        equity=equity,
        drawdown=drawdown,
        daily_loss=float(daily_loss),
        weekly_loss=float(weekly_loss),
        monthly_loss=float(monthly_loss),
        hard_violation=bool(hard_violation),
        data_bad=bool(data_bad),
    )
    safety_state = SafetyRuntimeState()
    safety_db_path = None
    safety_state_id = "global"
    asset_g0_active = False
    try:
        from src.live_data_stream.order_manager_factory import margin_kind_from_env
        from src.order_management.coin_m_asset_equity import (
            coin_m_asset_g0_enabled,
            collateral_asset_for_signal,
            live_asset_g0_place_block_reason,
            safety_state_id_for_asset,
        )

        asset_g0_active = coin_m_asset_g0_enabled(margin_kind=margin_kind_from_env())
        if asset_g0_active:
            from src.order_management.coin_m_asset_equity import live_coin_m_asset_book

            asset = collateral_asset_for_signal(symbol)
            if not asset:
                # Align with C kill_switch:unknown_collateral_asset (fail-closed).
                raise ConstitutionViolation(
                    code="UNKNOWN_COLLATERAL_ASSET",
                    message=(
                        f"coin_m asset G0: no collateral mapping for symbol={symbol}"
                    ),
                    context={"symbol": str(symbol), "safety_state_id": "unset"},
                )
            block = live_asset_g0_place_block_reason(asset=asset)
            if block:
                raise ConstitutionViolation(
                    code="ASSET_G0_REFRESH_FAILED",
                    message=(
                        f"coin_m asset G0 place blocked ({block}) for symbol={symbol}"
                    ),
                    context={
                        "symbol": str(symbol),
                        "reason": block,
                        "collateral_asset": asset,
                    },
                )
            book = live_coin_m_asset_book()
            if book is None:
                # Never fall back to SQLite/features when asset G0 is on — that
                # fail-opens after a missed bind or post-stop_all unbind.
                raise ConstitutionViolation(
                    code="ASSET_G0_BOOK_UNBOUND",
                    message=(f"coin_m asset G0 book unbound for symbol={symbol}"),
                    context={
                        "symbol": str(symbol),
                        "collateral_asset": asset,
                        "safety_state_id": safety_state_id_for_asset(asset),
                    },
                )
            if not book.has_asset(asset):
                raise ConstitutionViolation(
                    code="UNSEEDED_COLLATERAL_ASSET",
                    message=(
                        f"coin_m asset G0: collateral={asset} not seeded for "
                        f"symbol={symbol}"
                    ),
                    context={
                        "symbol": str(symbol),
                        "collateral_asset": asset,
                        "safety_state_id": safety_state_id_for_asset(asset),
                    },
                )
            # Book is place authority: do not rely on SQLite alone (persist can fail).
            if book.is_halted(asset):
                reasons = book.halt_reasons(asset) or ["halted"]
                raise ConstitutionViolation(
                    code="SAFETY_HALT",
                    message=f"Safety halted: {', '.join(reasons)}",
                    context={
                        "reasons": reasons,
                        "symbol": str(symbol),
                        "collateral_asset": asset,
                        "source": "coin_m_asset_book",
                    },
                )
            # Prefer live book metrics over stale cycle features.
            m = book.metrics_for(asset)
            if m:
                daily_loss = float(m.get("daily_loss", 0.0) or 0.0)
                weekly_loss = float(m.get("weekly_loss", 0.0) or 0.0)
                monthly_loss = float(m.get("monthly_loss", 0.0) or 0.0)
                drawdown = m.get("drawdown")
            safety_state_id = safety_state_id_for_asset(asset)
    except ConstitutionViolation:
        raise
    except Exception as exc:
        # Never fall back to ``global`` under coin_m asset G0 — that would
        # ignore an existing ``asset:BTC`` halt and fail-open.
        intended_asset_g0 = asset_g0_active
        if not intended_asset_g0:
            try:
                from src.live_data_stream.order_manager_factory import (
                    margin_kind_from_env,
                )
                from src.order_management.coin_m_asset_equity import (
                    coin_m_asset_g0_enabled,
                )

                intended_asset_g0 = coin_m_asset_g0_enabled(
                    margin_kind=margin_kind_from_env()
                )
            except Exception:
                intended_asset_g0 = False
        if intended_asset_g0:
            raise ConstitutionViolation(
                code="ASSET_G0_RESOLVE_FAILED",
                message=f"coin_m asset G0 resolve failed for symbol={symbol}: {exc}",
                context={"symbol": str(symbol), "error": str(exc)},
            ) from exc
        safety_state_id = "global"
    try:
        safety_db_path = executor.resolve_safety_db_path()
        safety_state = load_safety_state(
            db_path=str(safety_db_path), state_id=safety_state_id
        )
    except Exception:
        safety_state = SafetyRuntimeState()

    now = datetime.now(timezone.utc)
    decision = evaluate_safety_state(
        state=safety_state,
        now=now,
        cooldown_minutes=int(executor.cfg.cooldown_minutes),
        daily_reset_tz=executor.cfg.daily_reset_timezone,
        daily_loss=float(daily_loss),
        weekly_loss=float(weekly_loss),
        monthly_loss=float(monthly_loss),
        drawdown=drawdown,
        hard_violation=bool(hard_violation),
        data_bad=bool(data_bad),
        daily_cost_mean=daily_cost_mean,
        daily_turnover_mean=daily_turnover_mean,
        limits=safety_limits_from_cfg(executor.cfg),
    )
    if not bool(executor.cfg.kill_on_any_hard_violation):
        decision.state.halted = False
        decision.state.halt_reason = []
        decision.state.halt_since = None
        decision.state.cooldown_until = None
    if evt_risk_flag is not None:
        decision.state.last_metrics["evt_risk_flag"] = bool(evt_risk_flag)
    if safety_db_path is not None:
        try:
            save_safety_state(
                db_path=str(safety_db_path),
                state=decision.state,
                state_id=safety_state_id,
            )
        except Exception:
            pass

    if not decision.ok and bool(executor.cfg.kill_on_any_hard_violation):
        if snapshot_out is not None:
            p = Path(snapshot_out)
            p.parent.mkdir(parents=True, exist_ok=True)
            snap = SystemStateSnapshot(
                task_id=None,
                timestamp=None,
                constitution_hash=str(executor.meta().get("constitution_hash")),
                constitution_yaml=str(executor.meta().get("constitution_yaml")),
                router_mode=str(archetype),
                gate_decisions={},
                pcm_budget=dict(pcm_budget or {}),
                active_slots=int(runtime_state.slots.active_count()),
                drawdown=float(drawdown) if drawdown is not None else None,
                observability=None,
                live_dashboard=(
                    dict((snapshot_extra or {}).get("live_dashboard") or {})
                    if snapshot_extra is not None
                    else None
                ),
                kpi_gate=None,
                safety_state=decision.state.as_dict(),
                overrides=[],
            )
            write_state_snapshot(out_path=str(p), snapshot=snap)
        raise ConstitutionViolation(
            code="SAFETY_HALT",
            message=f"Safety halted: {', '.join(decision.reasons or [])}",
            context={
                "reasons": decision.reasons,
                "safety_state": decision.state.as_dict(),
                **st.as_dict(),
                **executor.meta(),
            },
        )

    # Slot reservation (hard constraint). Add legs skip this so they don't
    # consume a separate global slot (parity with event_backtest add path).
    if reserve_slot:
        executor.reserve_slot(
            st=runtime_state,
            position_id=str(position_id),
            symbol=str(symbol),
            archetype=str(archetype),
        )
        executor.save_runtime_state(runtime_state)

    snap_path = None
    if snapshot_out is not None:
        p = Path(snapshot_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        snap = SystemStateSnapshot(
            task_id=None,
            timestamp=None,
            constitution_hash=str(executor.meta().get("constitution_hash")),
            constitution_yaml=str(executor.meta().get("constitution_yaml")),
            router_mode=str(archetype),
            gate_decisions={},
            pcm_budget=dict(pcm_budget or {}),
            active_slots=int(runtime_state.slots.active_count()),
            drawdown=float(drawdown) if drawdown is not None else None,
            observability=None,
            live_dashboard=(
                dict((snapshot_extra or {}).get("live_dashboard") or {})
                if snapshot_extra is not None
                else None
            ),
            kpi_gate=None,
            safety_state=decision.state.as_dict(),
            overrides=[],
        )
        write_state_snapshot(out_path=str(p), snapshot=snap)
        snap_path = str(p)

    return LiveEnforcementResult(ok=True, reason="ok", snapshot_path=snap_path)
