"""Day/week kill-switch: hard halt vs soft derate (shared live + event_backtest).

When ``daily_soft_loss_limit`` / ``weekly_soft_loss_limit`` > 0, those periods
do **not** hard-halt via ``evaluate_safety_state``; callers derate
``risk_per_slot`` instead. ``max_dd`` / monthly hard limits stay unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Tuple


def load_ks_period_soft_config(ks: Mapping[str, Any] | None) -> Dict[str, float]:
    """Read optional soft day/week + peak-DD + derate from ``kill_switch`` mapping."""
    raw = ks or {}
    return {
        "daily_soft_loss_limit": float(raw.get("daily_soft_loss_limit") or 0.0),
        "weekly_soft_loss_limit": float(raw.get("weekly_soft_loss_limit") or 0.0),
        "derated_risk_per_slot": float(raw.get("derated_risk_per_slot") or 0.0),
        # Peak equity DD soft gates (EB + future live): do NOT hard-halt.
        "soft_max_dd_block_adds": float(raw.get("soft_max_dd_block_adds") or 0.0),
        "soft_max_dd_derate": float(raw.get("soft_max_dd_derate") or 0.0),
    }


def soft_config_from_cfg(cfg: Any) -> Dict[str, float]:
    """Pull soft fields from ``ConstitutionConfig`` / kill-switch config objects."""
    return {
        "daily_soft_loss_limit": float(
            getattr(cfg, "daily_soft_loss_limit", 0.0) or 0.0
        ),
        "weekly_soft_loss_limit": float(
            getattr(cfg, "weekly_soft_loss_limit", 0.0) or 0.0
        ),
        "derated_risk_per_slot": float(
            getattr(cfg, "derated_risk_per_slot", 0.0) or 0.0
        ),
        "soft_max_dd_block_adds": float(
            getattr(cfg, "soft_max_dd_block_adds", 0.0) or 0.0
        ),
        "soft_max_dd_derate": float(getattr(cfg, "soft_max_dd_derate", 0.0) or 0.0),
    }


def hard_period_limits_for_evaluate(
    *,
    daily_loss_limit: float,
    weekly_loss_limit: float,
    soft: Mapping[str, float],
) -> Tuple[float, float]:
    """If soft day/week enabled, disable hard halt on that period (limit=1.0)."""
    hard_daily = float(daily_loss_limit)
    hard_weekly = float(weekly_loss_limit)
    if float(soft.get("daily_soft_loss_limit") or 0.0) > 0.0:
        hard_daily = 1.0
    if float(soft.get("weekly_soft_loss_limit") or 0.0) > 0.0:
        hard_weekly = 1.0
    return hard_daily, hard_weekly


def safety_limits_for_evaluate(
    *,
    max_dd: float,
    daily_loss_limit: float,
    weekly_loss_limit: float,
    monthly_loss_limit: float,
    max_turnover_mean: float,
    max_cost_mean: float,
    soft: Mapping[str, float] | None = None,
) -> Dict[str, float]:
    """Build ``evaluate_safety_state`` limits with soft day/week hard-disabled."""
    soft = soft or {}
    hard_daily, hard_weekly = hard_period_limits_for_evaluate(
        daily_loss_limit=float(daily_loss_limit),
        weekly_loss_limit=float(weekly_loss_limit),
        soft=soft,
    )
    return {
        "max_dd": float(max_dd),
        "daily_loss_limit": float(hard_daily),
        "weekly_loss_limit": float(hard_weekly),
        "monthly_loss_limit": float(monthly_loss_limit),
        "max_turnover_mean": float(max_turnover_mean),
        "max_cost_mean": float(max_cost_mean),
    }


def safety_limits_from_cfg(cfg: Any) -> Dict[str, float]:
    """Convenience: limits dict from constitution / kill-switch config object."""
    return safety_limits_for_evaluate(
        max_dd=float(cfg.max_dd),
        daily_loss_limit=float(cfg.daily_loss_limit),
        weekly_loss_limit=float(cfg.weekly_loss_limit),
        monthly_loss_limit=float(cfg.monthly_loss_limit),
        max_turnover_mean=float(cfg.max_turnover_mean),
        max_cost_mean=float(cfg.max_cost_mean),
        soft=soft_config_from_cfg(cfg),
    )


def effective_risk_per_slot(
    *,
    base_risk: float,
    daily_loss: float,
    weekly_loss: float,
    soft: Mapping[str, float],
    peak_drawdown: float | None = None,
) -> Tuple[float, bool]:
    """Return (risk, derated?) after soft day/week and optional soft peak-DD."""
    risk = float(base_risk)
    derated_cap = float(soft.get("derated_risk_per_slot") or 0.0)
    if derated_cap <= 0.0:
        return risk, False
    hit = False
    d_soft = float(soft.get("daily_soft_loss_limit") or 0.0)
    w_soft = float(soft.get("weekly_soft_loss_limit") or 0.0)
    if d_soft > 0.0 and float(daily_loss) >= d_soft:
        risk = min(risk, derated_cap)
        hit = True
    if w_soft > 0.0 and float(weekly_loss) >= w_soft:
        risk = min(risk, derated_cap)
        hit = True
    dd_soft = float(soft.get("soft_max_dd_derate") or 0.0)
    if dd_soft > 0.0 and peak_drawdown is not None and float(peak_drawdown) >= dd_soft:
        risk = min(risk, derated_cap)
        hit = True
    return risk, hit


def soft_peak_dd_block_adds(
    *,
    peak_drawdown: float,
    soft: Mapping[str, float],
) -> bool:
    """True when peak equity DD reaches soft_max_dd_block_adds (adds only)."""
    thr = float(soft.get("soft_max_dd_block_adds") or 0.0)
    return thr > 0.0 and float(peak_drawdown) >= thr


def merge_soft_into_ks_stats(
    ks_stats: MutableMapping[str, Any],
    *,
    soft: Mapping[str, float],
    derated_bars: int,
    soft_dd_block_add_bars: int = 0,
) -> None:
    ks_stats["period_soft"] = {
        "daily_soft_loss_limit": float(soft.get("daily_soft_loss_limit") or 0.0),
        "weekly_soft_loss_limit": float(soft.get("weekly_soft_loss_limit") or 0.0),
        "derated_risk_per_slot": float(soft.get("derated_risk_per_slot") or 0.0),
        "derated_bars": int(derated_bars),
        "soft_max_dd_block_adds": float(soft.get("soft_max_dd_block_adds") or 0.0),
        "soft_max_dd_derate": float(soft.get("soft_max_dd_derate") or 0.0),
        "soft_dd_block_add_bars": int(soft_dd_block_add_bars),
    }
