"""
持仓管理共享逻辑 — 实盘 + 事件回测公用

两个核心函数:
  1. build_position_dict: 从 TradeIntent 构建持仓字典 (不含下单)
  2. enforce_position: 7步持仓管理 (不含下单)

使用方式:
  - 实盘 (order_flow_listener):
      pos = build_position_dict(intent, entry_price, atr, bar_minutes)
      reason, exit_price = enforce_position(
          pos, price, price, price, now, primary_tf_atr=features.get("atr")
      )
  - 回测 (event_backtest):
      pos = build_position_dict(intent, entry_price, atr, bar_minutes)
      reason, exit_price = enforce_position(
          pos, bar_high, bar_low, bar_close, now, primary_tf_atr=last_primary_atr
      )
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from src.time_series_model.live.execution_profile_apply import (
    compute_rr_prices,
    holding_expired,
    pick_atr,
)
from src.time_series_model.live.trail_overlay import resolve_trail_overlay

_REGIME_LIFECYCLE_EXIT_MODES = frozenset(
    {"abc_macro_regime_lifecycle", "abc_macro_regime_risk_on"}
)
_REGIME_STATE_MODES = _REGIME_LIFECYCLE_EXIT_MODES | frozenset({"weekly_macro_cycle"})


def _regime_lifecycle_cfg(pos: Dict[str, Any]) -> Dict[str, Any]:
    raw = pos.get("regime_lifecycle_exit")
    return dict(raw) if isinstance(raw, dict) else {}


def update_regime_lifecycle_state(
    pos: Dict[str, Any],
    *,
    macro_regime_score: Optional[float],
    now: Optional[datetime] = None,
) -> None:
    """Track peak score + bull phase while spot_accum position is open."""
    if macro_regime_score is None:
        return
    try:
        rs = float(macro_regime_score)
    except (TypeError, ValueError):
        return
    if rs != rs:
        return
    cfg = _regime_lifecycle_cfg(pos)
    bull_min = float(cfg.get("bull_min_score", 4.0) or 4.0)
    peak = float(pos.get("_regime_peak_score", rs) or rs)
    peak = max(peak, rs)
    pos["_regime_peak_score"] = peak
    if peak >= bull_min:
        if not pos.get("_regime_saw_bull") and now is not None:
            pos["_regime_first_bull_time"] = now
        pos["_regime_saw_bull"] = True
    phase = str(pos.get("_regime_lifecycle_phase") or "accumulating")
    enter_max = float(cfg.get("accumulate_max_score", 2.0) or 2.0)
    if rs < enter_max:
        phase = "accumulating"
    elif pos.get("_regime_saw_bull"):
        phase = "post_bull"
    else:
        phase = "holding"
    pos["_regime_lifecycle_phase"] = phase


def regime_lifecycle_risk_off_exit(
    pos: Dict[str, Any], *, macro_regime_score: Optional[float]
) -> Optional[str]:
    """Exit after bull (peak≥bull_min) on macro risk-off: drop from peak or floor breach."""
    mode = str(pos.get("structural_exit") or "").strip().lower()
    if mode not in _REGIME_LIFECYCLE_EXIT_MODES:
        return None
    cfg = _regime_lifecycle_cfg(pos)
    if cfg and not bool(cfg.get("allow_regime_risk_off", True)):
        return None
    if not cfg and mode == "abc_macro_regime_risk_on":
        try:
            min_sc = float(pos.get("regime_exit_min_score", 5.0) or 5.0)
        except (TypeError, ValueError):
            min_sc = 5.0
        if macro_regime_score is not None:
            try:
                rs = float(macro_regime_score)
                if rs == rs and rs >= min_sc:
                    return "structural_exit_abc_macro_regime_risk_on"
            except (TypeError, ValueError):
                pass
        return None
    if not pos.get("_regime_saw_bull"):
        return None
    if macro_regime_score is None:
        return None
    try:
        rs = float(macro_regime_score)
    except (TypeError, ValueError):
        return None
    if rs != rs:
        return None
    peak = float(pos.get("_regime_peak_score", rs) or rs)
    bull_min = float(cfg.get("bull_min_score", 4.0) or 4.0)
    drop_min = float(cfg.get("risk_off_drop_min", 1.0) or 1.0)
    floor_sc = float(cfg.get("risk_off_floor_score", 3.0) or 3.0)
    try:
        arm_peak = float(cfg.get("arm_risk_off_min_peak", 0) or 0)
    except (TypeError, ValueError):
        arm_peak = 0.0
    if peak < bull_min:
        return None
    # Hold through transition: risk-off only after peak reached arm_risk_off_min_peak (e.g. 5).
    if arm_peak > 0 and peak < arm_peak:
        return None
    if (peak - rs) >= drop_min:
        return "structural_exit_abc_macro_regime_risk_off"
    if rs < floor_sc:
        return "structural_exit_abc_macro_regime_risk_off"
    return None


def weekly_macro_cycle_exit_allowed(
    pos: Dict[str, Any], *, now: Optional[datetime] = None
) -> bool:
    """Gate A-layer cycle death so bear/transition noise cannot liquidate inventory."""
    cfg = _regime_lifecycle_cfg(pos)
    if not bool(cfg.get("cycle_exit_requires_bull", False)):
        return True
    if not bool(pos.get("_regime_saw_bull", False)):
        return False
    bull_min = float(cfg.get("bull_min_score", 4.0) or 4.0)
    try:
        arm_peak = float(cfg.get("arm_cycle_exit_min_peak", bull_min) or bull_min)
    except (TypeError, ValueError):
        arm_peak = bull_min
    peak = float(pos.get("_regime_peak_score", 0.0) or 0.0)
    if peak < arm_peak:
        return False
    min_days_raw = cfg.get("cycle_exit_min_days_after_bull", 0)
    try:
        min_days = float(min_days_raw or 0.0)
    except (TypeError, ValueError):
        min_days = 0.0
    if min_days <= 0:
        return True
    first_bull = pos.get("_regime_first_bull_time")
    if not isinstance(first_bull, datetime) or now is None:
        return False
    elapsed_days = (now - first_bull).total_seconds() / 86400.0
    return elapsed_days >= min_days


def build_position_dict(
    intent: Any,
    entry_price: float,
    atr: float,
    bar_minutes: int = 240,
    entry_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """从 TradeIntent 构建持仓字典 — 实盘/回测公用

    不执行任何下单操作, 只生成持仓数据结构.

    Args:
        intent: TradeIntent (需有 action, symbol, execution_profile, confidence)
        entry_price: 入场价
        atr: 当时的 ATR
        bar_minutes: 信号时钟分钟数 (如 240 = 4h, 60 = 1h)
        entry_time: 入场时间, None 则用 now()

    Returns:
        持仓字典, 可直接存入 _open_positions
    """
    exec_profile = intent.execution_profile or {}
    rr_constraints = exec_profile.get("rr_constraints") or {}
    strategy_specific = exec_profile.get("strategy_specific") or {}

    side = str(intent.action).upper()
    is_long = side in {"LONG", "BUY"}

    stop_loss_r = float(rr_constraints.get("stop_loss_r", 0.0) or 0.0)
    take_profit_r = float(rr_constraints.get("take_profit_r", 0.0) or 0.0)
    min_stop_pct = rr_constraints.get("min_stop_pct")
    max_stop_pct = rr_constraints.get("max_stop_pct")
    # max_holding_bars: 0 表示禁用时间止损 (fat tail 模式)
    _raw_mhb = rr_constraints.get("max_holding_bars")
    max_holding_bars = (
        int(_raw_mhb) if _raw_mhb is not None and int(_raw_mhb) > 0 else 0
    )

    sl_price, tp_price = None, None
    atr_stop_pct = 0.0
    effective_stop_pct = 0.0
    sizing_stop_source = "none"
    # SL 和 TP 独立计算 —— 不要求两者同时 > 0
    if entry_price > 0 and atr > 0 and stop_loss_r > 0:
        atr_stop_pct = max(0.0, (stop_loss_r * atr) / entry_price)
        effective_stop_pct = atr_stop_pct
        if min_stop_pct is not None:
            try:
                effective_stop_pct = max(effective_stop_pct, float(min_stop_pct))
            except Exception:
                pass
        if max_stop_pct is not None:
            try:
                effective_stop_pct = min(effective_stop_pct, float(max_stop_pct))
            except Exception:
                pass
        effective_stop_pct = max(1e-6, effective_stop_pct)
        stop_loss_r = effective_stop_pct * entry_price / atr
        sizing_stop_source = (
            "atr"
            if abs(effective_stop_pct - atr_stop_pct) <= 1e-9
            else "guardrail_clip"
        )
        # 始终计算 SL价格
        computed_sl, computed_tp = compute_rr_prices(
            side=side,
            entry_price=entry_price,
            atr=atr,
            stop_loss_r=stop_loss_r,
            take_profit_r=take_profit_r if take_profit_r > 0 else stop_loss_r,  # dummy
        )
        sl_price = computed_sl
        # TP 仅在启用时设置
        tp_price = computed_tp if take_profit_r > 0 else None

        # ── CRF / consolidation-box execution ─────────────────────────────
        # For range fade, an ATR stop is often inside the box noise. If the
        # intent carries causal box boundaries, anchor SL outside the entry edge
        # and TP near the opposite edge (or box mid), then update the risk
        # distance used for sizing / R accounting.
        _sl_type = str(rr_constraints.get("stop_loss_type") or "").strip().lower()
        _tp_type = str(rr_constraints.get("take_profit_type") or "").strip().lower()
        if _sl_type == "box_edge" or _tp_type in {"opposite_edge", "box_mid"}:
            try:
                box_hi = float(
                    rr_constraints.get("box_hi", rr_constraints.get("box_hi_120"))
                )
                box_lo = float(
                    rr_constraints.get("box_lo", rr_constraints.get("box_lo_120"))
                )
                box_width = box_hi - box_lo
                if (
                    box_hi == box_hi
                    and box_lo == box_lo
                    and box_hi > box_lo > 0.0
                    and box_width > 0.0
                ):
                    if _sl_type == "box_edge":
                        buf = float(
                            rr_constraints.get("box_stop_buffer_frac", 0.25) or 0.25
                        )
                        if is_long:
                            box_sl = box_lo - buf * box_width
                            struct_dist = entry_price - box_sl
                        else:
                            box_sl = box_hi + buf * box_width
                            struct_dist = box_sl - entry_price
                        if struct_dist > 0.0:
                            sl_price = box_sl
                            sizing_stop_source = "box_edge"
                            effective_stop_pct = struct_dist / entry_price
                            stop_loss_r = struct_dist / atr

                    if _tp_type in {"opposite_edge", "box_mid"}:
                        if _tp_type == "box_mid":
                            box_tp = (box_hi + box_lo) * 0.5
                        else:
                            edge_frac = float(
                                rr_constraints.get("box_target_edge_frac", 0.15) or 0.15
                            )
                            box_tp = (
                                box_hi - edge_frac * box_width
                                if is_long
                                else box_lo + edge_frac * box_width
                            )
                        if (is_long and box_tp > entry_price) or (
                            (not is_long) and box_tp < entry_price
                        ):
                            tp_price = box_tp
            except (TypeError, ValueError):
                pass

        # ── SRB 结构化 SL：SL 锚定"对面 SR"（LONG=support, SHORT=resistance）──
        # 语义：SRB 突破 resistance 做 LONG，真正的失效位是下方 support（不是 X×ATR）。
        # 距 entry 过近（SR 紧贴）时兜底到 ATR-based；距 entry 过远时不 clip，
        # 仓位通过 sizing 公式自然缩小以对冲。
        ssl_cfg = rr_constraints.get("structural_sl") or {}
        _opp_sr = strategy_specific.get("srb_opposite_sr_level")
        if ssl_cfg and bool(ssl_cfg.get("enabled", False)) and _opp_sr is not None:
            try:
                _opp_v = float(_opp_sr)
                _buf = float(ssl_cfg.get("opposite_sr_buffer_atr", 0.5) or 0.5) * atr
                _min_dist = float(ssl_cfg.get("min_distance_atr", 2.0) or 2.0) * atr
                if _opp_v == _opp_v and _opp_v > 0:
                    if is_long:
                        struct_sl = _opp_v - _buf
                        struct_dist = entry_price - struct_sl
                    else:
                        struct_sl = _opp_v + _buf
                        struct_dist = struct_sl - entry_price
                    # 紧贴兜底：结构化距离 < 最小距离时，保留 ATR-based SL
                    if struct_dist >= _min_dist:
                        sl_price = struct_sl
                        sizing_stop_source = "structural_opposite_sr"
                        effective_stop_pct = struct_dist / entry_price
                        stop_loss_r = struct_dist / atr
            except (TypeError, ValueError):
                pass

    if entry_time is None:
        entry_time = datetime.now(timezone.utc)
    if entry_time.tzinfo is None:
        entry_time = entry_time.replace(tzinfo=timezone.utc)

    pos: Dict[str, Any] = {
        "symbol": intent.symbol,
        "archetype": str(getattr(intent, "archetype", "") or ""),
        "side": side,
        "entry_price": entry_price,
        "entry_time": entry_time,
        "stop_loss_price": sl_price,
        "take_profit_price": tp_price,
        "allow_trailing": bool(rr_constraints.get("allow_trailing", False)),
        "trailing_atr": rr_constraints.get("trailing_atr"),
        "max_holding_bars": max_holding_bars,
        "atr_at_entry": atr,
        "initial_risk_distance": (
            stop_loss_r * atr if stop_loss_r > 0 and atr > 0 else atr
        ),
        "atr_stop_pct": atr_stop_pct,
        "effective_stop_pct": effective_stop_pct,
        "sizing_stop_source": sizing_stop_source,
        "bar_minutes": bar_minutes,
        "bars_counted": 0,
    }

    _tsr = strategy_specific.get("srb_true_sr_level")
    if _tsr is not None:
        try:
            pos["_srb_true_sr_level"] = float(_tsr)
        except (TypeError, ValueError):
            pass

    # ------------------------------------------------------------------
    # Unified breakeven lock（2026-04-22 重构）
    # 来源：rr_constraints.breakeven_{enabled,trigger_r,lock_level_r,measure}
    # 兼容：若 exec_profile.bpc_position_config 存在旧字段（历史测试固件），仍读取。
    # ------------------------------------------------------------------
    bpc_cfg = exec_profile.get("bpc_position_config") or {}
    be_enabled = bool(
        rr_constraints.get("breakeven_enabled", bpc_cfg.get("breakeven_enabled", False))
    )
    be_trigger_r = float(
        rr_constraints.get(
            "breakeven_trigger_r", bpc_cfg.get("breakeven_trigger_r", 1.0)
        )
        or 1.0
    )
    # 兼容旧 `breakeven_lock_profit_atr`（仅 bpc_position_config 测试固件在用）
    _legacy_lock_atr = bpc_cfg.get("breakeven_lock_profit_atr")
    be_lock_level_r = float(
        rr_constraints.get(
            "breakeven_lock_level_r",
            _legacy_lock_atr if _legacy_lock_atr is not None else 0.0,
        )
        or 0.0
    )
    # 默认 measure：新配置走 initial_risk；若存在 bpc_cfg（历史固件/旧调用方），保持 atr 口径。
    _default_measure = "atr" if bpc_cfg else "initial_risk"
    be_measure = (
        str(
            rr_constraints.get("breakeven_measure", _default_measure)
            or _default_measure
        )
        .strip()
        .lower()
    )
    if be_measure not in {"initial_risk", "atr"}:
        be_measure = "initial_risk"
    pos["breakeven_enabled"] = be_enabled
    pos["breakeven_trigger_r"] = be_trigger_r
    pos["breakeven_lock_level_r"] = be_lock_level_r
    pos["breakeven_measure"] = be_measure
    pos["breakeven_locked"] = False
    if "bar_minutes" in bpc_cfg:
        pos["bar_minutes"] = bpc_cfg["bar_minutes"]

    activation_r = (
        bpc_cfg.get("activation_r") if bpc_cfg else rr_constraints.get("activation_r")
    )
    if activation_r is None:
        activation_r = rr_constraints.get("activation_r")
    if activation_r is None:
        activation_r = rr_constraints.get("trailing_atr")
    trail_r = (bpc_cfg.get("trail_r") if bpc_cfg else None) or rr_constraints.get(
        "trailing_atr"
    )

    if activation_r is not None:
        pos["activation_r"] = float(activation_r)
        pos["trail_r"] = float(trail_r or 1.0)
        pos["trailing_activated"] = False
        pos["high_water_mark"] = entry_price if is_long else None
        pos["low_water_mark"] = entry_price if not is_long else None
    elif rr_constraints.get("allow_trailing", False):
        # 通用 trailing (allow_trailing=True 但无 activation_r)
        pos["activation_r"] = float(rr_constraints.get("trailing_atr", 1.0))
        pos["trail_r"] = float(rr_constraints.get("trailing_atr", 1.0))
        pos["trailing_activated"] = False
        pos["high_water_mark"] = entry_price if is_long else None
        pos["low_water_mark"] = entry_price if not is_long else None

    # SRB L3 dynamic trailing：反向 L3 距离阈值切换 trail_r_far / trail_r_near。
    # 仅当配置了 trail_r_far / trail_r_near / l3_near_threshold_atr 且有 trailing 启用时生效。
    if pos.get("activation_r") is not None:
        _trf = rr_constraints.get("trail_r_far")
        _trn = rr_constraints.get("trail_r_near")
        _thr = rr_constraints.get("l3_near_threshold_atr")
        if _trf is not None:
            try:
                pos["trail_r_far"] = float(_trf)
            except (TypeError, ValueError):
                pass
        if _trn is not None:
            try:
                pos["trail_r_near"] = float(_trn)
            except (TypeError, ValueError):
                pass
        if _thr is not None:
            try:
                pos["l3_near_threshold_atr"] = float(_thr)
            except (TypeError, ValueError):
                pass

    # Optional research trail overlay (default off when omitted / enabled:false)
    _ov = rr_constraints.get("trail_overlay")
    if isinstance(_ov, dict):
        pos["trail_overlay"] = dict(_ov)
        pos["_trail_overlay_pause_left"] = 0

    _sob = rr_constraints.get("scale_out_bank")
    if isinstance(_sob, dict) and _sob:
        pos["scale_out_bank"] = dict(_sob)
        pos["_scale_out_done"] = bool(pos.get("_scale_out_done", False))

    _structural_exit = rr_constraints.get("structural_exit")
    if _structural_exit:
        pos["structural_exit"] = str(_structural_exit)
    _regime_exit_min = rr_constraints.get("regime_exit_min_score")
    if _regime_exit_min is not None:
        try:
            pos["regime_exit_min_score"] = float(_regime_exit_min)
        except (TypeError, ValueError):
            pass
    _rlc = rr_constraints.get("regime_lifecycle_exit")
    if isinstance(_rlc, dict) and _rlc:
        pos["regime_lifecycle_exit"] = dict(_rlc)
    _regime_exit = rr_constraints.get("regime_exit")
    if isinstance(_regime_exit, dict) and _regime_exit:
        pos["regime_exit"] = dict(_regime_exit)
    _cycle_close = rr_constraints.get("cycle_close")
    if isinstance(_cycle_close, dict) and _cycle_close:
        pos["cycle_close"] = dict(_cycle_close)
    _ptl = rr_constraints.get("profit_take_ladder")
    if isinstance(_ptl, dict) and _ptl:
        pos["profit_take_ladder"] = dict(_ptl)
    if str(pos.get("structural_exit") or "").strip().lower() in _REGIME_STATE_MODES:
        pos["_regime_peak_score"] = float(
            rr_constraints.get("entry_regime_score", 0.0) or 0.0
        )
        pos["_regime_lifecycle_phase"] = "accumulating"
        pos["_regime_saw_bull"] = False
    if str(pos.get("structural_exit") or "").strip().lower() == "sr_break_level":
        _sp = rr_constraints.get("sr_exit_price")
        if _sp is not None:
            try:
                pos["sr_exit_price"] = float(_sp)
            except (TypeError, ValueError):
                pos["sr_exit_price"] = None
        pos["sr_exit_buffer_atr"] = float(
            rr_constraints.get("sr_exit_buffer_atr", 0.25) or 0.25
        )

    # Trailing 带宽：可选使用 max(入场ATR, 当前主周期ATR)，避免波动扩张后跟踪过紧
    pos["trail_expand_primary_atr"] = bool(
        rr_constraints.get("trail_expand_primary_atr", False)
    )

    # 2026-04-23 E1: time_stop 分层 — MFE 达阈值则不 time_stop（趋势在跑）
    _uncap = rr_constraints.get("time_stop_uncap_mfe_r")
    if _uncap is not None:
        try:
            pos["time_stop_uncap_mfe_r"] = float(_uncap)
        except (TypeError, ValueError):
            pass

    # 2026-04-23 E2: L3 structural exit（独立 flag；不走 structural_exit 枚举串行）
    if bool(rr_constraints.get("l3_structural_exit_enabled", False)):
        pos["l3_structural_exit_enabled"] = True
        try:
            pos["l3_structural_exit_buffer_atr"] = float(
                rr_constraints.get("l3_structural_exit_buffer_atr", 0.25) or 0.25
            )
        except (TypeError, ValueError):
            pos["l3_structural_exit_buffer_atr"] = 0.25

    return pos


def enforce_position(
    pos: Dict[str, Any],
    *,
    price_high: float,
    price_low: float,
    price_close: float,
    now: datetime,
    default_bar_minutes: int = 240,
    structural_price: Optional[float] = None,
    macro_tp_vwap_position: Optional[float] = None,
    ema_1200_position: Optional[float] = None,
    ema_50_position: Optional[float] = None,
    funding_rate_zscore_50: Optional[float] = None,
    macro_cycle_exit_signal: Optional[float] = None,
    macro_regime_score: Optional[float] = None,
    primary_tf_atr: Optional[float] = None,
    wide_sr_upper_px: Optional[float] = None,
    wide_sr_lower_px: Optional[float] = None,
    trail_overlay_features: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], float]:
    """7步持仓管理 — 实盘/回测公用

    检查顺序 (同 _enforce_open_positions):
      1. Time stop
      2. Unified breakeven lock (MFE ≥ trigger_r × R → SL = entry ± lock_level_r × R, tighten-only)
      3. Update HWM/LWM
      3b. Structural exit (EMA200)
      3c. Structural exit (VWAP1200)
      3d. Structural exit (EMA1200)
      3e. Structural exit (SR break level, SRB)
      3f. Structural exit (weekly macro-cycle signal)
      4. Activation trailing
      5. SL hit (保守: SL 优先于 TP)
      6. TP hit
      7. 返回结果

    价格参数:
      - 实盘: price_high = price_low = price_close = current_price
      - 回测: price_high = bar.high, price_low = bar.low, price_close = bar.close

    注意: 此函数会 **就地修改** pos 字典 (breakeven_locked, high_water_mark,
    trailing_activated, stop_loss_price 等), 与实盘行为一致.

    Args:
        pos: 持仓字典 (build_position_dict 产出)
        price_high: 当前最高价
        price_low: 当前最低价
        price_close: 当前收盘价/最新价
        now: 当前时间
        default_bar_minutes: 默认 bar 分钟数 (兜底)
        structural_price: EMA200 当前值 (仅 structural_exit="ema200" 时使用)
        macro_tp_vwap_position: macro_tp_vwap_1200_position 当前值 (vwap1200 出场)
        ema_1200_position: ema_1200_position 当前值 (ema1200 出场)
        ema_50_position: ema_50_position 当前值 (ema50 出场，公开金叉例子)
        funding_rate_zscore_50: 资金费率 50 根 z 分数（funding_zscore0 出场）
        macro_cycle_exit_signal: 周线宏观出场信号（1=触发 macro-cycle 退出）
        primary_tf_atr: 当前主周期（如 2H）ATR；若 pos.trail_expand_primary_atr 为 True，
            trailing 距离用 max(atr_at_entry, primary_tf_atr)，减轻趋势加速后被窄跟踪洗出。

    Returns:
        (close_reason, exit_price):
          - close_reason is None → 持仓继续
          - close_reason 非 None → 应关闭, exit_price 为成交价
    """
    entry_time = pos.get("entry_time")
    if not isinstance(entry_time, datetime):
        entry_time = datetime.now(timezone.utc)

    side_str = str(pos.get("side", "")).upper()
    is_long = side_str in {"LONG", "BUY"}
    entry_price = pos.get("entry_price", 0) or price_close
    pos_atr = pos.get("atr_at_entry", 0) or 0
    bar_minutes = pos.get("bar_minutes", default_bar_minutes)

    close_reason: Optional[str] = None
    exit_price = price_close

    # ── 1. Time stop（E1 2026-04-23：分层解除）──
    # 若 pos.time_stop_uncap_mfe_r 已配置且当前 MFE_r ≥ 阈值，跳过 time_stop 让趋势继续跑。
    # R 单位按 breakeven_measure 对齐：initial_risk → initial_risk_distance；atr → atr_at_entry。
    _uncap_r = pos.get("time_stop_uncap_mfe_r")
    _skip_time_stop = False
    if _uncap_r is not None and _uncap_r > 0:
        _m = str(pos.get("breakeven_measure", "initial_risk")).strip().lower()
        if _m == "atr":
            _r_unit = float(pos_atr or 0.0)
        else:
            _r_unit = float(pos.get("initial_risk_distance") or pos_atr or 0.0)
        if _r_unit > 0:
            _hwm = pos.get("high_water_mark")
            _lwm = pos.get("low_water_mark")
            if is_long and _hwm is not None:
                _mfe_r = (float(_hwm) - entry_price) / _r_unit
            elif (not is_long) and _lwm is not None:
                _mfe_r = (entry_price - float(_lwm)) / _r_unit
            else:
                _mfe_r = 0.0
            if _mfe_r >= float(_uncap_r):
                _skip_time_stop = True

    if not _skip_time_stop and holding_expired(
        entry_time=entry_time,
        now=now,
        max_holding_bars=pos.get("max_holding_bars"),
        bar_minutes=bar_minutes,
    ):
        close_reason = "time_stop"
        exit_price = price_close

    # ── 2. Unified breakeven lock（2026-04-22 重构：原 2 + 2b 合并）──
    # 语义：MFE ≥ trigger_r × R 时，SL = entry ± lock_level_r × R；tighten-only（硬编码）。
    #   measure="initial_risk"（默认）：R = |SL - entry| at entry（与 structural_sl 兼容）；
    #   measure="atr"：R = 入场时 ATR（BPC 历史口径）。
    # 范围：子仓的 breakeven 由 PositionTracker._sync_child_stop_from_parent 覆盖成母仓 SL，
    #       因此"母仓-only"语义由 inherit_parent_stop 机制天然保证，无需 scope 参数。
    if (
        close_reason is None
        and pos.get("breakeven_enabled")
        and not pos.get("breakeven_locked")
    ):
        be_measure = str(pos.get("breakeven_measure", "initial_risk")).strip().lower()
        if be_measure == "atr":
            r_unit = float(pos_atr or 0.0)
        else:
            r_unit = float(pos.get("initial_risk_distance") or pos_atr or 0.0)
        if r_unit > 0:
            check_price = price_high if is_long else price_low
            if is_long:
                mfe_r = (check_price - entry_price) / r_unit
            else:
                mfe_r = (entry_price - check_price) / r_unit
            trigger_r = float(pos.get("breakeven_trigger_r", 1.0) or 1.0)
            if mfe_r >= trigger_r:
                lock_level_r = float(pos.get("breakeven_lock_level_r", 0.0) or 0.0)
                if is_long:
                    new_sl = entry_price + lock_level_r * r_unit
                else:
                    new_sl = entry_price - lock_level_r * r_unit
                old_sl = pos.get("stop_loss_price")
                # tighten-only（硬编码）：SL 只能向入场有利方向移动
                if (
                    old_sl is None
                    or (is_long and new_sl > old_sl)
                    or (not is_long and new_sl < old_sl)
                ):
                    pos["stop_loss_price"] = new_sl
                pos["breakeven_locked"] = True

    # ── 3. Update high/low water mark ──
    if is_long and pos.get("high_water_mark") is not None:
        if price_high > pos["high_water_mark"]:
            pos["high_water_mark"] = price_high
    elif not is_long and pos.get("low_water_mark") is not None:
        if price_low < pos["low_water_mark"]:
            pos["low_water_mark"] = price_low

    # ── 3b. Structural exit (EMA200) ──
    # BPC trend_hold: 价格穿越 EMA200 = 趋势结构破坏
    # 与 breakeven 逻辑并行，不互相依赖。
    if (
        close_reason is None
        and pos.get("structural_exit") == "ema200"
        and structural_price is not None
        and structural_price > 0
    ):
        if is_long and price_close < structural_price:
            close_reason = "structural_exit_ema200"
            exit_price = price_close
        elif not is_long and price_close > structural_price:
            close_reason = "structural_exit_ema200"
            exit_price = price_close

    # ── 3c. Structural exit (VWAP1200) — pv = macro_tp_vwap_1200_position = (close-vwap)/close
    # 近 VWAP 噪声区由 gate 过滤；执行层仅判断穿越：多 pv<0、空 pv>0。
    if (
        close_reason is None
        and str(pos.get("structural_exit") or "").strip().lower() == "vwap1200"
        and macro_tp_vwap_position is not None
    ):
        try:
            pv = float(macro_tp_vwap_position)
            if not (pv != pv):
                if is_long and pv < 0.0:
                    close_reason = "structural_exit_vwap1200"
                    exit_price = price_close
                elif not is_long and pv > 0.0:
                    close_reason = "structural_exit_vwap1200"
                    exit_price = price_close
        except (TypeError, ValueError):
            pass

    # ── 3d. Structural exit (EMA1200) — ev = ema_1200_position = (close-ema1200)/close
    # 统计依据: EMA1200 零点穿越比 VWAP1200 更可靠 (VWAP/EMA 分歧时 EMA 正确率显著更高)
    if (
        close_reason is None
        and str(pos.get("structural_exit") or "").strip().lower() == "ema1200"
        and ema_1200_position is not None
    ):
        try:
            ev = float(ema_1200_position)
            if not (ev != ev):
                if is_long and ev < 0.0:
                    close_reason = "structural_exit_ema1200"
                    exit_price = price_close
                elif not is_long and ev > 0.0:
                    close_reason = "structural_exit_ema1200"
                    exit_price = price_close
        except (TypeError, ValueError):
            pass

    # ── 3d2. Structural exit (EMA50) — public golden-cross demo
    if (
        close_reason is None
        and str(pos.get("structural_exit") or "").strip().lower() == "ema50"
        and ema_50_position is not None
    ):
        try:
            ev50 = float(ema_50_position)
            if not (ev50 != ev50):
                if is_long and ev50 < 0.0:
                    close_reason = "structural_exit_ema50"
                    exit_price = price_close
                elif not is_long and ev50 > 0.0:
                    close_reason = "structural_exit_ema50"
                    exit_price = price_close
        except (TypeError, ValueError):
            pass

    # ── 3d3. Structural exit (funding z-score back to 0) — public fade demo
    if (
        close_reason is None
        and str(pos.get("structural_exit") or "").strip().lower()
        == "funding_zscore0"
        and funding_rate_zscore_50 is not None
    ):
        try:
            zf = float(funding_rate_zscore_50)
            if not (zf != zf):
                if is_long and zf >= 0.0:
                    close_reason = "structural_exit_funding_zscore0"
                    exit_price = price_close
                elif not is_long and zf <= 0.0:
                    close_reason = "structural_exit_funding_zscore0"
                    exit_price = price_close
        except (TypeError, ValueError):
            pass

    # ── 3e. Structural exit (SR break level) — SRB 专用：跌破冻结支撑 / 涨破冻结阻力
    if close_reason is None and str(
        pos.get("structural_exit") or ""
    ).strip().lower() == ("sr_break_level"):
        lvl = pos.get("sr_exit_price")
        buf_atr = float(pos.get("sr_exit_buffer_atr", 0.25) or 0.25) * float(
            pos_atr or 0.0
        )
        if lvl is not None and pos_atr > 0:
            try:
                lv = float(lvl)
                if lv == lv:
                    thr_long = lv - buf_atr
                    thr_short = lv + buf_atr
                    if is_long and entry_price > lv and price_close < thr_long:
                        close_reason = "structural_exit_sr_break"
                        exit_price = price_close
                    elif not is_long and entry_price < lv and price_close > thr_short:
                        close_reason = "structural_exit_sr_break"
                        exit_price = price_close
            except (TypeError, ValueError):
                pass

    # ── 3f. L3 Structural exit (wide_sr_swing_f) — SRB E2 2026-04-23 ──
    # 语义：L3 大级别 SR 是 SRB 突破的 "宏观边界"，被完全反向击穿即视为趋势结构失效。
    #   - LONG：price_close < wide_sr_lower_px - buffer × ATR（跌穿 L3 支撑）
    #   - SHORT：price_close > wide_sr_upper_px + buffer × ATR（涨穿 L3 阻力）
    # wide_sr_upper_px / wide_sr_lower_px 来自特征 wide_sr_swing_f（240 bar shift=12）。
    if (
        close_reason is None
        and bool(pos.get("l3_structural_exit_enabled", False))
        and pos_atr > 0
    ):
        buf = float(pos.get("l3_structural_exit_buffer_atr", 0.25) or 0.25) * float(
            pos_atr
        )
        try:
            if is_long and wide_sr_lower_px is not None:
                lv = float(wide_sr_lower_px)
                if lv > 0 and lv == lv and price_close < lv - buf:
                    close_reason = "structural_exit_l3"
                    exit_price = price_close
            elif (not is_long) and wide_sr_upper_px is not None:
                uv = float(wide_sr_upper_px)
                if uv > 0 and uv == uv and price_close > uv + buf:
                    close_reason = "structural_exit_l3"
                    exit_price = price_close
        except (TypeError, ValueError):
            pass

    if str(pos.get("structural_exit") or "").strip().lower() in _REGIME_STATE_MODES:
        update_regime_lifecycle_state(
            pos, macro_regime_score=macro_regime_score, now=now
        )

    # ── 3g. Structural exit (weekly macro-cycle signal) ──
    if (
        close_reason is None
        and str(pos.get("structural_exit") or "").strip().lower()
        == "weekly_macro_cycle"
        and macro_cycle_exit_signal is not None
    ):
        try:
            mcs = float(macro_cycle_exit_signal)
            if (
                mcs == mcs
                and mcs >= 0.5
                and weekly_macro_cycle_exit_allowed(pos, now=now)
            ):
                close_reason = "structural_exit_weekly_macro_cycle"
                exit_price = price_close
        except (TypeError, ValueError):
            pass

    # ── 3h. Structural exit (macro regime lifecycle: accumulate → bull → risk-off) ──
    if (
        close_reason is None
        and str(pos.get("structural_exit") or "").strip().lower()
        in _REGIME_LIFECYCLE_EXIT_MODES
    ):
        _lc_reason = regime_lifecycle_risk_off_exit(
            pos, macro_regime_score=macro_regime_score
        )
        if _lc_reason:
            close_reason = _lc_reason
            exit_price = price_close

    # ── 4. Activation trailing ──
    if close_reason is None and pos.get("activation_r") is not None and pos_atr > 0:
        check_price = price_high if is_long else price_low
        if is_long:
            profit_r = (check_price - entry_price) / pos_atr
        else:
            profit_r = (entry_price - check_price) / pos_atr

        activation_r = pos["activation_r"]
        trail_r = pos.get("trail_r", 1.0)
        trail_base_atr = float(pos_atr)
        if pos.get("trail_expand_primary_atr") and primary_tf_atr is not None:
            try:
                pta = float(primary_tf_atr)
                if pta > 0:
                    trail_base_atr = max(trail_base_atr, pta)
            except (TypeError, ValueError):
                pass
        # L3 dynamic trailing：价距反向 L3 近则收到 trail_r_near，远则放到 trail_r_far
        # 仅当 pos 下配置了三项 (trail_r_far / trail_r_near / l3_near_threshold_atr)
        # 且当前 bar 提供了对应侧的 wide_sr price 时生效。
        _trf = pos.get("trail_r_far")
        _trn = pos.get("trail_r_near")
        _thr = pos.get("l3_near_threshold_atr")
        if _trf is not None and _trn is not None and _thr is not None:
            try:
                _ref_px = (
                    float(wide_sr_upper_px)
                    if is_long and wide_sr_upper_px is not None
                    else (
                        float(wide_sr_lower_px)
                        if (not is_long) and wide_sr_lower_px is not None
                        else None
                    )
                )
                if _ref_px is not None and trail_base_atr > 0:
                    if is_long:
                        _rev_dist_atr = (_ref_px - float(price_close)) / trail_base_atr
                    else:
                        _rev_dist_atr = (float(price_close) - _ref_px) / trail_base_atr
                    if _rev_dist_atr < float(_thr):
                        trail_r = float(_trn)
                    else:
                        trail_r = float(_trf)
            except (TypeError, ValueError):
                pass
        if profit_r >= activation_r:
            _was_activated = pos.get("trailing_activated", False)
            if not _was_activated:
                # 首次激活: 标记本 bar 不触发 SL，避免同 bar 激活+触发
                pos["trailing_activated"] = True
                pos["trailing_activation_bar"] = True
            trail_r, pause_ratchet, bank_now, ov_tag = resolve_trail_overlay(
                pos,
                trail_r=float(trail_r),
                is_long=is_long,
                feats=trail_overlay_features,
                trailing_activated=True,
            )
            if ov_tag:
                pos["_trail_overlay_last"] = ov_tag
            if bank_now:
                close_reason = "trail_overlay_bank"
                exit_price = price_close
            elif not pause_ratchet:
                if is_long:
                    hwm = pos.get("high_water_mark", check_price)
                    trail_sl = hwm - trail_r * trail_base_atr
                else:
                    lwm = pos.get("low_water_mark", check_price)
                    trail_sl = lwm + trail_r * trail_base_atr
                old_sl = pos.get("stop_loss_price")
                if old_sl is not None:
                    if is_long and trail_sl > old_sl:
                        pos["stop_loss_price"] = trail_sl
                    elif not is_long and trail_sl < old_sl:
                        pos["stop_loss_price"] = trail_sl
                else:
                    pos["stop_loss_price"] = trail_sl

    # ── 5. SL hit (保守: SL 优先于 TP) ──
    # 首次 trailing 激活的 bar 不触发 SL（避免激活即出场）
    if not pos.pop("trailing_activation_bar", False):
        if close_reason is None:
            sl = pos.get("stop_loss_price")
            if sl is not None:
                sl = float(sl)
                if is_long and price_low <= sl:
                    close_reason = "stop_loss"
                    exit_price = sl
                elif not is_long and price_high >= sl:
                    close_reason = "stop_loss"
                    exit_price = sl

    # ── 6. TP hit ──
    if close_reason is None:
        tp = pos.get("take_profit_price")
        if tp is not None:
            tp = float(tp)
            if is_long and price_high >= tp:
                close_reason = "take_profit"
                exit_price = tp
            elif not is_long and price_low <= tp:
                close_reason = "take_profit"
                exit_price = tp

    return close_reason, exit_price
