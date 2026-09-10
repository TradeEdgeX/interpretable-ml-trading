"""
SRB 实验辅助：ADX、路径效率、SR 摆动位 — 供事件回测注入 features 与执行层分档。

- ADX / ER 仅使用已收盘 primary bar 的 OHLC 序列（因果）。
- SR 支撑/阻力：入场 bar 前 lookback 根 K 的 low 最小 / high 最大（方案 3b：事件层独立算法）。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


def path_efficiency_last(close: np.ndarray, window: int) -> float:
    """Kaufman 风格路径效率：|C_t - C_{t-w}| / sum(|ΔC|)，取最后一窗。"""
    if close is None or len(close) < window + 1:
        return float("nan")
    seg = close[-(window + 1) :]
    net = abs(float(seg[-1] - seg[0]))
    path = float(np.sum(np.abs(np.diff(seg.astype(float)))))
    if path <= 1e-12:
        return float("nan")
    return float(net / path)


def adx14_last(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> float:
    """Wilder ADX，返回最后一根的值。"""
    n = len(close)
    if n < period + 2 or high is None or low is None:
        return float("nan")
    h = high.astype(float)
    l = low.astype(float)
    c = close.astype(float)
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum.reduce([np.abs(h - l), np.abs(h - prev_c), np.abs(l - prev_c)])
    up = h - np.roll(h, 1)
    down = -(l - np.roll(l, 1))
    up[0] = down[0] = 0.0
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    alpha = 1.0 / float(period)
    tr14 = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean()
    pdi = (
        100.0
        * pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean()
        / tr14.clip(lower=1e-12)
    )
    mdi = (
        100.0
        * pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean()
        / tr14.clip(lower=1e-12)
    )
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).clip(lower=1e-12)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    v = float(adx.iloc[-1])
    return v if np.isfinite(v) else float("nan")


def swing_sr_levels(
    df: pd.DataFrame,
    ts: pd.Timestamp,
    lookback: int,
) -> Tuple[Optional[float], Optional[float]]:
    """
    在 ts 及之前已收盘的 primary bar 上，用最近 lookback 根 K 计算：
    - 支撑：min(low)
    - 阻力：max(high)
    若缺 high/low，则退回 close。
    """
    if df is None or df.empty or lookback < 1:
        return None, None
    try:
        sub = df.loc[:ts].tail(lookback)
    except Exception:
        return None, None
    if sub.empty or len(sub) < max(3, min(lookback, len(sub))):
        return None, None
    if "low" in sub.columns:
        sup = float(sub["low"].astype(float).min())
    elif "close" in sub.columns:
        sup = float(sub["close"].astype(float).min())
    else:
        sup = None
    if "high" in sub.columns:
        res = float(sub["high"].astype(float).max())
    elif "close" in sub.columns:
        res = float(sub["close"].astype(float).max())
    else:
        res = None
    if sup is not None and res is not None and sup > res:
        sup, res = res, sup
    return sup, res


def should_reject_srb_wide_entry(
    side: str,
    close: float,
    atr: float,
    wide_lower_px: Optional[float],
    wide_upper_px: Optional[float],
    min_distance_atr: float,
) -> bool:
    """
    L3 大级别 SR 入场屏蔽：距反向的大级别边界过近时拒绝新开仓。

    LONG：若 ``wide_upper_px`` 在价格上方且 (upper - close) < min_distance_atr × ATR → True。
    SHORT：若 ``wide_lower_px`` 在价格下方且 (close - lower) < min_distance_atr × ATR → True。

    wide_{upper,lower}_px 来源于特征 ``wide_sr_swing_f``（L3，默认 240 bar, anchor_shift=12）。
    min_distance_atr ≤ 0 / 无效价格或 ATR 时不拦截。
    """
    try:
        mn = float(min_distance_atr)
        px = float(close)
        a = float(atr)
    except (TypeError, ValueError):
        return False
    if mn <= 0 or not (px > 0) or not (a > 0):
        return False
    try:
        lo = (
            float(wide_lower_px)
            if wide_lower_px is not None and wide_lower_px == wide_lower_px
            else 0.0
        )
        up = (
            float(wide_upper_px)
            if wide_upper_px is not None and wide_upper_px == wide_upper_px
            else 0.0
        )
    except (TypeError, ValueError):
        lo = up = 0.0

    u = str(side or "").upper()
    if u in ("LONG", "BUY"):
        # 上方宽窗阻力过近 → 拒单
        if up > px and (up - px) < mn * a:
            return True
    elif u in ("SHORT", "SELL"):
        # 下方宽窗支撑过近 → 拒单
        if 0 < lo < px and (px - lo) < mn * a:
            return True
    return False


def resolve_srb_opposite_sr_level(
    side: str,
    *,
    narrow_support: Optional[float],
    narrow_resistance: Optional[float],
    wide_lower_px: Optional[float],
    wide_upper_px: Optional[float],
    source: str = "L1",
) -> Optional[float]:
    """对面结构位：LONG→下方 support，SHORT→上方 resistance。

    ``source``：``L1`` 用窄窗 swing；``L3`` 用 wide_sr 上下沿。
    """

    def _f(x: Any) -> Optional[float]:
        if x is None:
            return None
        try:
            v = float(x)
            return v if v == v and np.isfinite(v) and v > 0 else None
        except (TypeError, ValueError):
            return None

    src = str(source or "L1").strip().upper()
    u = str(side or "").upper()
    if u in ("LONG", "BUY"):
        return _f(wide_lower_px) if src == "L3" else _f(narrow_support)
    if u in ("SHORT", "SELL"):
        return _f(wide_upper_px) if src == "L3" else _f(narrow_resistance)
    return None


def pick_srb_true_sr_level_with_source(
    side: str,
    entry_px: float,
    atr: float,
    *,
    narrow_support: Optional[float],
    narrow_resistance: Optional[float],
    wide_lower_px: Optional[float],
    wide_upper_px: Optional[float],
    fallback_atr: float,
    prefer: Optional[str] = None,
) -> Tuple[float, str]:
    """
    同 ``pick_srb_true_sr_level``，额外返回锚点来源：
      - ``L1``：窄窗 swing（support/resistance）
      - ``L3``：宽窗 wide_sr
      - ``entry``：两侧均不可用，退回入场价

    ``prefer``（``true_sr_level.prefer``）：
      - ``L1``：始终窄窗
      - ``L3``：始终宽窗（缺失则退 L1 / entry）
      - ``L1_then_L3``：窄窗优先；距入场 < fallback_atr×ATR 时改用 L3
      - 未设：legacy —— fallback_atr>0 等价 L1_then_L3，否则 L1
    """
    try:
        ep = float(entry_px)
        ae = float(atr)
        fb = float(fallback_atr)
    except (TypeError, ValueError):
        return float(entry_px or 0), "entry"

    def _f(x: Any) -> Optional[float]:
        if x is None:
            return None
        try:
            v = float(x)
            return v if v == v and np.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    ns = _f(narrow_support)
    nr = _f(narrow_resistance)
    wlo = _f(wide_lower_px)
    wup = _f(wide_upper_px)

    # _srb_true_sr_level 语义：方向侧的"结构参考位"
    #   LONG  → 下方 support（加仓底线 / 结构化 SL 锚点）
    #   SHORT → 上方 resistance（镜像）
    u = str(side or "").upper()
    if u in ("LONG", "BUY"):
        narrow = ns
        wide = wlo
    elif u in ("SHORT", "SELL"):
        narrow = nr
        wide = wup
    else:
        return ep, "entry"

    pref = str(prefer or "").strip().upper().replace("-", "_")
    if not pref:
        pref = "L1_THEN_L3" if fb > 0 else "L1"
    if pref in ("L1_THEN_WIDE", "L1THENL3"):
        pref = "L1_THEN_L3"

    if pref == "L3":
        if wide is not None:
            return float(wide), "L3"
        if narrow is not None:
            return float(narrow), "L1"
        return ep, "entry"

    if narrow is None:
        return ep, "entry"
    pick = float(narrow)
    source = "L1"
    if (
        pref == "L1_THEN_L3"
        and fb > 0
        and ae > 0
        and wide is not None
        and abs(ep - narrow) < fb * ae
    ):
        pick = float(wide)
        source = "L3"
    return float(pick), source


def pick_srb_true_sr_level(
    side: str,
    entry_px: float,
    atr: float,
    *,
    narrow_support: Optional[float],
    narrow_resistance: Optional[float],
    wide_lower_px: Optional[float],
    wide_upper_px: Optional[float],
    fallback_atr: float,
    prefer: Optional[str] = None,
) -> float:
    """
    SRB 真 SR 锚点选择（见 ``pick_srb_true_sr_level_with_source``）。

    用途：结构化止损的对面 SR 与突破确认的 true_sr_level（见 build_position_dict /
    event_backtest.open_position 后写入 ``_srb_true_sr_level``）。

    wide_{upper,lower}_px 来自 ``wide_sr_swing_f``（240 bar, shift=12）。
    """
    px, _src = pick_srb_true_sr_level_with_source(
        side,
        entry_px,
        atr,
        narrow_support=narrow_support,
        narrow_resistance=narrow_resistance,
        wide_lower_px=wide_lower_px,
        wide_upper_px=wide_upper_px,
        fallback_atr=fallback_atr,
        prefer=prefer,
    )
    return px


def resolve_regime_bucket(
    adx: float,
    er: float,
    th: Dict[str, float],
) -> str:
    """返回 bucket key: high_adx_low_er / high_adx_high_er / low_adx_low_er / low_adx_high_er / unknown"""
    adx_hi = th.get("adx_high", 40.0)
    er_hi = th.get("er_high", 0.36)
    if not (adx == adx) or not (er == er):
        return "unknown"
    ah = adx >= adx_hi
    eh = er >= er_hi
    if ah and not eh:
        return "high_adx_low_er"
    if ah and eh:
        return "high_adx_high_er"
    if not ah and not eh:
        return "low_adx_low_er"
    return "low_adx_high_er"


def maybe_inject_srb_experiment_features(
    *,
    df: pd.DataFrame,
    ts: pd.Timestamp,
    exec_raw: Dict[str, Any],
    out: Dict[str, Any],
) -> Dict[str, Any]:
    """
    将 SRB 实验用字段合并到 out（通常为 primary_features）。
    exec_raw: srb archetypes/execution.yaml 的 raw dict。

    只注入 SRB 专有字段：
      - srb_regime_* (ADX/ER bucket)
      - srb_sr_support / srb_sr_resistance  （L1 窄窗，lookback_bars=20）

    L3 大级别 SR（wide_sr_upper_px / wide_sr_lower_px / wide_sr_dist_atr ...）由
    统一的特征 ``wide_sr_swing_f`` 在特征管线里计算，不在此函数里重复。
    """
    re_cfg = (exec_raw or {}).get("regime_execution") or {}
    sr_cfg = (exec_raw or {}).get("sr_structural_exit") or {}
    add_pol = (exec_raw or {}).get("srb_add_position_policy") or {}
    _path_map = add_pol.get("add_path_by_bucket")
    need_regime = bool(re_cfg.get("enabled")) or (
        bool(add_pol.get("enabled"))
        and (
            (
                isinstance(add_pol.get("allow_regime_buckets"), list)
                and len(add_pol.get("allow_regime_buckets") or []) > 0
            )
            or (isinstance(_path_map, Mapping) and bool(_path_map))
        )
    )
    # L1 窄窗 SR：始终注入（结构化 SL / true_sr 锚点 / trading-map 审计）
    need_sr = True

    er_w = int(
        re_cfg.get("er_window_bars") or add_pol.get("regime_er_window_bars") or 20
    )
    adx_p = int(re_cfg.get("adx_period") or add_pol.get("regime_adx_period") or 14)
    lb = int(sr_cfg.get("lookback_bars", 20) or 20)

    if "close" not in df.columns:
        return out
    try:
        hist = df.loc[:ts]
    except Exception:
        return out
    if hist.empty:
        return out

    close = hist["close"].astype(float).values
    if need_regime:
        need = max(er_w + 2, adx_p * 2 + 2)
        if len(close) >= need:
            high = (
                hist["high"].astype(float).values if "high" in hist.columns else close
            )
            low = hist["low"].astype(float).values if "low" in hist.columns else close
            adx_v = adx14_last(high, low, close, period=adx_p)
            er_v = path_efficiency_last(close, er_w)
            if adx_v == adx_v:
                out["srb_regime_adx14"] = float(adx_v)
            if er_v == er_v:
                out["srb_regime_er20"] = float(er_v)
            th = dict(re_cfg.get("thresholds") or {})
            if not th:
                th = dict(add_pol.get("regime_thresholds") or {})
            out["srb_regime_bucket"] = resolve_regime_bucket(
                float(adx_v) if adx_v == adx_v else float("nan"),
                float(er_v) if er_v == er_v else float("nan"),
                {
                    "adx_high": float(th.get("adx_high", 40.0)),
                    "er_high": float(th.get("er_high", 0.36)),
                },
            )

    if need_sr:
        sup, res = swing_sr_levels(df, ts, max(3, lb))
        if sup is not None:
            out["srb_sr_support"] = float(sup)
        if res is not None:
            out["srb_sr_resistance"] = float(res)

    return out


def resolve_srb_add_path(
    features: Mapping[str, Any] | None,
    policy: Mapping[str, Any] | None,
    *,
    default_path: str = "float_ladder",
) -> str:
    """Pick add path from online ``srb_regime_bucket`` (not ops ``active_regime``).

    ``srb_add_position_policy.add_path_by_bucket`` maps bucket → one of:
      - ``float_ladder``: bar-driven float-profit rungs
      - ``signal_add``: PCM re-signal + full trigger validation
      - ``block``: no adds in this bucket

    Missing / disabled policy → ``default_path``. Unknown bucket →
    ``default_add_path`` (policy) or ``default_path``.
    """
    pol = dict(policy or {})
    if not pol or not bool(pol.get("enabled")):
        return str(default_path or "float_ladder")
    by_bucket = pol.get("add_path_by_bucket")
    if not isinstance(by_bucket, Mapping) or not by_bucket:
        return str(default_path or "float_ladder")
    bucket = str((features or {}).get("srb_regime_bucket", "unknown") or "unknown")
    raw = by_bucket.get(bucket)
    if raw is None:
        raw = pol.get("default_add_path", default_path)
    path = str(raw or default_path).strip().lower().replace("-", "_")
    if path in {"float_ladder", "float_r_ladder", "ladder"}:
        return "float_ladder"
    if path in {"signal_add", "signal", "resignal"}:
        return "signal_add"
    if path in {"block", "deny", "off", "none"}:
        return "block"
    return str(default_path or "float_ladder")


def srb_add_position_allowed(
    features: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> Tuple[bool, str]:
    """
    SRB 加仓门控（事件回测 signal-add 与 float_r_ladder 共用）。

    policy (execution.yaml → srb_add_position_policy):
      - enabled: 为 true 时才检查；否则恒通过。
      - allow_regime_buckets: 非空 list 时，仅当 ``srb_regime_bucket`` 在此集合内才允许加仓。
      - add_path_by_bucket: optional per-bucket path (see :func:`resolve_srb_add_path`);
        ``block`` here is also enforced as a deny.
      - max_volume_compression_pct: 当前 ``volume_compression_feature`` **大于** 该阈值则禁止加仓。
      - volume_compression_feature: 默认 ``bpc_volume_compression_pct``。
    """
    if not policy or not bool(policy.get("enabled")):
        return True, ""
    if resolve_srb_add_path(features, policy) == "block":
        return False, "srb_policy_add_path_block"
    allow = policy.get("allow_regime_buckets")
    if isinstance(allow, list) and len(allow) > 0:
        allowed_set = {str(x).strip() for x in allow}
        bucket = str(features.get("srb_regime_bucket", "unknown"))
        if bucket not in allowed_set:
            return False, "srb_policy_regime_bucket"
    max_pct = policy.get("max_volume_compression_pct")
    if max_pct is not None:
        feat_key = str(
            policy.get("volume_compression_feature") or "bpc_volume_compression_pct"
        )
        raw = features.get(feat_key)
        if raw is not None:
            try:
                fv = float(raw)
                if fv == fv and fv > float(max_pct):
                    return False, "srb_policy_volume_compression"
            except (TypeError, ValueError):
                pass
    return True, ""


def should_reject_srb_add_by_shape(
    features: Mapping[str, Any],
    mother_ctx: Mapping[str, Any],
    gate_cfg: Mapping[str, Any],
) -> Tuple[bool, str]:
    """
    SRB 加仓 "事后形态门" (Phase D, 2026-04-22)。

    用于在 ``float_r_ladder_only`` 绕过 validate_add_position_trigger 的情况下，
    对加仓行为追加执行层形态确认。全部子项独立 enable/disable，单测覆盖单项行为。

    gate_cfg (execution.yaml → srb_add_position_policy.post_hoc_shape_gate):
      - retrace_guard.enabled + min_captured_pct (current_r / mfe_r ≥ pct 才允许)
      - recent_momentum.enabled + lookback_bars + min_net_move_atr
      - trend_r2_gate.enabled + min_r2
      - wide_sr_expansion.enabled + min_expansion_atr (与母仓入场时 wide_sr_dist_atr 对比)

    features 期望含（不强制）：
      - mfe_r (已计算的母仓 MFE R 值)
      - current_r (加仓候选的即时 R)
      - trend_r2_20
      - wide_sr_dist_atr
      - recent_net_move_atr  (caller 预算：最近 N bar 同向净位移)
    mother_ctx 期望含：
      - entry_wide_sr_dist_atr (母仓入场时 wide_sr_dist_atr)
      - side (LONG / SHORT)

    Returns:
      (reject, reason)：reject=True 表示拒绝本次加仓，reason 作为 funnel 细分 tag。
    """
    if not gate_cfg:
        return False, ""

    # A. retrace_guard：current_r 相对 MFE 不能缩水过多
    rg = gate_cfg.get("retrace_guard") or {}
    if bool(rg.get("enabled", False)):
        try:
            mfe = float(features.get("mfe_r") or 0.0)
            cur = float(features.get("current_r") or 0.0)
            min_pct = float(rg.get("min_captured_pct", 0.7) or 0.7)
        except (TypeError, ValueError):
            mfe, cur, min_pct = 0.0, 0.0, 0.7
        if mfe > 0 and cur < min_pct * mfe:
            return True, "shape_gate_retrace"

    # B. recent_momentum：近 N bar 的同向净位移 ≥ 阈值
    rm = gate_cfg.get("recent_momentum") or {}
    if bool(rm.get("enabled", False)):
        side = str(mother_ctx.get("side", "")).upper()
        try:
            move = float(features.get("recent_net_move_atr") or 0.0)
            min_move = float(rm.get("min_net_move_atr", 1.5) or 1.5)
        except (TypeError, ValueError):
            move, min_move = 0.0, 1.5
        if side in ("LONG", "BUY") and move < min_move:
            return True, "shape_gate_momentum"
        if side in ("SHORT", "SELL") and move > -min_move:
            return True, "shape_gate_momentum"

    # C. trend_r2_gate：当前 trend_r2_20 ≥ 阈值
    tg = gate_cfg.get("trend_r2_gate") or {}
    if bool(tg.get("enabled", False)):
        try:
            r2 = float(features.get("trend_r2_20") or 0.0)
            min_r2 = float(tg.get("min_r2", 0.4) or 0.4)
        except (TypeError, ValueError):
            r2, min_r2 = 0.0, 0.4
        if r2 < min_r2:
            return True, "shape_gate_r2"

    # D. wide_sr_expansion：加仓时 wide_sr_dist_atr 相对入场扩张 ≥ 阈值
    we = gate_cfg.get("wide_sr_expansion") or {}
    if bool(we.get("enabled", False)):
        try:
            cur_dist = float(features.get("wide_sr_dist_atr") or 0.0)
            entry_dist = float(mother_ctx.get("entry_wide_sr_dist_atr") or 0.0)
            min_exp = float(we.get("min_expansion_atr", 1.0) or 1.0)
        except (TypeError, ValueError):
            cur_dist, entry_dist, min_exp = 0.0, 0.0, 1.0
        if (cur_dist - entry_dist) < min_exp:
            return True, "shape_gate_wide_expansion"

    # E. trend_health_gate（E4, 2026-04-23）：只允许在母仓"已赚到钱 + 未过久"时加仓
    # 语义：
    #   - min_mother_mfe_r：母仓 MFE_r 必须 ≥ 阈值（避免 0 盈利还补仓 → 放大失败）
    #   - max_bars_since_mother_entry：母仓入场超过 N bar 仍无突破性盈利 → 禁加仓
    # features 需含 mfe_r（母仓） + bars_since_mother_entry（caller 计算）。
    th = gate_cfg.get("trend_health_gate") or {}
    if bool(th.get("enabled", False)):
        try:
            min_mfe = float(th.get("min_mother_mfe_r", 1.0) or 1.0)
        except (TypeError, ValueError):
            min_mfe = 1.0
        try:
            max_bars = float(th.get("max_bars_since_mother_entry", 360) or 360)
        except (TypeError, ValueError):
            max_bars = 360.0
        try:
            mother_mfe = float(features.get("mfe_r") or 0.0)
        except (TypeError, ValueError):
            mother_mfe = 0.0
        try:
            bars_since = float(features.get("bars_since_mother_entry") or 0.0)
        except (TypeError, ValueError):
            bars_since = 0.0
        if mother_mfe < min_mfe:
            return True, "shape_gate_trend_health_mfe"
        if max_bars > 0 and bars_since > max_bars:
            return True, "shape_gate_trend_health_stale"

    return False, ""
