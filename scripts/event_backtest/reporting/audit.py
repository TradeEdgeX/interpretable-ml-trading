from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.event_backtest.types.trade import ClosedTrade


def apply_pcm_direction_ffill(
    symbol: str,
    timeframe: str,
    features: Dict[str, float],
    cache: Dict[Tuple[str, str], Dict[str, float]],
    *,
    keys: Tuple[str, ...] = ("ema_1200_position", "roc_20"),
) -> None:
    """因果填充：本 bar 缺列/NaN 时用该 (symbol, tf) 上一有效值，避免 Direction 因键缺失恒为 0。

    ``row_to_features`` 会丢弃 NaN；慢窗特征在部分 bar 上为空时，decide() 收不到
    ``ema_1200_position`` / ``roc_20``，signal_match 与 dual 均失败 —— 与 prefilter 是否通过无关。
    """
    ck = (str(symbol), str(timeframe))
    slot = cache.setdefault(ck, {})
    for k in keys:
        v = features.get(k)
        if v is not None and v == v and np.isfinite(v):
            slot[k] = float(v)
        elif k in slot:
            features[k] = slot[k]


def extract_path_efficiency_pct(features: Mapping[str, Any]) -> Optional[float]:
    """path_efficiency 的滚动历史分位 [0,1]（path_efficiency_pct_f / 列 path_efficiency_pct），语义类似 ER。"""
    for k in ("path_efficiency_pct", "path_efficiency_pct_f"):
        v = features.get(k)
        if v is None:
            continue
        try:
            x = float(v)
            if np.isfinite(x):
                return x
        except (TypeError, ValueError):
            continue
    return None


_ADD_ATTEMPT_CORE_FEATURES: Tuple[str, ...] = (
    "bpc_semantic_chop",
    "bpc_semantic_chop_ts_q",
    "tpc_semantic_chop",
    "tpc_semantic_chop_ts_q",
    "path_efficiency_pct",
    "path_efficiency_pct_f",
    "atr",
    "atr_percentile",
    "atr_percentile_f",
    "volatility_regime_f",
    "ema_200_position",
    "ema_1200_position",
    "macro_tp_vwap_1200_position",
    "vpin_ma20",
    "vpin_ma_max",
    "oi_flow_zscore",
    "oi_zscore",
    "funding_rate",
    "roc_5",
    "roc_20",
    "macd_atr",
    "recent_compression_decay",
    "compression_duration",
    "oi_compression_score",
    "dual_ignition_score",
    "bars_since_local_high",
    "bars_since_local_low",
    "recent_net_move_atr",
    "mfe_r",
    "current_r",
    "parent_initial_r",
    "current_leverage",
    "current_notional_frac",
    "base_leverage_unit",
    "base_notional_frac",
    "add_ml_score",
)


def safe_float_or_none(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def trade_audit_merge_key(
    *,
    symbol: str,
    archetype: str,
    entry_time: Any,
    is_add: bool,
) -> Tuple[str, str, int, bool]:
    """撮合 audit 与 ClosedTrade：(symbol, archetype_ns, utc_ns_int, is_add)。"""
    ct = pd.Timestamp(entry_time)
    if ct.tzinfo is None:
        ct = ct.tz_localize("UTC")
    else:
        ct = ct.tz_convert("UTC")
    return (
        str(symbol).upper().strip(),
        str(archetype or "").strip().lower(),
        int(ct.value),
        bool(is_add),
    )


def closed_trade_to_csv_row(t: ClosedTrade) -> Dict[str, Any]:
    """单行成交导出（不含审计列）。"""
    return {
        "symbol": t.symbol,
        "side": t.side,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "entry_time": t.entry_time.isoformat(),
        "exit_time": t.exit_time.isoformat(),
        "atr": t.atr_at_entry,
        "pnl_r": round(t.pnl_r, 4),
        "pnl_usd": round(t.pnl_usd, 4),
        "pnl_usd_realized": round(t.pnl_usd_realized, 4),
        "notional_usdt": round(t.notional_usdt, 4),
        "qty_base": round(t.qty_base, 10),
        "entry_fee_usdt": round(t.entry_fee_usdt, 6),
        "exit_fee_usdt": round(t.exit_fee_usdt, 6),
        "exit_notional_usdt": round(t.exit_notional_usdt, 4),
        "exit_reason": t.exit_reason,
        "archetype": t.archetype,
        "bars_held": t.bars_held,
        "is_add_position": t.is_add_position,
        "is_reverse": t.is_reverse,
        "size_multiplier": round(t.size_multiplier, 4),
        "atr_stop_pct": round(t.atr_stop_pct, 6),
        "effective_stop_pct": round(t.effective_stop_pct, 6),
        "sizing_stop_source": t.sizing_stop_source,
        "breakeven_locked_at_exit": t.breakeven_locked_at_exit,
    }


def empty_closed_trade_csv_shell() -> Dict[str, Any]:
    """用于仅有 audit 无主成交行时的占位列。"""
    return {
        "symbol": "",
        "side": "",
        "entry_price": float("nan"),
        "exit_price": float("nan"),
        "entry_time": "",
        "exit_time": "",
        "atr": float("nan"),
        "pnl_r": float("nan"),
        "pnl_usd": float("nan"),
        "pnl_usd_realized": float("nan"),
        "notional_usdt": float("nan"),
        "qty_base": float("nan"),
        "entry_fee_usdt": float("nan"),
        "exit_fee_usdt": float("nan"),
        "exit_notional_usdt": float("nan"),
        "exit_reason": "",
        "archetype": "",
        "bars_held": 0,
        "is_add_position": False,
        "is_reverse": False,
        "size_multiplier": float("nan"),
        "atr_stop_pct": float("nan"),
        "effective_stop_pct": float("nan"),
        "sizing_stop_source": "",
        "breakeven_locked_at_exit": False,
    }


def merge_closed_trades_with_audit_rows(
    trades: List[ClosedTrade],
    audits: Optional[List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], int]:
    """将 ``trade_map_audit_rows`` 按成交键并入导出行（成交列优先，不覆盖同名基础列）。"""
    audit_list = list(audits or [])
    if not audit_list:
        out = []
        for t in sorted(trades, key=lambda x: (x.entry_time, x.symbol)):
            row = closed_trade_to_csv_row(t)
            row["audit_matched_to_trade"] = False
            out.append(row)
        return out, 0

    bucket: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for a in audit_list:
        k = trade_audit_merge_key(
            symbol=str(a.get("symbol", "")),
            archetype=str(a.get("archetype", "")),
            entry_time=a.get("entry_timestamp"),
            is_add=bool(a.get("is_add_position")),
        )
        bucket[k].append(dict(a))

    out: List[Dict[str, Any]] = []
    for t in sorted(trades, key=lambda x: (x.entry_time, x.symbol)):
        base = closed_trade_to_csv_row(t)
        tk = trade_audit_merge_key(
            symbol=t.symbol,
            archetype=t.archetype,
            entry_time=t.entry_time,
            is_add=t.is_add_position,
        )
        cand = bucket.get(tk, [])
        audit_flat: Dict[str, Any] = {}
        if cand:
            audit_flat = dict(cand.pop(0))
            if cand:
                audit_flat["audit_remaining_same_key"] = len(cand)
        for ak, av in audit_flat.items():
            if ak in base:
                continue
            base[ak] = av
        base["audit_matched_to_trade"] = bool(audit_flat)
        out.append(base)

    orphan_n = 0
    for _k, lst in bucket.items():
        for orphan in lst:
            orphan_n += 1
            shell = empty_closed_trade_csv_shell()
            shell.update(orphan)
            shell["audit_orphan_only"] = True
            shell["audit_matched_to_trade"] = False
            out.append(shell)
    return out, orphan_n


def trade_audit_row_from_fill(
    *,
    strats: Mapping[str, Any],
    symbol: str,
    archetype: str,
    ts: Any,
    is_add_position: bool,
    entry_source: str,
    features: Mapping[str, Any],
    kill_switch_blocked_at_eval: bool,
    intent_action: str = "",
) -> Dict[str, Any]:
    """单次真实成交快照：时间点、入场源、核心特征列、`_last_funnel` 扁平为 layer_*。"""
    ct = pd.Timestamp(ts)
    if ct.tzinfo is None:
        ct = ct.tz_localize("UTC")
    else:
        ct = ct.tz_convert("UTC")
    arch_lc = str(archetype or "").strip().lower()
    sym_u = str(symbol or "").strip().upper()
    row: Dict[str, Any] = {
        "entry_timestamp": ct.isoformat(),
        "symbol": sym_u,
        "archetype": arch_lc,
        "is_add_position": bool(is_add_position),
        "entry_source": str(entry_source),
        "kill_switch_blocked_at_eval": bool(kill_switch_blocked_at_eval),
    }
    _ia = str(intent_action or "").strip()
    if _ia:
        row["intent_action"] = _ia

    merged = dict(features or {})
    for key in _ADD_ATTEMPT_CORE_FEATURES:
        val = _safe_float_or_none(merged.get(key))
        if val is not None:
            row[f"feat_{key}"] = float(val)

    pct = _extract_path_efficiency_pct(merged)
    if pct is not None and "feat_path_efficiency_pct_f" not in row:
        row["feat_path_efficiency_pct_roll"] = float(pct)

    _extra_prefixes = ("bpc_", "tpc_", "me_", "srb_")
    _budget = 40
    _n = 0
    for k in sorted(merged.keys(), key=str):
        if _n >= _budget:
            break
        sk = str(k)
        if sk in _ADD_ATTEMPT_CORE_FEATURES:
            continue
        if any(sk.startswith(p) for p in _extra_prefixes):
            val = _safe_float_or_none(merged.get(k))
            if val is not None:
                row[f"feat_{sk}"] = float(val)
                _n += 1

    _ohlc_xy = ("close", "high", "low", "open", "volume")
    for hk in _ohlc_xy:
        if hk not in merged:
            continue
        val = _safe_float_or_none(merged.get(hk))
        if val is not None:
            row[f"feat_{hk}"] = float(val)

    st = strats.get(arch_lc)
    lf = getattr(st, "_last_funnel", None) if st is not None else None
    if isinstance(lf, dict):
        for fk, fv in lf.items():
            col = f"layer_{fk}"
            if fk == "gate_reasons" and isinstance(fv, (list, tuple)):
                row[col] = ";".join(str(x) for x in fv)
            elif isinstance(fv, (bool, str)) or fv is None:
                row[col] = fv
            elif isinstance(fv, (int, float)) and fv == fv:
                row[col] = fv
            else:
                row[col] = json.dumps(fv, default=str)
    return row


def add_attempt_snapshot(
    *,
    timestamp: Any,
    symbol: str,
    archetype: str,
    side: str,
    path_type: str,
    features: Mapping[str, Any],
    signal: Mapping[str, Any],
    outcome: str,
) -> Dict[str, Any]:
    """Compact row for add-on rule research sidecars."""
    row: Dict[str, Any] = {
        "timestamp": str(timestamp),
        "symbol": str(symbol),
        "archetype": str(archetype),
        "side": str(side),
        "path_type": str(path_type),
        "outcome": str(outcome or "other"),
        "added": str(outcome or "") == "ok",
    }
    merged = dict(features or {})
    merged.update(dict(signal or {}))
    for key in _ADD_ATTEMPT_CORE_FEATURES:
        val = _safe_float_or_none(merged.get(key))
        if val is not None:
            row[key] = val
    pct = _extract_path_efficiency_pct(merged)
    if pct is not None:
        row["path_efficiency_pct"] = float(pct)
    return row


def er_pct_numeric_summary(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {}
    arr = np.asarray(xs, dtype=float)
    return {
        "n": float(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def er_pct_attempt_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """rows: {pct: Optional[float], outcome: str}"""
    attempts = len(rows)
    with_vals = [float(r["pct"]) for r in rows if r.get("pct") is not None]
    missing = attempts - len(with_vals)
    out: Dict[str, Any] = {
        "attempts": attempts,
        "missing_path_efficiency_pct": missing,
        "with_feature": len(with_vals),
        "overall": _er_pct_numeric_summary(with_vals),
    }
    by_out: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        p = r.get("pct")
        if p is None:
            continue
        by_out[str(r.get("outcome", "unknown"))].append(float(p))
    out["by_outcome"] = {
        k: _er_pct_numeric_summary(v) for k, v in sorted(by_out.items())
    }
    return out


def format_er_pct_summary_lines(
    stats: Dict[str, Any],
    title: str,
) -> List[str]:
    lines: List[str] = [f"    {title}:"]
    if int(stats.get("attempts", 0) or 0) == 0:
        lines.append("      (本路径无加仓尝试)")
        return lines
    lines.append(
        f"      尝试={stats['attempts']}, "
        f"有特征={stats['with_feature']}, "
        f"缺失 path_efficiency_pct={stats['missing_path_efficiency_pct']}"
    )
    ov = stats.get("overall") or {}
    if not ov:
        lines.append(
            "      无有效数值 — 请在策略 features 中包含 path_efficiency_pct_f（→ path_efficiency_pct）"
        )
        return lines
    lines.append(
        f"      分位[0,1]: n={int(ov['n'])} mean={ov['mean']:.3f} std={ov['std']:.3f} "
        f"min={ov['min']:.3f} p10={ov['p10']:.3f} p25={ov['p25']:.3f} "
        f"p50={ov['p50']:.3f} p75={ov['p75']:.3f} p90={ov['p90']:.3f} max={ov['max']:.3f}"
    )
    for ok, sub in (stats.get("by_outcome") or {}).items():
        if not sub or int(sub.get("n", 0) or 0) == 0:
            continue
        lines.append(
            f"      └ outcome={ok}: n={int(sub['n'])} "
            f"p50={sub['p50']:.3f} p10={sub['p10']:.3f} p90={sub['p90']:.3f}"
        )
    return lines


def json_safe(value: Any) -> Any:
    """递归转换为 JSON-safe 值."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


# Legacy private aliases (backtester body unchanged from monolith)
_apply_pcm_direction_ffill = apply_pcm_direction_ffill
_extract_path_efficiency_pct = extract_path_efficiency_pct
_safe_float_or_none = safe_float_or_none
_trade_audit_row_from_fill = trade_audit_row_from_fill
_add_attempt_snapshot = add_attempt_snapshot
_er_pct_numeric_summary = er_pct_numeric_summary
_er_pct_attempt_stats = er_pct_attempt_stats
_format_er_pct_summary_lines = format_er_pct_summary_lines
_json_safe = json_safe
