"""
FER 研究用诊断 — 入场漏斗 + SR 特征快照 + 平仓原因。

启用: 环境变量 MLBOT_FER_DIAG=1
日志: MLBOT_FER_DIAG_LOG（可选，默认当前目录 fer_sr_diag.jsonl）

输出 JSONL，每行一条，kind=entry_eval | exit，便于 grep/jq/pandas 分析。
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from typing import Any, Dict, Optional

_DIAG_ENV = "MLBOT_FER_DIAG"
_LOG_ENV = "MLBOT_FER_DIAG_LOG"
_SAMPLE_ENV = "MLBOT_FER_DIAG_SAMPLE"  # 可选: 0~1，仅抽样写入 entry_eval

# 与「支撑/阻力位」及 prefilter 常见项对齐，用于回答「为何不在 SR 附近开仓」
SR_SNAPSHOT_KEYS: tuple[str, ...] = (
    "dist_to_nearest_sr",
    "sqs_hal_low",
    "sqs_hal_low_pct",
    "sr_strength_max",
    "vp_poc_deviation",
    "fer_signed_efficiency_pct",
    "fer_trapped_shorts_score",
    "vol_mom_10",
    "cvd_change_5_normalized",
    "fer_impulse_failure_direction",
    "fer_impulse_failure_direction_signed",
    "fer_sr_failed_breakout_score_pct",
    "fer_sr_failed_breakout_direction_signed",
    "roc_20",
    "close",
    "atr",
    "ema_200",
    "macro_tp_vwap_1200_position",
)


def fer_diag_enabled() -> bool:
    v = os.environ.get(_DIAG_ENV, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _diag_log_path() -> Optional[str]:
    p = os.environ.get(_LOG_ENV, "").strip()
    if p:
        return p
    if fer_diag_enabled():
        return "fer_sr_diag.jsonl"
    return None


def _diag_sample_rate() -> float:
    raw = os.environ.get(_SAMPLE_ENV, "").strip()
    if not raw:
        return 1.0
    try:
        x = float(raw)
        return max(0.0, min(1.0, x))
    except ValueError:
        return 1.0


def snapshot_sr_features(features: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not features:
        return out
    for k in SR_SNAPSHOT_KEYS:
        v = features.get(k)
        if v is None:
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = str(v)
    return out


def append_fer_diag_record(record: Dict[str, Any]) -> None:
    if not fer_diag_enabled():
        return
    path = _diag_log_path()
    if not path:
        return
    record = dict(record)
    record.setdefault("ts_wall", datetime.now(timezone.utc).isoformat())
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    try:
        abs_path = os.path.abspath(path)
        parent = os.path.dirname(abs_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(abs_path, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass


def record_fer_entry_eval(
    *,
    strategy: str,
    symbol: str,
    signal_ts: Any,
    outcome: str,
    funnel: Dict[str, Any],
    features: Dict[str, Any],
) -> None:
    if str(strategy or "").lower() != "fer":
        return
    if not fer_diag_enabled():
        return
    rate = _diag_sample_rate()
    if rate < 1.0 and outcome != "signal" and random.random() > rate:
        return
    prefilter_ok = funnel.get("prefilter")
    if prefilter_ok is None:
        prefilter_ok = True
    rec: Dict[str, Any] = {
        "kind": "entry_eval",
        "strategy": strategy,
        "symbol": symbol,
        "signal_ts": str(signal_ts) if signal_ts is not None else None,
        "outcome": outcome,
        "prefilter": prefilter_ok,
        "prefilter_reason": funnel.get("prefilter_reason"),
        "direction": funnel.get("direction"),
        "direction_value": funnel.get("direction_value"),
        "direction_rule": funnel.get("direction_rule"),
        "gate": funnel.get("gate"),
        "gate_reasons": funnel.get("gate_reasons"),
        "entry_filter": funnel.get("entry_filter"),
        "gate_weight": funnel.get("gate_weight"),
        "pcm_direction_filter": funnel.get("pcm_direction_filter"),
        "sr_snapshot": snapshot_sr_features(features),
    }
    append_fer_diag_record(rec)


def record_fer_exit(
    *,
    pos: Dict[str, Any],
    close_reason_raw: str,
    exit_reason_normalized: str,
    exit_price: float,
    now: datetime,
    pnl_r: float,
) -> None:
    if not fer_diag_enabled():
        return
    if str(pos.get("archetype", "") or "").lower() != "fer":
        return
    entry_time = pos.get("entry_time")
    hold_minutes: Optional[float] = None
    if isinstance(entry_time, datetime):
        try:
            hold_minutes = (now - entry_time).total_seconds() / 60.0
        except Exception:
            hold_minutes = None
    max_bars = pos.get("max_holding_bars")
    bar_minutes = pos.get("bar_minutes")
    time_stop_minutes: Optional[float] = None
    try:
        if max_bars is not None and bar_minutes is not None:
            mb = int(max_bars)
            bm = int(bar_minutes)
            if mb > 0 and bm > 0:
                time_stop_minutes = float(mb * bm)
    except (TypeError, ValueError):
        time_stop_minutes = None
    rec: Dict[str, Any] = {
        "kind": "exit",
        "symbol": pos.get("symbol"),
        "side": pos.get("side"),
        "exit_reason_raw": close_reason_raw,
        "exit_reason_norm": exit_reason_normalized,
        "exit_price": exit_price,
        "pnl_r": pnl_r,
        "hold_minutes": hold_minutes,
        "bars_held": pos.get("bars_counted", 0),
        "trailing_activated": bool(pos.get("trailing_activated")),
        "breakeven_locked": bool(pos.get("breakeven_locked")),
        "breakeven_enabled": bool(pos.get("breakeven_enabled")),
        "max_holding_bars": max_bars,
        "bar_minutes": bar_minutes,
        "time_stop_limit_minutes": time_stop_minutes,
        "activation_r": pos.get("activation_r"),
        "trail_r": pos.get("trail_r"),
        "take_profit_price": pos.get("take_profit_price"),
        "stop_loss_price": pos.get("stop_loss_price"),
        "entry_time": str(entry_time) if entry_time is not None else None,
        "exit_time": str(now),
    }
    append_fer_diag_record(rec)
