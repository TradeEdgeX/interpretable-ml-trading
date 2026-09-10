from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml

from scripts.event_backtest._bootstrap import logger

if TYPE_CHECKING:
    from scripts.event_backtest.simulator.position import PositionSimulator


def _timeframe_to_timedelta(tf: str) -> Optional[pd.Timedelta]:
    """Parse timeframe token like 120T/4H/1D to Timedelta."""
    token = str(tf or "").strip().upper()
    if not token:
        return None
    m = re.fullmatch(r"(\d+)\s*([A-Z]+)", token)
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2)
    unit_map = {
        "T": "min",
        "MIN": "min",
        "M": "min",
        "H": "h",
        "D": "d",
    }
    if unit not in unit_map:
        return None
    try:
        return pd.to_timedelta(n, unit=unit_map[unit])
    except Exception:
        return None


def _align_feature_index_to_bar_close(
    features_df: pd.DataFrame, timeframe: str
) -> pd.DataFrame:
    """Shift feature index from bar-open label to bar-close timestamp."""
    if features_df is None or features_df.empty:
        return features_df
    tf_delta = _timeframe_to_timedelta(timeframe)
    if tf_delta is None or tf_delta <= pd.Timedelta(minutes=1):
        return features_df
    aligned = features_df.copy()
    aligned.index = pd.to_datetime(aligned.index, utc=True) + tf_delta
    return aligned


def _next_open_entry_price(
    tf_df: Optional[pd.DataFrame], decision_ts: Any
) -> Optional[float]:
    """Open of the *next* bar after a close-aligned decision timestamp.

    After ``_align_feature_index_to_bar_close``, deciding at ``ts`` uses the
    just-closed bar. The following row's ``open`` is the next session open —
    knowable at that wall clock, stricter than filling at the decision close.
    """
    if tf_df is None or getattr(tf_df, "empty", True):
        return None
    try:
        ts = pd.Timestamp(decision_ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        future = tf_df.loc[tf_df.index > ts]
    except Exception:
        return None
    if future.empty:
        return None
    raw = future.iloc[0].get("open")
    try:
        px = float(raw)
    except (TypeError, ValueError):
        return None
    if px != px or px <= 0.0:
        return None
    return px


def _iter_update_bars_1min(
    bars_1min: pd.DataFrame,
    prev_ts: pd.Timestamp,
    cur_ts: pd.Timestamp,
    *,
    fast_mode: bool = False,
):
    """Yield 1min bars in (prev_ts, cur_ts] for position updates."""
    del fast_mode  # fast path uses _iter_update_bars_primary_tf instead
    if bars_1min is None or bars_1min.empty:
        return
    mask = (bars_1min.index > prev_ts) & (bars_1min.index <= cur_ts)
    for bar_ts, bar_row in bars_1min[mask].iterrows():
        yield bar_ts, bar_row


def _iter_update_bars_primary_tf(
    sym_bundle: Dict[str, Any],
    prev_ts: pd.Timestamp,
    cur_ts: pd.Timestamp,
    primary_tf: str,
):
    """Yield primary-timeframe feature rows (bar-close index) in (prev_ts, cur_ts].

    Used by ``--fast``: one OHLC update per strategy bar instead of every 1min bar.
    SL/trailing/structural exits are coarser than 1min-exact mode.
    """
    tf_features = sym_bundle.get("tf_features") or {}
    tdf = tf_features.get(primary_tf)
    if tdf is None or getattr(tdf, "empty", True):
        return
    mask = (tdf.index > prev_ts) & (tdf.index <= cur_ts)
    for bar_ts, bar_row in tdf.loc[mask].iterrows():
        yield bar_ts, bar_row


def _ohlc_dict_from_bar_row(bar_ts: pd.Timestamp, bar_row: pd.Series) -> Dict[str, Any]:
    """Feature / OHLC row → bar dict for PositionSimulator.update."""
    return {
        "timestamp": bar_ts,
        "open": float(bar_row.get("open", bar_row.get("close", 0)) or 0),
        "high": float(bar_row.get("high", bar_row.get("close", 0)) or 0),
        "low": float(bar_row.get("low", bar_row.get("close", 0)) or 0),
        "close": float(bar_row.get("close", 0) or 0),
    }


def _feature_asof_from_sym_tf_features(
    sym_entry: Dict[str, Any],
    bar_ts: Any,
    column: str,
) -> Optional[float]:
    """Pick latest ``column`` from any timeframe row with index <= ``bar_ts``.

    Multi-symbol timeline: structural inputs must follow **the symbol being
    updated**, not only the symbol of the current PCM event. Previously only
    ``simulators[sym_event]._macro_tp_vwap_position`` was refreshed, so ADA
    could keep stale/BTC macro values during 1m ``update()`` → vwap1200 deadband
    rarely matched reality (late / missing structural exits).
    """
    tfd = sym_entry.get("tf_features") or {}
    best_ix: Optional[pd.Timestamp] = None
    best_val: Optional[float] = None
    for _tf, tdf in tfd.items():
        if tdf is None or getattr(tdf, "empty", True):
            continue
        if column not in tdf.columns:
            continue
        try:
            sub = tdf.loc[tdf.index <= bar_ts]
        except Exception:
            continue
        if sub.empty:
            continue
        ts_last = sub.index[-1]
        raw = sub[column].iloc[-1]
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if v != v:  # NaN
            continue
        if best_ix is None or ts_last > best_ix:
            best_ix = ts_last
            best_val = v
    return best_val


def _feature_row_asof_from_sym_tf_features(
    sym_entry: Dict[str, Any],
    bar_ts: Any,
    *,
    require_macro: bool = True,
) -> Optional[pd.Series]:
    """Latest feature row with index <= ``bar_ts`` (max timestamp across TFs).

    When ``require_macro`` is True, only consider frames that expose
    ``macro_tp_vwap_1200_position`` and rows with a finite value, so the
    returned row can drive both stored position and frozen VWAP level.
    """
    tfd = sym_entry.get("tf_features") or {}
    best_ix: Optional[pd.Timestamp] = None
    best_row: Optional[pd.Series] = None
    for _tf, tdf in tfd.items():
        if tdf is None or getattr(tdf, "empty", True):
            continue
        if require_macro and "macro_tp_vwap_1200_position" not in tdf.columns:
            continue
        try:
            sub = tdf.loc[tdf.index <= bar_ts]
        except Exception:
            continue
        if sub.empty:
            continue
        ts_last = sub.index[-1]
        row = sub.iloc[-1]
        if require_macro:
            raw = row.get("macro_tp_vwap_1200_position")
            try:
                pv = float(raw)
            except (TypeError, ValueError):
                continue
            if pv != pv:
                continue
        if best_ix is None or ts_last > best_ix:
            best_ix = ts_last
            best_row = row
    return best_row


def _sync_macro_tp_vwap_from_feature_row(
    sim: "PositionSimulator",
    row: Optional[pd.Series],
) -> None:
    """Set simulator macro position + frozen typical-price VWAP from one feature row.

    ``macro_tp_vwap_1200_position`` = (close - vwap) / close on the decision bar.
    Between primary-TF bar closes, VWAP level is held fixed so each 1m close can
    recompute pv = (close_1m - vwap) / close_1m — otherwise crossing the deadband
    between 2H updates would never be seen (stale pv).
    """
    if row is None:
        return
    try:
        mv = row.get("macro_tp_vwap_1200_position")
        if mv is None:
            return
        pv = float(mv)
        if pv != pv:
            return
        sim._macro_tp_vwap_position = pv
        cfeat = row.get("close")
        if cfeat is None:
            sim._macro_tp_vwap_level = None
            return
        c2 = float(cfeat)
        if c2 <= 0:
            sim._macro_tp_vwap_level = None
            return
        sim._macro_tp_vwap_level = c2 * (1.0 - pv)
    except (TypeError, ValueError):
        pass


def _sync_ema_1200_from_feature_row(
    sim: "PositionSimulator",
    row: Optional[pd.Series],
) -> None:
    """Set simulator EMA1200 position + frozen EMA1200 level from one feature row.

    Same mechanism as VWAP1200: freeze the EMA1200 price level at primary-TF
    bar close, recompute position on each 1m bar in between.
    """
    if row is None:
        return
    try:
        mv = row.get("ema_1200_position")
        if mv is None:
            return
        ev = float(mv)
        if ev != ev:
            return
        sim._ema_1200_position = ev
        cfeat = row.get("close")
        if cfeat is None:
            sim._ema_1200_level = None
            return
        c2 = float(cfeat)
        if c2 <= 0:
            sim._ema_1200_level = None
            return
        sim._ema_1200_level = c2 * (1.0 - ev)
    except (TypeError, ValueError):
        pass


def _sync_weekly_ema_200_from_feature_row(
    sim: "PositionSimulator",
    row: Optional[pd.Series],
) -> None:
    """Spot regime_exit / regime-aware ladder: sync weekly_ema_200_position."""
    if row is None:
        return
    try:
        mv = row.get("weekly_ema_200_position")
        if mv is None:
            sim._weekly_ema_200_position = None
            return
        wv = float(mv)
        if wv != wv:
            sim._weekly_ema_200_position = None
            return
        sim._weekly_ema_200_position = wv
    except (TypeError, ValueError):
        pass


def row_to_features(row: pd.Series) -> Dict[str, float]:
    """DataFrame 行 → 特征 dict"""
    features = {}
    for k, v in row.items():
        try:
            if v is not None and np.isscalar(v) and not pd.isna(v):
                features[str(k)] = float(v)
        except (ValueError, TypeError):
            continue
    return features


def _tf_to_minutes(tf: str) -> int:
    """'15T' → 15, '240T' → 240, '1D' → 1440, '4H' → 240."""
    td = _timeframe_to_timedelta(tf)
    if td is not None:
        mins = int(td / pd.Timedelta(minutes=1))
        if mins > 0:
            return mins
    token = str(tf or "").strip().upper()
    if token.endswith("T"):
        return int(token[:-1])
    if token.endswith("MIN"):
        return int(token[:-3])
    return int(token)


def _timeframe_from_strategy_meta(strategy: str, strategies_root: str) -> Optional[str]:
    """从策略目录 meta.yaml 读取 timeframe（与 run_live / backtest_execution_layer 对齐）。"""
    import yaml

    meta_path = Path(strategies_root) / strategy / "meta.yaml"
    if not meta_path.is_file():
        return None
    try:
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        tf = meta.get("timeframe")
        if isinstance(tf, str) and tf.strip():
            return tf.strip()
        st = meta.get("strategy")
        if isinstance(st, dict):
            tf = st.get("timeframe")
            if isinstance(tf, str) and tf.strip():
                return tf.strip()
    except Exception:
        return None
    return None


def _get_bar_minutes(
    strategy: str, *, strategies_root: str = "config/strategies"
) -> int:
    """策略 → 信号时钟分钟数"""
    return _tf_to_minutes(_get_timeframe(strategy, strategies_root=strategies_root))


# 缺失 meta timeframe 时每个 (strategies_root, strategy) 只 warn 一次
_TIMEFRAME_FALLBACK_WARNED: Set[Tuple[str, str]] = set()


def _get_timeframe(strategy: str, *, strategies_root: str = "config/strategies") -> str:
    """策略 → timeframe：仅策略目录 meta.yaml（顶层或 strategy.timeframe）；缺失则 240T 并打一次 warning。"""
    meta_tf = _timeframe_from_strategy_meta(strategy, strategies_root)
    if meta_tf:
        return meta_tf
    key = (strategies_root, strategy)
    if key not in _TIMEFRAME_FALLBACK_WARNED:
        _TIMEFRAME_FALLBACK_WARNED.add(key)
        logger.warning(
            "strategy %r: no timeframe in %s/%s/meta.yaml — using 240T",
            strategy,
            strategies_root,
            strategy,
        )
    return "240T"
