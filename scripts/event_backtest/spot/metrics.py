"""Spot KPI, deploy curves, buy-hold benchmarks, funnel audit for event backtest."""

from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from src.time_series_model.live.spot_accum_simple import is_spot_accum_archetype

if TYPE_CHECKING:
    from scripts.event_backtest.engine import ClosedTrade


def ts_utc(value: Any) -> Optional[pd.Timestamp]:
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def compute_spot_account_mtm_components(
    *,
    book_equity_usdt: float,
    simulators: Mapping[str, Any],
    marks_by_symbol: Mapping[str, float],
) -> Dict[str, float]:
    """Return cash + open-inventory MTM without replaying historical fills.

    ``book_equity_usdt`` is initial capital plus realized PnL. Open entry cost
    and its fee are therefore still embedded in book equity; subtracting both
    recovers cash exactly, including after proportional partial harvests.
    """
    open_cost = 0.0
    open_entry_fee = 0.0
    inventory_mtm = 0.0
    estimated_exit_fee = 0.0

    for raw_symbol, simulator in simulators.items():
        symbol = str(raw_symbol).upper()
        mark = float(marks_by_symbol.get(symbol, 0.0) or 0.0)
        fee_rate = max(0.0, float(getattr(simulator, "fee_rate", 0.0) or 0.0))
        positions = getattr(simulator, "_positions", {}) or {}
        for pos in positions.values():
            if not is_spot_accum_archetype(str(pos.get("archetype") or "")):
                continue
            qty = max(0.0, float(pos.get("_qty_base", 0.0) or 0.0))
            cost = max(
                0.0,
                float(pos.get("_entry_notional_usdt", 0.0) or 0.0),
            )
            entry_fee = max(0.0, float(pos.get("_entry_fee_usdt", 0.0) or 0.0))
            if qty <= 0.0:
                continue
            if mark <= 0.0:
                mark = max(0.0, float(pos.get("entry_price", 0.0) or 0.0))
            marked_value = qty * mark
            open_cost += cost
            open_entry_fee += entry_fee
            inventory_mtm += marked_value
            estimated_exit_fee += marked_value * fee_rate

    cash = float(book_equity_usdt) - open_cost - open_entry_fee
    equity = cash + inventory_mtm - estimated_exit_fee
    return {
        "book_equity_usdt": float(book_equity_usdt),
        "cash_usdt": float(cash),
        "inventory_mtm_usdt": float(inventory_mtm),
        "open_cost_usdt": float(open_cost),
        "open_entry_fee_usdt": float(open_entry_fee),
        "estimated_exit_fee_usdt": float(estimated_exit_fee),
        "equity_usdt": float(equity),
    }


def summarize_spot_daily_mtm(rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    """Build true account KPIs from one end-of-UTC-day MTM observation."""
    if not rows:
        return {"status": "empty", "rows": []}

    frame = pd.DataFrame(rows).copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    equity = pd.to_numeric(frame["equity_usdt"], errors="coerce").astype(float)
    valid = np.isfinite(equity.to_numpy()) & (equity.to_numpy() > 0.0)
    frame = frame.loc[valid].reset_index(drop=True)
    if frame.empty:
        return {"status": "empty", "rows": []}

    equity = frame["equity_usdt"].astype(float)
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    max_drawdown_pct = float(drawdown.min())
    peak_giveback_usdt = float((running_peak - equity).max())
    terminal_giveback_usdt = float(running_peak.iloc[-1] - equity.iloc[-1])
    terminal_giveback_pct = (
        terminal_giveback_usdt / float(running_peak.iloc[-1])
        if float(running_peak.iloc[-1]) > 0.0
        else 0.0
    )

    daily_returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return_std = float(daily_returns.std(ddof=1)) if len(daily_returns) > 1 else 0.0
    sharpe = (
        float(daily_returns.mean()) / return_std * sqrt(365.0)
        if return_std > 0.0
        else 0.0
    )
    elapsed_days = max(
        1.0,
        (
            pd.Timestamp(frame["date"].iloc[-1]) - pd.Timestamp(frame["date"].iloc[0])
        ).total_seconds()
        / 86400.0,
    )
    first_equity = float(equity.iloc[0])
    final_equity = float(equity.iloc[-1])
    cagr = (
        (final_equity / first_equity) ** (365.25 / elapsed_days) - 1.0
        if first_equity > 0.0 and final_equity > 0.0
        else 0.0
    )
    calmar = cagr / abs(max_drawdown_pct) if max_drawdown_pct < 0.0 else 0.0
    win_rate = float((daily_returns > 0.0).mean()) if not daily_returns.empty else 0.0

    serial_rows: List[Dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        item = dict(row)
        item["date"] = pd.Timestamp(item["date"]).isoformat()
        serial_rows.append(item)
    return {
        "status": "ok",
        "accounting_basis": "cash_plus_inventory_mtm_net_estimated_exit_fee",
        "n_days": int(len(frame)),
        "start_date": serial_rows[0]["date"],
        "end_date": serial_rows[-1]["date"],
        "initial_equity_usdt": first_equity,
        "final_equity_usdt": final_equity,
        "peak_equity_usdt": float(running_peak.max()),
        "cagr": float(cagr),
        "calmar": float(calmar),
        "win_rate": float(win_rate),
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_daily_annualized": float(sharpe),
        "peak_giveback_usdt": peak_giveback_usdt,
        "peak_giveback_pct": float(abs(max_drawdown_pct)),
        "terminal_giveback_usdt": terminal_giveback_usdt,
        "terminal_giveback_pct": float(terminal_giveback_pct),
        "rows": serial_rows,
    }


def bucket_spot_accum_funnel_row(row: Mapping[str, Any]) -> str:
    """离线 bucket：与 GenericLiveStrategy spot_accum funnel 语义对齐。"""
    if str(row.get("accumulation_policy") or "") == "bull_exposure_stop_deploy":
        return "bull_exposure_stop"
    pf = row.get("prefilter")
    if pf is True:
        return "prefilter_pass"
    if pf is False:
        if bool(row.get("accumulation_transition_override")):
            return "transition_override_path"
        if bool(row.get("prefilter_alignment_override")):
            return "prefilter_recent_alignment_only"
        if not bool(row.get("alignment_used")):
            return "prefilter_hard_deny"
    return "other_unknown"


def compute_spot_accum_accumulation_audit(
    funnel_rows: List[Dict[str, Any]],
    *,
    strategy_key: str = "spot_accum",
) -> Dict[str, Any]:
    """按 PCM eval 行的占比统计（每桶互斥计数 / 总行数）。

    Shares are **evaluation counts** aligned to the recorded ``funnel_per_bar`` cadence,
    not wall-clock durations (intervals vary by timeframe / symbols).
    """
    key_lc = strategy_key.strip().lower()
    rows = [
        r
        for r in (funnel_rows or [])
        if str(r.get("strategy") or "").strip().lower() == key_lc
    ]
    n = len(rows)
    if n <= 0:
        return {
            "status": "no_rows",
            "eval_rows_used": 0,
            "note": (
                "No funnel_per_bar snapshots for spot_accum (strategy idle or PCM "
                "did not propagate _last_funnel)."
            ),
        }
    counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        counts[_bucket_spot_accum_funnel_row(r)] += 1
    denom = float(n)
    shares = {k: float(v) / denom for k, v in sorted(counts.items())}
    return {
        "status": "ok",
        "eval_rows_used": int(n),
        "counts": dict(sorted(counts.items())),
        "shares_eval_count": shares,
        "bucket_definitions": {
            "bull_exposure_stop": "accumulation_policy stops new deploy after bull_seen",
            "transition_override_path": "prefilter fail but accumulation_transition_override",
            "prefilter_recent_alignment_only": "recent prefilter pass alignment without "
            "exclusive transition_override flag",
            "prefilter_pass": "prefilter True on this evaluation",
            "prefilter_hard_deny": "prefilter False and alignment_used=False",
            "other_unknown": "missing/ambiguous funnel fields",
        },
    }


def open_spot_accum_quote_deploy_tails(
    open_rows_end: Optional[List[Dict[str, Any]]],
    *,
    last_equity_ts: pd.Timestamp,
) -> Tuple[
    List[Tuple[pd.Timestamp, float, str]], List[Tuple[pd.Timestamp, float, str]]
]:
    """开仓未平：在回放末尾用 sentinel 时间戳收口 quote delta（仍为半开区间语义）."""
    tails_plus: List[Tuple[pd.Timestamp, float, str]] = []
    tails_minus: List[Tuple[pd.Timestamp, float, str]] = []
    sent = pd.Timestamp(last_equity_ts) + pd.Timedelta(microseconds=1)
    if sent.tzinfo is None:
        sent = sent.tz_localize("UTC")
    else:
        sent = sent.tz_convert("UTC")

    for row in open_rows_end or []:
        if not isinstance(row, dict):
            continue
        pos = row.get("position")
        if not isinstance(pos, dict):
            continue
        if not is_spot_accum_archetype(str(pos.get("archetype", "") or "")):
            continue
        sym_u = str(row.get("symbol") or pos.get("symbol") or "").strip().upper()
        try:
            n = float(pos.get("_spot_quote_deployed", 0.0) or 0.0)
        except (TypeError, ValueError):
            n = 0.0
        if sym_u == "" or n <= 0.0:
            continue
        entry_ts = ts_utc(pos.get("entry_time"))
        if entry_ts is None:
            continue
        tails_plus.append((entry_ts, n, sym_u))
        tails_minus.append((sent, -n, sym_u))
    return tails_plus, tails_minus


def compute_deploy_quote_pct_series(
    trades: List["ClosedTrade"],
    open_rows_end: List[Dict[str, Any]],
    spot_budget: Dict[str, Any],
    equity_ts_iso: List[str],
) -> Dict[str, Any]:
    """与 equity_curve_ts 对齐的 deployed quote（入场名义）占总/分 symbol budget 的比。"""
    if not equity_ts_iso:
        return {"status": "empty_timeline"}

    budgets_raw = spot_budget.get("symbol_budgets_usdt") or {}
    if not isinstance(budgets_raw, dict) or not budgets_raw:
        return {"status": "no_symbol_budgets"}
    budgets: Dict[str, float] = {}
    for k, v in budgets_raw.items():
        kk = str(k or "").strip().upper()
        try:
            budgets[kk] = float(v or 0.0)
        except (TypeError, ValueError):
            continue
    total_budget = float(sum(budgets.values()))
    if total_budget <= 0.0:
        return {"status": "bad_total_budget"}

    spot_closed = [
        t
        for t in (trades or [])
        if is_spot_accum_archetype(str(t.archetype or ""))
        and str(t.symbol or "").strip() != ""
    ]

    equity_ts_list = [ts_utc(x) for x in equity_ts_iso]
    equity_ts_clean = [tt for tt in equity_ts_list if tt is not None]
    if not equity_ts_clean:
        return {"status": "bad_equity_ts"}
    last_eq = equity_ts_clean[-1]

    events: List[Tuple[pd.Timestamp, float, str]] = []
    for t in spot_closed:
        et = ts_utc(t.entry_time)
        xt = ts_utc(t.exit_time)
        if et is None or xt is None:
            continue
        try:
            n = float(t.notional_usdt or 0.0)
        except (TypeError, ValueError):
            n = 0.0
        if n <= 0:
            continue
        sym_u = str(t.symbol or "").strip().upper()
        events.append((et, n, sym_u))
        events.append((xt, -n, sym_u))

    _tp, _tm = _open_spot_accum_quote_deploy_tails(
        open_rows_end, last_equity_ts=last_eq
    )
    events.extend(_tp)
    events.extend(_tm)

    events.sort(key=lambda item: item[0])
    j = 0
    active_by_sym: Dict[str, float] = defaultdict(float)

    pct_total_out: List[float] = []
    pct_sym_out: Dict[str, List[float]] = {sym: [] for sym in sorted(budgets.keys())}

    for ts_eq in equity_ts_clean:
        while j < len(events) and events[j][0] <= ts_eq:
            _tse, dn, dsym = events[j]
            if dsym in budgets:
                active_by_sym[dsym] = max(0.0, active_by_sym[dsym] + float(dn))
            j += 1
        active_total = float(sum(active_by_sym.values()))
        pct_total_out.append(
            (100.0 * active_total / total_budget) if total_budget > 0 else 0.0
        )
        for sym in sorted(budgets.keys()):
            denom = budgets[sym]
            numer = active_by_sym.get(sym, 0.0)
            pct_sym_out[sym].append((100.0 * numer / denom) if denom > 0 else 0.0)

    return {
        "status": "ok",
        "note": (
            "Open quote ~= sum(notional_usdt on still-open trades from merges). Sentinel "
            "close at equity end includes open_positions_end overlays."
        ),
        "curve_ts_iso": [t.isoformat() for t in equity_ts_clean],
        "total_deployed_quote_pct_of_sum_budget": [
            round(float(x), 4) for x in pct_total_out
        ],
        "per_symbol_pct_of_symbol_budget": {
            sym: [round(float(x), 4) for x in xs] for sym, xs in pct_sym_out.items()
        },
    }


def bar_close_asof_backward(
    bars: pd.DataFrame, ts_eval: pd.Timestamp
) -> Optional[float]:
    """无未来函数：用最晚一根 bar.close (index <= ts_eval)。"""
    if bars is None or getattr(bars, "empty", True) or ts_eval is None:
        return None
    bx = bars["close"].astype(float)
    ts = pd.Timestamp(ts_eval)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    mask = bx.index <= ts
    if hasattr(mask, "any") and not bool(mask.any()):
        return None
    out = float(bx.loc[mask].iloc[-1])
    return out if out == out and out > 0.0 else None


def compute_spot_buy_hold_benchmarks(
    *,
    equity_ts_iso: Optional[List[str]],
    bars_by_sym: Dict[str, pd.DataFrame],
    spot_budget: Dict[str, Any],
) -> Dict[str, Any]:
    """BTC 全仓买入持有 + constitution symbol 篮子等权买入持有（无手续费）。

    Curve 与时间轴：**与 equity_curve_ts 等长对齐**；早于 basket 可用的 t0 前保持锚定现金不变。
    """
    if not isinstance(spot_budget, dict):
        return {"status": "no_spot_budget"}
    budgets = spot_budget.get("symbol_budgets_usdt") or {}
    if not isinstance(budgets, dict) or not budgets:
        return {"status": "no_symbol_budgets"}

    try:
        init = float(spot_budget.get("equity_usdt") or 0.0)
    except (TypeError, ValueError):
        init = 0.0
    budget_sum = float(sum(float(v or 0.0) for v in budgets.values()))
    if init <= 0 and budget_sum > 0:
        init = budget_sum
    if init <= 0:
        return {"status": "bad_initial_cash"}

    syms_sorted = sorted(
        str(k).strip().upper() for k in budgets.keys() if str(k).strip()
    )
    btc_sym = "BTCUSDT" if "BTCUSDT" in budgets else ""
    ew_syms = [s for s in syms_sorted if s in (bars_by_sym or {})]

    if not ew_syms:
        return {"status": "missing_bars", "symbols": syms_sorted}

    # t0：所有篮子 symbol 都能在各自 1m bars 中取到收盘价的最晚首日
    dfs = {s: bars_by_sym[s] for s in ew_syms if s in bars_by_sym}
    t0_ok = [
        tt
        for tt in (ts_utc(df.index.min()) for df in dfs.values() if not df.empty)
        if tt is not None
    ]
    if not t0_ok:
        return {"status": "empty_bars_index"}
    t0 = max(t0_ok)

    p0_map: Dict[str, float] = {}
    missing: List[str] = []
    for s in ew_syms:
        df = dfs.get(s)
        if df is None or getattr(df, "empty", True):
            missing.append(s)
            continue
        p0 = _bar_close_asof_backward(df, t0)
        if p0 is None or p0 <= 0:
            missing.append(s)
            continue
        p0_map[s] = p0

    if not p0_map:
        return {
            "status": "no_prices_at_t0",
            "missing_symbols": missing,
            "t0": t0.isoformat(),
        }

    n_ew = float(len([s for s in ew_syms if s in p0_map]))
    per_alloc = init / max(n_ew, 1.0)

    btc_wbase: Optional[float] = None
    if btc_sym and btc_sym in p0_map:
        btc_wbase = float(init) / float(p0_map[btc_sym])

    weights_ew_base: Dict[str, float] = {
        s: (per_alloc / p0_map[s]) for s in ew_syms if s in p0_map
    }

    if equity_ts_iso is None or len(equity_ts_iso) == 0:
        return {"status": "no_equity_timeline", "t0_iso": t0.isoformat()}

    btc_eq: List[float] = []
    ew_eq: List[float] = []
    equity_ts_iso_out: List[str] = []
    for iso in equity_ts_iso:
        ts_ev = ts_utc(iso)
        if ts_ev is None:
            continue

        btc_eq.append(float(init))
        ew_eq.append(float(init))
        equity_ts_iso_out.append(ts_ev.isoformat())

        if ts_ev < t0:
            continue

        if btc_wbase is not None and btc_sym in p0_map:
            df_b = dfs.get(btc_sym)
            if df_b is not None and not getattr(df_b, "empty", True):
                pb = _bar_close_asof_backward(df_b, ts_ev)
                if pb is not None and pb > 0.0:
                    btc_eq[-1] = float(btc_wbase * pb)

        basket_val = 0.0
        for sym_bb, qty_b in weights_ew_base.items():
            df_s = dfs.get(sym_bb)
            if df_s is None or getattr(df_s, "empty", True):
                continue
            px = _bar_close_asof_backward(df_s, ts_ev)
            if px is None or px <= 0.0:
                continue
            basket_val += qty_b * px
        if basket_val > 0.0:
            ew_eq[-1] = float(basket_val)

    btc_fin = btc_eq[-1] if btc_eq else float(init)
    ew_fin = ew_eq[-1] if ew_eq else float(init)
    btc_pct = (100.0 * (btc_fin - init) / init) if init > 1e-9 else 0.0
    ew_pct = (100.0 * (ew_fin - init) / init) if init > 1e-9 else 0.0

    return {
        "status": "ok",
        "note": (
            "Fee-free buy-and-hold benchmarks using 1min close "
            "(as-of backward without lookahead)."
        ),
        "initial_cash_usdt": float(init),
        "t0_iso": t0.isoformat(),
        "curve_ts_iso": equity_ts_iso_out,
        "btc_symbol": btc_sym or None,
        "ew_basket_symbols": ew_syms,
        "symbols_priced_t0": sorted(p0_map.keys()),
        "btc_hold_equity_usdt_curve": btc_eq,
        "ew_hold_equity_usdt_curve": ew_eq,
        "btc_hold_final_equity_usdt": float(btc_fin),
        "btc_hold_total_return_pct": float(btc_pct),
        "ew_hold_final_equity_usdt": float(ew_fin),
        "ew_hold_total_return_pct": float(ew_pct),
    }


def compute_spot_inventory_metrics(
    trades: List["ClosedTrade"], spot_budget: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    if not isinstance(spot_budget, dict):
        return {}
    budgets = spot_budget.get("symbol_budgets_usdt") or {}
    if not isinstance(budgets, dict) or not budgets:
        return {}
    units = spot_budget.get("symbol_unit_notional_usdt") or {}
    total_budget = float(sum(float(v or 0.0) for v in budgets.values()))
    spot_trades = [t for t in trades if is_spot_accum_archetype(str(t.archetype or ""))]

    cutoffs = {
        "2023-01-01": pd.Timestamp("2023-01-01", tz="UTC"),
        "2023-06-01": pd.Timestamp("2023-06-01", tz="UTC"),
    }
    pre_bull: Dict[str, Any] = {}
    for label, cutoff in cutoffs.items():
        per_symbol: Dict[str, float] = {str(sym): 0.0 for sym in budgets.keys()}
        for t in spot_trades:
            entry_ts = ts_utc(t.entry_time)
            exit_ts = ts_utc(t.exit_time)
            if entry_ts is None or exit_ts is None:
                continue
            if entry_ts < cutoff <= exit_ts:
                per_symbol[str(t.symbol)] = per_symbol.get(str(t.symbol), 0.0) + float(
                    t.notional_usdt or 0.0
                )
        open_usdt = float(sum(per_symbol.values()))
        pre_bull[label] = {
            "open_usdt": open_usdt,
            "open_pct": (100.0 * open_usdt / total_budget) if total_budget > 0 else 0.0,
            "per_symbol_usdt": per_symbol,
            "per_symbol_pct": {
                sym: (
                    100.0 * float(per_symbol.get(sym, 0.0)) / float(budget)
                    if float(budget or 0.0) > 0
                    else 0.0
                )
                for sym, budget in budgets.items()
            },
        }

    deploy_curve: Dict[str, Any] = {}
    for sym, budget in budgets.items():
        sym_s = str(sym)
        unit = float((units or {}).get(sym_s, 0.0) or 0.0)
        max_notional = max(
            (float(t.notional_usdt or 0.0) for t in spot_trades if t.symbol == sym_s),
            default=0.0,
        )
        entry_times = [
            ts
            for ts in (ts_utc(t.entry_time) for t in spot_trades if t.symbol == sym_s)
            if ts is not None
        ]
        last_entry = max(entry_times) if entry_times else None
        deploy_curve[sym_s] = {
            "budget_usdt": float(budget or 0.0),
            "max_position_notional_usdt": max_notional,
            "budget_utilization_pct": (
                100.0 * max_notional / float(budget)
                if float(budget or 0.0) > 0
                else 0.0
            ),
            "legs_deployed_est": (max_notional / unit) if unit > 0 else 0.0,
            "tranches_per_symbol": int(spot_budget.get("tranches_per_symbol") or 0),
            "last_entry_ts": last_entry.isoformat() if last_entry is not None else None,
        }

    reason_counts: Dict[str, int] = {}
    for t in spot_trades:
        reason = str(t.exit_reason or "")
        reason_counts[reason] = int(reason_counts.get(reason, 0) or 0) + 1

    return {
        "budget_total_usdt": total_budget,
        "pre_bull_inventory": pre_bull,
        "deploy_curve": deploy_curve,
        "exit_reason_counts": reason_counts,
    }


_ts_utc = ts_utc
_bucket_spot_accum_funnel_row = bucket_spot_accum_funnel_row
_compute_spot_accum_accumulation_audit = compute_spot_accum_accumulation_audit
_open_spot_accum_quote_deploy_tails = open_spot_accum_quote_deploy_tails
_compute_deploy_quote_pct_series = compute_deploy_quote_pct_series
_bar_close_asof_backward = bar_close_asof_backward
_compute_spot_buy_hold_benchmarks = compute_spot_buy_hold_benchmarks
_compute_spot_inventory_metrics = compute_spot_inventory_metrics
