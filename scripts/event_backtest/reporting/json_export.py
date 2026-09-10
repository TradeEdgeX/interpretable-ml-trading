from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from scripts.event_backtest.reporting.audit import json_safe
from scripts.event_backtest.results import BacktestResult
from scripts.event_backtest.types.stats import tail_contribution_rate
from scripts.event_backtest.types.trade import ClosedTrade


def trade_to_dict(t: ClosedTrade) -> dict:
    """ClosedTrade → JSON-safe dict"""
    return {
        "symbol": t.symbol,
        "side": t.side,
        "archetype": t.archetype,
        "entry_price": round(t.entry_price, 6),
        "exit_price": round(t.exit_price, 6),
        "entry_time": t.entry_time.isoformat(),
        "exit_time": t.exit_time.isoformat(),
        "pnl_r": round(t.pnl_r, 4),
        "pnl_usd": round(t.pnl_usd, 6),
        "pnl_usd_realized": round(t.pnl_usd_realized, 6),
        "notional_usdt": round(t.notional_usdt, 6),
        "qty_base": round(t.qty_base, 10),
        "entry_fee_usdt": round(t.entry_fee_usdt, 6),
        "exit_fee_usdt": round(t.exit_fee_usdt, 6),
        "exit_notional_usdt": round(t.exit_notional_usdt, 6),
        "exit_reason": t.exit_reason,
        "bars_held": t.bars_held,
        "size_multiplier": round(t.size_multiplier, 4),
        "is_add_position": t.is_add_position,
        "is_reverse": t.is_reverse,
        "atr_stop_pct": round(t.atr_stop_pct, 6),
        "effective_stop_pct": round(t.effective_stop_pct, 6),
        "sizing_stop_source": t.sizing_stop_source,
        "breakeven_locked_at_exit": t.breakeven_locked_at_exit,
        "srb_true_sr_level": round(
            float(getattr(t, "srb_true_sr_level", 0.0) or 0.0), 6
        ),
        "srb_true_sr_source": str(getattr(t, "srb_true_sr_source", "") or ""),
        "srb_l1_support": round(float(getattr(t, "srb_l1_support", 0.0) or 0.0), 6),
        "srb_l1_resistance": round(
            float(getattr(t, "srb_l1_resistance", 0.0) or 0.0), 6
        ),
        "srb_l3_lower": round(float(getattr(t, "srb_l3_lower", 0.0) or 0.0), 6),
        "srb_l3_upper": round(float(getattr(t, "srb_l3_upper", 0.0) or 0.0), 6),
    }


def save_json(result: BacktestResult, path: str):
    """保存 JSON 结果 (含完整交易列表)"""
    wins = [t for t in result.trades if t.pnl_r > 0]
    tail_rate, tail_n, winner_n = tail_contribution_rate(result.trades)
    per_arch: dict = {}
    for t in result.trades:
        a = t.archetype or "unknown"
        per_arch.setdefault(a, [])
        per_arch[a].append(t)

    out = {
        "strategy": result.strategy,
        "n_trades": result.n_trades,
        "win_rate": round(result.win_rate, 4),
        "sharpe_r": round(result.sharpe, 4),
        "mean_r": round(result.mean_r, 4),
        "total_r": round(result.total_r, 4),
        "max_drawdown_r": round(result.max_drawdown_r, 4),
        "tail_contribution_rate": round(tail_rate, 4),
        "tail_trade_count": tail_n,
        "winner_count": winner_n,
        "funnel": result.funnel,
        "per_archetype": {
            arch: {
                "n_trades": len(ts),
                "win_rate": (
                    round(sum(1 for t in ts if t.pnl_r > 0) / len(ts), 4) if ts else 0
                ),
                "mean_r": round(sum(t.pnl_r for t in ts) / len(ts), 4) if ts else 0,
                "total_r": round(sum(t.pnl_r for t in ts), 4),
            }
            for arch, ts in per_arch.items()
        },
        "per_symbol": {
            sym: {
                "n_trades": len(trades),
                "win_rate": (
                    round(sum(1 for t in trades if t.pnl_r > 0) / len(trades), 4)
                    if trades
                    else 0
                ),
                "mean_r": (
                    round(sum(t.pnl_r for t in trades) / len(trades), 4)
                    if trades
                    else 0
                ),
                "total_r": round(sum(t.pnl_r for t in trades), 4),
                "per_archetype": {
                    arch: sum(1 for t in trades if t.archetype == arch)
                    for arch in sorted(set(t.archetype for t in trades))
                },
            }
            for sym, trades in result.per_symbol.items()
        },
        "add_position_stats": result.add_position_stats or {},
        "add_trigger_types": result.add_trigger_types,
        "spot_inventory_metrics": json_safe(result.spot_inventory_metrics or {}),
        "spot_benchmarks": json_safe(result.spot_benchmarks or {}),
        "spot_daily_mtm": json_safe(result.spot_daily_mtm or {}),
        "open_positions_end": result.open_positions_end,
        "funnel_per_bar": json_safe(result.funnel_per_bar or []),
        "equity_curve": result.equity_curve,
        "equity_curve_ts": result.equity_curve_ts,
        "constitution_execution_summary": json_safe(
            result.constitution_execution_summary or {}
        ),
        "kill_switch_stats": json_safe(result.kill_switch_stats or {}),
        "trades": [trade_to_dict(t) for t in result.trades],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  📄 Results saved → {path}")


def save_path_efficiency_sidecar(
    result: BacktestResult, anchor_path: Optional[str]
) -> None:
    """与 event_backtest JSON 同目录写入 path_efficiency 分布，供后续分析。"""
    ap = result.add_position_stats
    if not isinstance(ap, dict):
        return
    pe = ap.get("path_efficiency_pct_at_add")
    if not isinstance(pe, dict):
        return
    if not anchor_path:
        return
    slug = (result.strategy or "multi").replace("+", "_").replace(",", "_")
    out_path = Path(anchor_path).with_name(f"path_efficiency_pct_at_add_{slug}.json")
    payload = {
        "strategy": result.strategy,
        "path_efficiency_pct_at_add": json_safe(pe),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n  📄 path_efficiency 分布 → {out_path}")
