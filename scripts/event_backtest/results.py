from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.event_backtest.reporting.audit import (
    format_er_pct_summary_lines,
    merge_closed_trades_with_audit_rows,
)
from scripts.event_backtest.types.stats import tail_contribution_rate
from scripts.event_backtest.types.trade import ClosedTrade


@dataclass
class BacktestResult:
    strategy: str
    trades: List[ClosedTrade] = field(default_factory=list)
    funnel: Dict[str, int] = field(default_factory=dict)
    per_symbol: Dict[str, List[ClosedTrade]] = field(default_factory=dict)
    # 1min bar data per symbol (for trading map)
    bars_1min: Dict[str, pd.DataFrame] = field(default_factory=dict)
    # Kill switch 模拟统计
    kill_switch_stats: Optional[Dict[str, Any]] = None
    # 风险 equity curve（与 equity_curve_ts 等长；时间戳 ISO8601 UTC）
    equity_curve: Optional[List[float]] = None
    equity_curve_ts: Optional[List[str]] = None
    # 宪法层摘要（供交易地图 HTML 展示；非完整 constitution dump）
    constitution_execution_summary: Optional[Dict[str, Any]] = None
    # 加仓模拟统计
    add_position_stats: Optional[Dict[str, Any]] = None
    # 月末未平仓快照 (用于跨月续跑)
    open_positions_end: List[Dict[str, Any]] = field(default_factory=list)
    # 各注册策略 execution.add_position.trigger.type（小写 key）
    add_trigger_types: Dict[str, str] = field(default_factory=dict)
    # 时间线上每次 PCM 评估后的策略漏斗快照（用于交易地图附图）
    funnel_per_bar: List[Dict[str, Any]] = field(default_factory=list)
    # 每次加仓尝试的特征快照（可选 sidecar 导出，用于规则研究）
    add_attempt_rows: List[Dict[str, Any]] = field(default_factory=list)
    # 每次实际成交的母仓/加仓时刻：特征 + 策略 _last_funnel 各层布尔与 reason（可与 trading map 对照）
    trade_map_audit_rows: List[Dict[str, Any]] = field(default_factory=list)
    # spot_accum inventory-first KPI（pre-bull inventory / deploy curve / exit mix）
    spot_inventory_metrics: Optional[Dict[str, Any]] = None
    # BTC / EW basket fee-free benchmarks vs same anchor + timeline as equity_curve_ts
    spot_benchmarks: Optional[Dict[str, Any]] = None
    # UTC daily account equity: cash + open inventory MTM, net estimated exit fee.
    spot_daily_mtm: Optional[Dict[str, Any]] = None
    margin_mode: str = "usd_m"
    equity_anchor_usdt: float = 1000.0

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def pnl_r_array(self) -> np.ndarray:
        return np.array([t.pnl_r for t in self.trades]) if self.trades else np.array([])

    @property
    def win_rate(self) -> float:
        arr = self.pnl_r_array
        return float(np.mean(arr > 0)) if len(arr) > 0 else 0.0

    @property
    def sharpe(self) -> float:
        arr = self.pnl_r_array
        if len(arr) < 2:
            return 0.0
        return (
            float(np.mean(arr) / np.std(arr, ddof=1))
            if np.std(arr, ddof=1) > 0
            else 0.0
        )

    @property
    def mean_r(self) -> float:
        arr = self.pnl_r_array
        return float(np.mean(arr)) if len(arr) > 0 else 0.0

    @property
    def total_r(self) -> float:
        return float(np.sum(self.pnl_r_array)) if self.trades else 0.0

    @property
    def max_drawdown_r(self) -> float:
        arr = self.pnl_r_array
        if len(arr) == 0:
            return 0.0
        cum = np.cumsum(arr)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        return float(np.max(dd))

    def print_report(self):
        """输出汇总报告"""
        arr = self.pnl_r_array
        print()
        print("=" * 72)
        print(f"  📊 事件驱动回测报告: {self.strategy.upper()}")
        print("=" * 72)
        print(f"  交易数:       {self.n_trades}")
        print(f"  胜率:         {self.win_rate:.1%}")
        print(f"  Sharpe (R):   {self.sharpe:.4f}")
        print(f"  Mean R:       {self.mean_r:.4f}")
        print(f"  Total R:      {self.total_r:.2f}")
        print(f"  Max DD (R):   {self.max_drawdown_r:.2f}")
        if len(arr) > 0:
            print(f"  Best trade:   {arr.max():.2f}R")
            print(f"  Worst trade:  {arr.min():.2f}R")
        tail_rate, tail_n, winner_n = tail_contribution_rate(self.trades)
        if winner_n > 0:
            print(f"  Tail contrib: {tail_rate:.1%}  (top {tail_n}/{winner_n} winners)")

        # 出场原因分布
        reasons = defaultdict(int)
        for t in self.trades:
            reasons[t.exit_reason] += 1
        if reasons:
            print(f"\n  出场原因:")
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                print(f"    {reason:20s}: {count:4d} ({count/self.n_trades:.1%})")

        # 每 symbol 明细
        if self.per_symbol:
            print(f"\n  Per-symbol:")
            print(
                f"    {'Symbol':12s} {'Trades':>7s} {'WinRate':>8s} {'MeanR':>8s} {'TotalR':>8s}"
            )
            for sym in sorted(self.per_symbol.keys()):
                strades = self.per_symbol[sym]
                sarr = np.array([t.pnl_r for t in strades])
                swr = float(np.mean(sarr > 0)) if len(sarr) > 0 else 0.0
                print(
                    f"    {sym:12s} {len(strades):>7d} {swr:>7.1%} "
                    f"{np.mean(sarr):>8.3f} {np.sum(sarr):>8.2f}"
                )

        # 漏斗
        if self.funnel:
            print(f"\n  信号漏斗:")
            for k, v in self.funnel.items():
                print(f"    {k:30s}: {v}")

        # Kill switch 模拟统计
        if self.kill_switch_stats:
            ks = self.kill_switch_stats
            print(f"\n  🚨 Kill Switch 模拟:")
            print(f"    触发次数: {ks.get('trigger_count', 0)}")
            print(f"    跳过入场: {ks.get('trades_skipped', 0)}")
            print(f"    实际执行: {ks.get('trades_executed', 0)}")
            for trig in ks.get("triggers", [])[:5]:
                print(
                    f"    │ {trig['timestamp']}: {', '.join(trig['reasons'])} (eq=${trig['equity']:.0f})"
                )

        # 风险 Equity Curve 摘要
        if self.equity_curve and len(self.equity_curve) > 1:
            final_eq = self.equity_curve[-1]
            peak_eq = max(self.equity_curve)
            anchor = float(self.equity_anchor_usdt or self.equity_curve[0] or 1000.0)
            ret_pct = (final_eq - self.equity_curve[0]) / self.equity_curve[0] * 100
            max_dd_eq = 0.0
            peak = self.equity_curve[0]
            for eq in self.equity_curve:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak if peak > 0 else 0.0
                if dd > max_dd_eq:
                    max_dd_eq = dd
            if self.margin_mode == "coin_m":
                print(f"\n  💰 Wallet MTM equity (anchor ${anchor:.0f}):")
                print(
                    "    口径: wallet×collateral_price + 未平仓浮盈 "
                    "（含抵押物 beta；非 capital_report 的 strategy_realized）"
                )
            else:
                print(f"\n  💰 Risk-based equity (anchor ${anchor:.0f}):")
            print(f"    Final: ${final_eq:.0f} ({ret_pct:+.1f}%)")
            print(f"    Peak: ${peak_eq:.0f}")
            print(f"    Max DD: {max_dd_eq:.1%}")

        # 加仓统计
        if self.add_position_stats:
            ap = self.add_position_stats
            print("\n  📈 加仓模拟 (per_strategy_limits + execution.add_position):")
            print(f"    加仓成功: {ap.get('add_count', 0)} 次")
            print(f"    加仓拒绝: {ap.get('rejected_count', 0)} 次")
            print(f"    加仓交易: {ap.get('add_trades', 0)} 笔")
            if ap.get("add_trades", 0) > 0:
                print(f"    加仓 Mean R: {ap.get('add_mean_r', 0):.4f}")
                print(f"    加仓 Win%: {ap.get('add_win_rate', 0):.1%}")
                print(f"    加仓平均倍率: {ap.get('add_mean_size', 0):.2f}x")
                print(f"    观测最大杠杆: {ap.get('max_observed_leverage', 0):.2f}x")
            if isinstance(ap.get("path_efficiency_pct_at_add"), dict):
                print(
                    "\n  📐 path_efficiency_pct 分位分布 → 见脚本末尾（避免被交易地图日志顶掉）"
                )

        if self.open_positions_end:
            print(f"\n  ♻️  月末未平仓: {len(self.open_positions_end)}")

        if self.spot_inventory_metrics:
            inv = self.spot_inventory_metrics
            pb = (inv.get("pre_bull_inventory") or {}).get("2023-01-01") or {}
            print("\n  🧺 Spot inventory KPI:")
            print(
                "    2023-01 inventory: "
                f"${float(pb.get('open_usdt', 0.0) or 0.0):.0f} "
                f"({float(pb.get('open_pct', 0.0) or 0.0):.1f}%)"
            )
            ro = int(
                (inv.get("exit_reason_counts") or {}).get(
                    "structural_exit_abc_macro_regime_risk_off", 0
                )
                or 0
            )
            print(f"    risk_off exits: {ro}")
            aud = inv.get("accumulation_audit") or {}
            sh = aud.get("shares_eval_count") or {}
            if isinstance(sh, dict) and sum(float(v or 0) for v in sh.values()) > 0:
                nrows = aud.get("eval_rows_used", 0)
                print("\n  🧭 spot_accum funnel (share of PCM eval rows):")
                print(f"    funnel rows (spot_accum): {int(nrows)}")
                for k in (
                    "bull_exposure_stop",
                    "transition_override_path",
                    "prefilter_recent_alignment_only",
                    "prefilter_pass",
                    "prefilter_hard_deny",
                    "other_unknown",
                ):
                    if k in sh:
                        print(f"    {k:36s}: {float(sh[k] or 0.0):.1%}")

        if self.spot_benchmarks:
            bm = self.spot_benchmarks
            if bm.get("status") == "ok":
                print(
                    "\n  📈 Spot benchmarks (fee-free, anchored to constitution cash):"
                )
                print(
                    f"    btc_buy_hold_final: "
                    f"${float(bm.get('btc_hold_final_equity_usdt') or 0.0):.0f}"
                    f" ({float(bm.get('btc_hold_total_return_pct') or 0.0):+.1f}% vs t0)"
                )
                ew = bm.get("ew_basket_symbols") or []
                print(
                    f"    ew_basket_hold_final ({'+'.join(ew)}): "
                    f"${float(bm.get('ew_hold_final_equity_usdt') or 0.0):.0f}"
                    f" ({float(bm.get('ew_hold_total_return_pct') or 0.0):+.1f}% vs t0)"
                )
                print(
                    f"    benchmark anchor cash: ${float(bm.get('initial_cash_usdt') or 0.0):.0f}"
                )

        self._print_add_position_diagnostics_footer()

        print("=" * 72)

    def _print_add_position_diagnostics_footer(self) -> None:
        """置底简短加仓诊断（pipeline run_step 只打 stdout 末段时仍能看见）。"""
        fn = self.funnel or {}
        print("\n  🔎 加仓诊断摘要 (execution.add_position / PCM slot)")
        if self.add_trigger_types:
            for sk, tv in sorted(self.add_trigger_types.items()):
                print(f"    trigger.type [{sk}]: {tv}")
        keys = [
            ("total_signals_checked", "时间线检查次数"),
            ("signals_generated", "产生意图次数"),
            ("reject_pcm_slot_full", "PCM全局/策略槽满(drop_slot)"),
            ("reject_pcm_trend_symbol_conflict", "PCM同symbol趋势位冲突"),
            (
                "reject_pcm_trend_pool_anchor_first",
                "PCM趋势池锚点symbol首仓限制(trend_pool_guard)",
            ),
            (
                "reject_pcm_trend_pool_unprotected_cap",
                "PCM趋势池未保护symbol上限(trend_pool_guard)",
            ),
            (
                "reject_pcm_trend_pool_post_unlock_cap",
                "PCM趋势池解锁后symbol上限(trend_pool_guard)",
            ),
            ("reject_regime", "Regime慢变量拒单(chop/box)"),
            ("reject_regime_side", "Regime allowed_sides掩码拒单"),
            ("reject_pcm_direction_policy", "PCM宪法方向过滤(按候选intent计次)"),
            ("reject_pcm_family_conflict", "PCM同symbol家族反向冲突"),
            ("reject_pcm_daily_throttle", "PCM家族日内入场上限"),
            ("reject_pcm_struct_pass_no_intent", "结构全过但无候选intent(极少)"),
            ("reject_open_atr_nonpositive", "开仓时ATR≤0拒单"),
            ("reject_open_duplicate_archetype", "同symbol同archetype已持仓拒新开"),
            ("reject_account_risk_limit", "账户级杠杆/保证金红线拒新开"),
            ("reject_add_account_risk_limit", "账户级杠杆/保证金红线拒加仓"),
            (
                "reject_spot_capital_budget",
                "spot_accum 名义/日额度 deploy 预算用尽",
            ),
            (
                "reject_spot_tranches_full",
                "spot_accum 该 symbol 已买满 tranches_per_symbol",
            ),
            (
                "reject_spot_min_interval",
                "spot_accum 距上次 deploy 未满 min_order_interval_minutes",
            ),
            ("add_position_ok", "加仓成功次数"),
            ("add_position_rejected", "加仓拒绝次数"),
            ("float_ladder_add_ok", "浮盈阶梯成功"),
            ("reject_add_trigger", "加仓 trigger 类拒绝(合计)"),
            ("reject_add_detail_min_r", "  └ 未达 min_current_r"),
            ("reject_add_detail_bpc_breakout", "  └ bpc_breakout 与仓向不一致"),
            ("reject_add_detail_me_features", "  └ ME/其它特征 trigger"),
            ("reject_add_no_parent", "无父仓(不应常见)"),
            ("reject_add_max_times", "超 max_add_times"),
            ("reject_add_locked_profit_required", "未锁盈前禁止加仓"),
            ("reject_add_constitution", "constitution 拒"),
        ]
        for k, label in keys:
            if k in fn and int(fn[k]) > 0:
                print(f"    {label}: {fn[k]}")
        ap = self.add_position_stats
        if isinstance(ap, dict) and ap.get("enabled"):
            tries = int(ap.get("add_count", 0) or 0) + int(
                ap.get("rejected_count", 0) or 0
            )
            print(
                f"    提示: 信号加仓尝试≈{tries}；若很少，先看槽满/是否常进「无持仓加仓」分支。"
            )
            db = int(fn.get("reject_add_detail_bpc_breakout", 0) or 0)
            tr = int(fn.get("reject_add_trigger", 0) or 0)
            if tr > 0 and db >= max(1, tr // 2):
                print(
                    "    提示: bpc_breakout 拒绝占比高 → 浮盈阶梯不校验该特征，"
                    "对比 float_r_ladder_only 可区分「无信号」vs「信号方向过滤」。"
                )

    def print_path_efficiency_footer(self) -> None:
        """path_efficiency_pct 分布：放在 main() 最后打印，便于 pipeline 截尾仍可见 + 与 sidecar JSON 对齐。"""
        ap = self.add_position_stats
        if not isinstance(ap, dict):
            return
        pe = ap.get("path_efficiency_pct_at_add")
        if not isinstance(pe, dict):
            return
        print()
        print("=" * 72)
        print(
            "  📐 path_efficiency_pct @ 加仓尝试 "
            "(类 ER 历史分位 [0,1]，path_efficiency_pct_f → path_efficiency_pct)"
        )
        print("=" * 72)
        for line in format_er_pct_summary_lines(
            pe.get("signal_add_attempts") or {},
            "signal_add（PCM 再意图）",
        ):
            print(line)
        for line in format_er_pct_summary_lines(
            pe.get("float_ladder_attempts") or {},
            "float_r_ladder_only（阶梯路径）",
        ):
            print(line)
        print("=" * 72)

    def export_trades_csv(self, path: str) -> None:
        """导出交易明细 CSV（合并成交列 + 入场审计：entry_source / feat_* / layer_* 等）。"""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows, orphan_n = merge_closed_trades_with_audit_rows(
            self.trades,
            self.trade_map_audit_rows,
        )
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        msg = (
            f"\n  📤 Trades exported: {len(df)} rows → {out_path}"
            f" ({len(self.trades)} closes"
        )
        if orphan_n:
            msg += f", +{orphan_n} audit-only"
        msg += ")"
        print(msg)
