"""实盘 15 分钟统计快照收集器

功能:
  - 信号漏斗计数: direction → gate → entry_filter → evidence → pcm → order
  - 按策略分层统计 (bpc / me / fer)，含 gate 拦截原因
  - 持仓状态快照
  - 系统健康指标 (CPU / 内存)
  - 写入 SQLite `stats_15min` 表
  - 自动清理 > retention_days 天数据
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.time_series_model.live.metrics_exporter import METRICS

logger = logging.getLogger(__name__)

# ── SQLite Schema ──────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS stats_15min (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    symbol      TEXT    NOT NULL DEFAULT '',
    window      TEXT    NOT NULL DEFAULT '15min',
    -- 信号漏斗 (全局汇总)
    bars_processed          INTEGER DEFAULT 0,
    direction_assigned      INTEGER DEFAULT 0,
    gate_passed             INTEGER DEFAULT 0,
    entry_filter_passed     INTEGER DEFAULT 0,
    evidence_passed         INTEGER DEFAULT 0,
    pcm_selected            INTEGER DEFAULT 0,
    orders_placed           INTEGER DEFAULT 0,
    -- 按策略分层 (JSON)
    by_strategy             TEXT    DEFAULT '{}',
    -- 持仓快照 (JSON)
    positions               TEXT    DEFAULT '{}',
    -- 系统健康 (JSON)
    system_health           TEXT    DEFAULT '{}',
    -- 当前 regime
    regime                  TEXT    DEFAULT 'NORMAL'
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_stats_15min_ts ON stats_15min(timestamp);
"""

CLEANUP_SQL = """
DELETE FROM stats_15min WHERE timestamp < ?;
"""


# ── Funnel Stage 常量 ──────────────────────────────────────────


class FunnelStage:
    DIRECTION = "direction"
    GATE = "gate"
    ENTRY_FILTER = "entry_filter"
    EVIDENCE = "evidence"
    PCM_SELECTED = "pcm_selected"
    ORDER_PLACED = "order_placed"


# ── StatsCollector ─────────────────────────────────────────────


class StatsCollector:
    """15 分钟统计快照收集器

    使用方式:
        collector = StatsCollector(db_path="data/db/live_monitor.db")

        # 每次策略决策后调用
        collector.record_strategy_eval(symbol, strategy, funnel_result)

        # PCM 选中后
        collector.record_pcm_selected(symbol, strategy)

        # 下单后
        collector.record_order_placed(symbol, strategy)

        # 每 15 分钟 flush
        snapshot = collector.flush(regime="NORMAL", symbol="BTCUSDT", positions={...})
    """

    def __init__(
        self,
        db_path: str | Path = "data/db/live_monitor.db",
        retention_days: int = 30,
        auto_cleanup: bool = False,
    ):
        self.db_path = Path(db_path)
        self.retention_days = retention_days
        self.auto_cleanup = auto_cleanup

        # 全局漏斗计数
        self._bars_processed: int = 0
        self._direction_assigned: int = 0
        self._gate_passed: int = 0
        self._entry_filter_passed: int = 0
        self._evidence_passed: int = 0
        self._pcm_selected: int = 0
        self._orders_placed: int = 0

        # 按策略分层 (value 可以是 int 或 dict，所以用 Any)
        self._by_strategy: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: defaultdict(int)
        )
        # Per-symbol mirror of ``_by_strategy`` so multi-leg (and mixed-symbol)
        # flushes can write one stats_15min row per symbol. The Signals matrix
        # reads by symbol; writing only ``symbol=ALL`` made every grid cell ``—``.
        self._by_symbol_strategy: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

        # 漏斗 digest：跨多次 flush 累加，每 digest_interval_s 打一条 INFO
        self.digest_interval_s = float(os.getenv("MLBOT_FUNNEL_DIGEST_SECONDS", "900"))
        self._digest_last_mono: float = 0.0
        self._digest_globals: Dict[str, int] = defaultdict(int)
        self._digest_by_strategy: Dict[str, Dict[str, Any]] = {}

        # 初始化 DB
        self._ensure_db()

    # ── 记录接口 ──

    def record_bar_processed(self, count: int = 1) -> None:
        """记录处理了多少根 bar (每次 15min 计算时调用)"""
        self._bars_processed += count

    @staticmethod
    def _symbol_key(symbol: str) -> str:
        return str(symbol or "").upper().strip()

    def _strategy_buckets(self, symbol: str, strategy: str) -> List[Dict[str, Any]]:
        """Global strategy bucket first; optional per-symbol mirror second."""
        from src.time_series_model.live.non_trend_funnel import lane_tag_funnel_strategy

        sid = lane_tag_funnel_strategy(strategy)
        out: List[Dict[str, Any]] = [self._by_strategy[sid]]
        sym = self._symbol_key(symbol)
        if sym and sym not in {"ALL", "*"}:
            out.append(self._by_symbol_strategy[sym][sid])
        return out

    @staticmethod
    def _bump_regime_deny_reason(
        strat_stats: Dict[str, Any], funnel: Dict[str, Any]
    ) -> None:
        reason = funnel.get("regime_reason")
        if not reason:
            return
        counts = strat_stats.setdefault("regime_deny_reasons", {})
        if not isinstance(counts, dict):
            return
        rk = str(reason)[:60]
        counts[rk] = int(counts.get(rk, 0) or 0) + 1

    def record_strategy_eval(
        self,
        symbol: str,
        strategy: str,
        funnel: Dict[str, Any],
    ) -> None:
        """记录单个策略的漏斗结果

        Args:
            symbol: 交易对
            strategy: 策略名 (bpc/me/fer)
            funnel: 漏斗各阶段结果, 如:
                {"regime": True, "prefilter": True, "direction": True,
                 "direction_value": 1, "gate": False, "gate_reasons": ["vol_too_low"]}
        """
        buckets = self._strategy_buckets(symbol, strategy)
        for i, strat_stats in enumerate(buckets):
            self._apply_strategy_eval(strat_stats, funnel, bump_global=(i == 0))

    def _apply_strategy_eval(
        self,
        strat_stats: Dict[str, Any],
        funnel: Dict[str, Any],
        *,
        bump_global: bool,
    ) -> None:
        strat_stats["evals"] += 1

        if funnel.get("multileg"):
            self._record_multileg_strategy_eval(
                strat_stats, funnel, bump_global=bump_global
            )
            return

        if "regime" in funnel:
            if funnel.get("regime"):
                strat_stats["regime_passed"] += 1
            else:
                strat_stats["regime_denied"] += 1
                self._bump_regime_deny_reason(strat_stats, funnel)

        if "prefilter" in funnel:
            if funnel.get("prefilter"):
                strat_stats["prefilter_passed"] += 1
            else:
                strat_stats["prefilter_denied"] += 1

        if funnel.get("regime_side_block"):
            strat_stats["regime_side_blocked"] += 1

        if funnel.get("direction"):
            if bump_global:
                self._direction_assigned += 1
            strat_stats["direction"] += 1

            # 方向分布
            dv = funnel.get("direction_value", 0)
            if dv == 1:
                strat_stats["long"] += 1
            elif dv == -1:
                strat_stats["short"] += 1

            if funnel.get("gate"):
                if bump_global:
                    self._gate_passed += 1
                strat_stats["gate_passed"] += 1

                if funnel.get("entry_filter"):
                    if bump_global:
                        self._entry_filter_passed += 1
                    strat_stats["entry_filter_passed"] += 1

                    if funnel.get("evidence"):
                        if bump_global:
                            self._evidence_passed += 1
                        strat_stats["signals"] += 1
            else:
                # gate 拦截: 记录拦截次数 + 原因
                if "gate" in funnel:
                    strat_stats["gate_rejected"] += 1
                    reasons = funnel.get("gate_reasons") or []
                    reason_counts = strat_stats.setdefault("gate_reject_reasons", {})
                    for r in reasons:
                        rk = str(r)[:60]
                        reason_counts[rk] = (
                            reason_counts.get(rk, 0) + 1
                            if isinstance(reason_counts.get(rk), int)
                            else 1
                        )

    def _record_multileg_strategy_eval(
        self,
        strat_stats: Dict[str, Any],
        funnel: Dict[str, Any],
        *,
        bump_global: bool = True,
    ) -> None:
        """Multi-leg: regime/prefilter/wanted_enter/outcomes — not TPC no_dir."""
        strat_stats["multileg"] = True
        eng = str(funnel.get("engine") or "")
        if eng:
            strat_stats["engine"] = eng

        if "regime" in funnel:
            if funnel.get("regime"):
                strat_stats["regime_passed"] += 1
            else:
                strat_stats["regime_denied"] += 1
                self._bump_regime_deny_reason(strat_stats, funnel)

        if "prefilter" in funnel:
            if funnel.get("prefilter"):
                strat_stats["prefilter_passed"] += 1
            else:
                strat_stats["prefilter_denied"] += 1

        if funnel.get("wanted_enter"):
            strat_stats["wanted_enter"] += 1
        if funnel.get("active_segment"):
            strat_stats["active_segment"] += 1
        if funnel.get("open_grid"):
            strat_stats["open_grid"] += 1
        if funnel.get("exit_grid"):
            strat_stats["exit_grid"] += 1
        if funnel.get("exit_close"):
            strat_stats["exit_close"] += 1

        oc = funnel.get("outcome")
        if oc:
            outcomes = strat_stats.setdefault("outcomes", {})
            key = str(oc)[:60]
            outcomes[key] = int(outcomes.get(key, 0) or 0) + 1

        if funnel.get("direction"):
            if bump_global:
                self._direction_assigned += 1
            strat_stats["direction"] += 1
            dv = funnel.get("direction_value", 0)
            if dv == 1:
                strat_stats["long"] += 1
            elif dv == -1:
                strat_stats["short"] += 1

        if "risk_gate" in funnel or "gate" in funnel:
            risk_ok = bool(funnel.get("risk_gate", funnel.get("gate")))
            if risk_ok:
                strat_stats["risk_approved"] += 1
                if bump_global:
                    self._gate_passed += 1
                strat_stats["gate_passed"] += 1
            else:
                strat_stats["risk_rejected"] += 1
                strat_stats["gate_rejected"] += 1
                reasons = funnel.get("gate_reasons") or []
                reason_counts = strat_stats.setdefault("gate_reject_reasons", {})
                for r in reasons:
                    rk = str(r)[:60]
                    reason_counts[rk] = int(reason_counts.get(rk, 0) or 0) + 1

    def record_pcm_selected(self, symbol: str, strategy: str) -> None:
        """记录 PCM 选中"""
        self._pcm_selected += 1
        for bucket in self._strategy_buckets(symbol, strategy):
            bucket["pcm_selected"] += 1

    def record_order_placed(self, symbol: str, strategy: str) -> None:
        """记录下单"""
        self._orders_placed += 1
        for bucket in self._strategy_buckets(symbol, strategy):
            bucket["orders"] += 1

    # ── Flush (每 15 分钟) ──

    def flush(
        self,
        regime: str = "NORMAL",
        positions: Optional[Dict[str, Any]] = None,
        system_health: Optional[Dict[str, Any]] = None,
        symbol: str = "",
    ) -> Dict[str, Any]:
        """将当前窗口的统计写入 SQLite 并重置计数器

        Args:
            regime: 当前 regime 状态
            positions: 持仓快照
            system_health: 系统健康指标 (外部传入可包含 tick_count 等)
            symbol: 当前币种 (空字符串表示汇总)

        Returns:
            写入的快照 dict (用于日志/调试)
        """
        now = datetime.now(timezone.utc)

        # 系统健康指标 (合并外部传入 + 自动采集)
        base_health = self._collect_system_health()
        if system_health:
            base_health.update(system_health)

        # Per-symbol rows first (Signals matrix keys on symbol).
        for sym, by_strat in sorted(self._by_symbol_strategy.items()):
            if not by_strat:
                continue
            sym_snap = self._build_flush_snapshot(
                now=now,
                symbol=sym,
                by_strategy={s: dict(d) for s, d in by_strat.items()},
                regime=regime,
                positions=positions or {},
                system_health=base_health,
                # Approximate top-level counters from this symbol's evals.
                bars_processed=sum(int(d.get("evals") or 0) for d in by_strat.values()),
                direction_assigned=sum(
                    int(d.get("direction") or 0) for d in by_strat.values()
                ),
                gate_passed=sum(
                    int(d.get("gate_passed") or 0) for d in by_strat.values()
                ),
                entry_filter_passed=sum(
                    int(d.get("entry_filter_passed") or 0) for d in by_strat.values()
                ),
                evidence_passed=sum(
                    int(d.get("signals") or 0) for d in by_strat.values()
                ),
                pcm_selected=sum(
                    int(d.get("pcm_selected") or 0) for d in by_strat.values()
                ),
                orders_placed=sum(int(d.get("orders") or 0) for d in by_strat.values()),
            )
            try:
                self._write_to_db(sym_snap)
            except Exception:
                logger.exception("stats_collector: 写入 SQLite 失败 symbol=%s", sym)

        by_strategy_serializable = {s: dict(d) for s, d in self._by_strategy.items()}
        # When per-symbol rows were emitted, keep the rollup labeled ALL so the
        # Signals page can key off real symbols without colliding with the rollup.
        if self._by_symbol_strategy:
            flush_symbol = "ALL"
        else:
            flush_symbol = self._symbol_key(symbol) or "ALL"
        snapshot = self._build_flush_snapshot(
            now=now,
            symbol=flush_symbol,
            by_strategy=by_strategy_serializable,
            regime=regime,
            positions=positions or {},
            system_health=base_health,
            bars_processed=self._bars_processed,
            direction_assigned=self._direction_assigned,
            gate_passed=self._gate_passed,
            entry_filter_passed=self._entry_filter_passed,
            evidence_passed=self._evidence_passed,
            pcm_selected=self._pcm_selected,
            orders_placed=self._orders_placed,
        )

        # 写入汇总行 (ALL / 显式 symbol 标签，兼容既有运维查询)
        try:
            self._write_to_db(snapshot)
        except Exception:
            logger.exception("stats_collector: 写入 SQLite 失败")

        # 定期清理 (仅当 auto_cleanup=True)
        if self.auto_cleanup:
            try:
                self._cleanup_old_data()
            except Exception:
                logger.exception("stats_collector: 清理旧数据失败")

        # 日志摘要
        strat_summary = " ".join(
            f"{s}:{d.get('signals', 0)}/{d.get('evals', 0)}"
            f"(gr={d.get('gate_rejected', 0)})"
            for s, d in self._by_strategy.items()
        )
        gate_rejected = sum(
            d.get("gate_rejected", 0) for d in self._by_strategy.values()
        )
        has_activity = any(
            [
                self._bars_processed,
                self._direction_assigned,
                self._gate_passed,
                gate_rejected,
                self._entry_filter_passed,
                self._evidence_passed,
                self._pcm_selected,
                self._orders_placed,
            ]
        )
        log_mode = os.getenv("MLBOT_STATS_FLUSH_LOG", "quiet").strip().lower()
        should_log = log_mode in {"verbose", "always", "1", "true", "on"}
        if not should_log and log_mode not in {"off", "0", "false", "never"}:
            should_log = has_activity
        log_fn = logger.info if should_log else logger.debug
        log_fn(
            "📊 [%s] 15min Stats: dir=%d gate=%d/%d ef=%d ev=%d pcm=%d order=%d | %s",
            flush_symbol,
            self._direction_assigned,
            self._gate_passed,
            self._gate_passed + gate_rejected,
            self._entry_filter_passed,
            self._evidence_passed,
            self._pcm_selected,
            self._orders_placed,
            strat_summary,
        )

        # 同步更新 Prometheus 指标
        try:
            METRICS.update_from_flush(
                bars=self._bars_processed,
                direction=self._direction_assigned,
                gate=self._gate_passed,
                entry_filter=self._entry_filter_passed,
                evidence=self._evidence_passed,
                pcm_selected=self._pcm_selected,
                orders=self._orders_placed,
                by_strategy=by_strategy_serializable,
                positions_count=len(positions or {}),
                symbol=flush_symbol,
                scope="trend",
            )
            METRICS.update_system_health()
        except Exception:
            pass  # Prometheus 更新失败不影响主流程

        self._accumulate_funnel_digest(snapshot)
        self._maybe_log_funnel_digest()

        # 重置计数器
        self._reset()

        return snapshot

    @staticmethod
    def _build_flush_snapshot(
        *,
        now: datetime,
        symbol: str,
        by_strategy: Dict[str, Any],
        regime: str,
        positions: Dict[str, Any],
        system_health: Dict[str, Any],
        bars_processed: int,
        direction_assigned: int,
        gate_passed: int,
        entry_filter_passed: int,
        evidence_passed: int,
        pcm_selected: int,
        orders_placed: int,
    ) -> Dict[str, Any]:
        return {
            "timestamp": now.isoformat(),
            "symbol": symbol,
            "window": "15min",
            "bars_processed": int(bars_processed),
            "direction_assigned": int(direction_assigned),
            "gate_passed": int(gate_passed),
            "entry_filter_passed": int(entry_filter_passed),
            "evidence_passed": int(evidence_passed),
            "pcm_selected": int(pcm_selected),
            "orders_placed": int(orders_placed),
            "by_strategy": by_strategy,
            "positions": positions,
            "system_health": system_health,
            "regime": regime,
        }

    # ── 内部方法 ──

    def _accumulate_funnel_digest(self, snapshot: Dict[str, Any]) -> None:
        """把本窗口快照累加到 digest buffer（多条 flush / 多条 symbol 合一）"""
        self._digest_globals["bars_processed"] += int(
            snapshot.get("bars_processed", 0) or 0
        )
        self._digest_globals["direction_assigned"] += int(
            snapshot.get("direction_assigned", 0) or 0
        )
        self._digest_globals["gate_passed"] += int(snapshot.get("gate_passed", 0) or 0)
        self._digest_globals["entry_filter_passed"] += int(
            snapshot.get("entry_filter_passed", 0) or 0
        )
        self._digest_globals["evidence_passed"] += int(
            snapshot.get("evidence_passed", 0) or 0
        )
        self._digest_globals["pcm_selected"] += int(
            snapshot.get("pcm_selected", 0) or 0
        )
        self._digest_globals["orders_placed"] += int(
            snapshot.get("orders_placed", 0) or 0
        )

        bys = snapshot.get("by_strategy") or {}
        for strat, raw in bys.items():
            if not isinstance(raw, dict):
                continue
            tgt = self._digest_by_strategy.setdefault(str(strat), {})

            cnt_keys = (
                "evals",
                "direction",
                "long",
                "short",
                "gate_passed",
                "gate_rejected",
                "entry_filter_passed",
                "signals",
                "pcm_selected",
                "orders",
                "regime_passed",
                "regime_denied",
                "prefilter_passed",
                "prefilter_denied",
                "wanted_enter",
                "active_segment",
                "open_grid",
                "exit_grid",
                "exit_close",
                "risk_approved",
                "risk_rejected",
            )
            for k in cnt_keys:
                if k not in raw:
                    continue
                v = raw.get(k)
                if isinstance(v, (int, float)):
                    tgt[k] = int(tgt.get(k, 0)) + int(v)

            greasons = raw.get("gate_reject_reasons") or {}
            rc: Counter = tgt.setdefault("_reject_reasons", Counter())
            if isinstance(greasons, dict):
                for reason, cnt in greasons.items():
                    if isinstance(cnt, (int, float)):
                        rc[str(reason)[:80]] += int(cnt)

            if raw.get("multileg"):
                tgt["multileg"] = True
            eng = raw.get("engine")
            if eng:
                tgt["engine"] = str(eng)
            outcomes = raw.get("outcomes") or {}
            if isinstance(outcomes, dict):
                oc: Counter = tgt.setdefault("_outcomes", Counter())
                for k, v in outcomes.items():
                    if isinstance(v, (int, float)):
                        oc[str(k)[:60]] += int(v)

    def _format_multileg_digest_line(self, strat: str, d: Dict[str, Any]) -> str:
        ev = int(d.get("evals", 0) or 0)
        regime = int(d.get("regime_passed", 0) or 0)
        pre = int(d.get("prefilter_passed", 0) or 0)
        wanted = int(d.get("wanted_enter", 0) or 0)
        active = int(d.get("active_segment", 0) or 0)
        ord_c = int(d.get("orders", 0) or 0)
        risk_ok = int(d.get("risk_approved", 0) or 0)
        risk_no = int(d.get("risk_rejected", 0) or 0)
        oc = d.get("_outcomes") or Counter()
        top_oc = ""
        if isinstance(oc, Counter) and oc:
            top_oc = ",".join(f"{k}:{v}" for k, v in oc.most_common(5))

        sn = str(strat).lower()
        if "chop" in sn or d.get("engine") == "chop_grid":
            exit_g = int(d.get("exit_grid", 0) or 0)
            open_g = int(d.get("open_grid", 0) or 0)
            blk = (
                f"{strat}[ev={ev} regime✓{regime} prefilter✓{pre} "
                f"wanted_enter={wanted} active={active} exit_grid={exit_g} "
                f"open_grid={open_g} risk✓{risk_ok} risk✗{risk_no} ord={ord_c}"
            )
        else:
            lng = int(d.get("long", 0) or 0)
            sht = int(d.get("short", 0) or 0)
            exit_c = int(d.get("exit_close", 0) or 0)
            blk = (
                f"{strat}[ev={ev} regime✓{regime} prefilter✓{pre} "
                f"wanted_enter={wanted} long={lng} short={sht} active={active} "
                f"exit={exit_c} risk✓{risk_ok} risk✗{risk_no} ord={ord_c}"
            )
        if top_oc:
            blk += f" audit{{{top_oc}}}"
        return blk

    def _clear_funnel_digest(self) -> None:
        self._digest_globals.clear()
        self._digest_by_strategy.clear()

    def _maybe_log_funnel_digest(self) -> None:
        if self.digest_interval_s <= 0:
            return
        now_mono = time.monotonic()
        if self._digest_last_mono <= 0:
            self._digest_last_mono = now_mono
            return
        if now_mono - self._digest_last_mono < self.digest_interval_s:
            return

        dg = dict(self._digest_globals)
        bys = sorted(self._digest_by_strategy.items(), key=lambda x: x[0])
        chunks: List[str] = []
        multileg_any = any(d.get("multileg") for _, d in bys)
        for strat, d in bys:
            ev = int(d.get("evals", 0) or 0)
            if ev == 0:
                continue
            if d.get("multileg") or (
                multileg_any and str(strat).lower() in ("chop_grid", "trend_scalp")
            ):
                chunks.append(self._format_multileg_digest_line(strat, d))
                continue

            dir_ok = int(d.get("direction", 0) or 0)
            g_ok = int(d.get("gate_passed", 0) or 0)
            g_bad = int(d.get("gate_rejected", 0) or 0)
            ef_ok = int(d.get("entry_filter_passed", 0) or 0)
            sig = int(d.get("signals", 0) or 0)
            ord_c = int(d.get("orders", 0) or 0)
            pcm_c = int(d.get("pcm_selected", 0) or 0)
            no_dir = max(0, ev - dir_ok)

            hint = ""
            if no_dir >= max(ev * 50 // 100, g_bad + ef_ok):
                hint = "hint: mostly direction (no side / no rule fit)"
            elif g_bad >= max(ev * 30 // 100, 1):
                hint = "hint: many gate vetoes (see reasons)"
            elif g_ok > ef_ok and ef_ok < g_ok:
                hint = "hint: past gate but entry_filter rarely passes"

            rc = d.get("_reject_reasons") or Counter()
            if isinstance(rc, Counter):
                top = ",".join(f"{rr}:{cc}" for rr, cc in rc.most_common(4))
            else:
                top = ""

            blk = f"{strat}[ev={ev} no_dir≈{no_dir} " f"gate✓{g_ok} gate✗{g_bad}" + (
                f" reasons({top})" if top else ""
            ) + f" ef✓{ef_ok} pcm={pcm_c} evid_sig≈{sig} ord={ord_c}" + (
                f" | {hint}" if hint else ""
            )
            chunks.append(blk)

        if chunks:
            if multileg_any:
                logger.info(
                    "📈 multileg funnel digest（过去 ~%.0fs｜regime→prefilter→wanted_enter→active→risk→order；"
                    "chop 看 wanted_enter/exit_grid/open_grid，勿用 no_dir）："
                    " bars=%s | risk✓/✗=%s/%s orders=%s\n    %s",
                    self.digest_interval_s,
                    dg.get("bars_processed", 0),
                    dg.get("gate_passed", 0),
                    sum(
                        int(d.get("gate_rejected", 0) or 0)
                        for _, d in bys
                        if d.get("multileg")
                    ),
                    dg.get("orders_placed", 0),
                    "\n    ".join(chunks),
                )
            else:
                logger.info(
                    "📈 funnel digest（过去 ~%.0fs 累计｜每层：eval→dir→gate→ef→signal→PCM→order）："
                    " bars=%s | GLOBAL dir/gate+/ef/ev/pcm/order=%s/%s/%s/%s/%s/%s\n    %s",
                    self.digest_interval_s,
                    dg.get("bars_processed", 0),
                    dg.get("direction_assigned", 0),
                    dg.get("gate_passed", 0),
                    dg.get("entry_filter_passed", 0),
                    dg.get("evidence_passed", 0),
                    dg.get("pcm_selected", 0),
                    dg.get("orders_placed", 0),
                    "\n    ".join(chunks),
                )
        else:
            logger.info(
                "📈 funnel digest（过去 ~%.0fs）：无策略 evaluate 计数（或未触发 PCM 链路）｜GLOBAL bars=%s",
                self.digest_interval_s,
                dg.get("bars_processed", 0),
            )

        self._clear_funnel_digest()
        self._digest_last_mono = now_mono

    def _reset(self) -> None:
        """重置所有计数器"""
        self._bars_processed = 0
        self._direction_assigned = 0
        self._gate_passed = 0
        self._entry_filter_passed = 0
        self._evidence_passed = 0
        self._pcm_selected = 0
        self._orders_placed = 0
        self._by_strategy = defaultdict(lambda: defaultdict(int))
        self._by_symbol_strategy = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

    def _ensure_db(self) -> None:
        """确保 SQLite 表存在"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(CREATE_TABLE_SQL)
                conn.execute(CREATE_INDEX_SQL)
                # 兼容旧数据库: 添加 symbol 列
                try:
                    conn.execute(
                        "ALTER TABLE stats_15min ADD COLUMN symbol TEXT DEFAULT ''"
                    )
                except sqlite3.OperationalError:
                    pass  # 列已存在
        except Exception:
            logger.exception("stats_collector: 初始化 SQLite 失败: %s", self.db_path)

    def _write_to_db(self, snapshot: Dict[str, Any]) -> None:
        """写入一条 15min 快照"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO stats_15min (
                    timestamp, symbol, window,
                    bars_processed, direction_assigned, gate_passed,
                    entry_filter_passed, evidence_passed,
                    pcm_selected, orders_placed,
                    by_strategy, positions, system_health, regime
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot["timestamp"],
                    snapshot.get("symbol", ""),
                    snapshot["window"],
                    snapshot["bars_processed"],
                    snapshot["direction_assigned"],
                    snapshot["gate_passed"],
                    snapshot["entry_filter_passed"],
                    snapshot["evidence_passed"],
                    snapshot["pcm_selected"],
                    snapshot["orders_placed"],
                    json.dumps(snapshot["by_strategy"]),
                    json.dumps(snapshot["positions"]),
                    json.dumps(snapshot["system_health"]),
                    snapshot["regime"],
                ),
            )

    def _cleanup_old_data(self) -> None:
        """清理超过 retention_days 的旧数据"""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        cutoff_str = cutoff.isoformat()
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(CLEANUP_SQL, (cutoff_str,))
            if cursor.rowcount > 0:
                logger.info(
                    "stats_collector: 清理了 %d 条 > %d 天的旧记录",
                    cursor.rowcount,
                    self.retention_days,
                )

    @staticmethod
    def _collect_system_health() -> Dict[str, Any]:
        """收集系统健康指标"""
        health: Dict[str, Any] = {}
        try:
            import psutil

            proc = psutil.Process(os.getpid())
            health["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            health["memory_rss_mb"] = round(proc.memory_info().rss / 1024 / 1024, 1)
            mem = psutil.virtual_memory()
            health["memory_mb"] = round(mem.used / 1024 / 1024, 1)
            health["memory_percent"] = mem.percent
        except ImportError:
            pass
        except Exception:
            pass
        return health
