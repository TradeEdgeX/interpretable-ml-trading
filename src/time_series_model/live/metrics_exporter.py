"""Prometheus 指标暴露模块

启动一个 HTTP server 在指定端口 (默认 9090)，暴露 /metrics 端点供 Prometheus 抓取。
所有指标以 mlbot_ 为前缀。

使用:
    from src.time_series_model.live.metrics_exporter import (
        start_metrics_server, METRICS,
    )
    start_metrics_server(port=9090)

    # 更新指标
    METRICS.bars_processed.inc(6)
    METRICS.funnel_stage.labels(stage="gate", strategy="me").inc()
    METRICS.positions_active.set(1)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, List, Optional, Set, Tuple

from src.order_management.execution_truth_sync import RECONCILIATION_ISSUE_BUCKETS

logger = logging.getLogger(__name__)

# ── 延迟导入: prometheus_client 可选依赖 ──────────────────────

_PROM_AVAILABLE = False
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Info,
        start_http_server,
    )

    _PROM_AVAILABLE = True
except ImportError:
    pass


# ── 指标定义 ──────────────────────────────────────────────────


class _NoopMetric:
    """prometheus_client 未安装时的空操作替身"""

    def inc(self, *a, **kw):
        pass

    def dec(self, *a, **kw):
        pass

    def set(self, *a, **kw):
        pass

    def observe(self, *a, **kw):
        pass

    def labels(self, *a, **kw):
        return self

    def info(self, *a, **kw):
        pass


_NOOP = _NoopMetric()


class Metrics:
    """全部 Prometheus 指标的集中定义"""

    def __init__(self) -> None:
        if not _PROM_AVAILABLE:
            # 所有属性指向空操作
            self.bars_processed = _NOOP
            self.funnel_stage = _NOOP
            self.signals_total = _NOOP
            self.orders_total = _NOOP
            self.positions_active = _NOOP
            self.pnl_realized_total = _NOOP
            self.drawdown = _NOOP
            self.loss = _NOOP
            self.kill_switch_halted = _NOOP
            self.last_bar_age = _NOOP
            self.oms_db_age_seconds = _NOOP
            self.external_file_age_seconds = _NOOP
            self.ws_connected = _NOOP
            self.cpu_percent = _NOOP
            self.memory_mb = _NOOP
            self.uptime_seconds = _NOOP
            self.gate_reject_rate = _NOOP
            self.gate_rejected_total = _NOOP
            self.gate_reject_reasons_total = _NOOP
            self.direction_total = _NOOP
            self.memory_rss_mb = _NOOP
            self.disk_used_percent = _NOOP
            self.disk_free_gb = _NOOP
            self.dir_size_gb = _NOOP
            self.funding_rate = _NOOP
            self.mark_price = _NOOP
            self.open_interest_usd = _NOOP
            self.account_balance = _NOOP
            self.account_margin_ratio = _NOOP
            self.unrealized_pnl_total = _NOOP
            self.system_mode = _NOOP
            self.bot_info = _NOOP
            self.regime_state = _NOOP
            self.ood_score = _NOOP
            # Retrain Monitor
            self.retrain_triggered = _NOOP
            self.retrain_trigger_count = _NOOP
            self.sharpe_live_30d = _NOOP
            self.sharpe_decay_ratio = _NOOP
            self.consecutive_losses = _NOOP
            self.days_since_last_train = _NOOP
            self.alpha_decay_max = _NOOP
            # Per-strategy slot metrics
            self.strategy_slots_active = _NOOP
            self.strategy_slots_max = _NOOP
            # PCM notional runtime metrics
            self.pcm_notional_total = _NOOP
            self.pcm_notional_by_symbol = _NOOP
            self.pcm_notional_by_family = _NOOP
            self.pcm_notional_reject_count = _NOOP
            self.pcm_notional_soft_cap = _NOOP
            self.pcm_notional_hard_cap = _NOOP
            self.bc_coin_active_mode_mismatch = _NOOP
            self.bc_coin_margin_kind = _NOOP
            self.bc_coin_pilot_mode = _NOOP
            self.bc_coin_guard_blocked_total = _NOOP
            self.exposure_consensus_state = _NOOP
            self.exposure_lock_active = _NOOP
            self.multi_leg_bars_processed = _NOOP
            self.multi_leg_actions_total = _NOOP
            self.multi_leg_risk_rejected_total = _NOOP
            self.multi_leg_execution_results_total = _NOOP
            self.multi_leg_reconciliation_issues_total = _NOOP
            self.multi_leg_user_stream_events_total = _NOOP
            self.multi_leg_daemon_polls_total = _NOOP
            self.multi_leg_risk_reject_codes_total = _NOOP
            self.multi_leg_cancel_reason_bucket_total = _NOOP
            self.multi_leg_market_exit_total = _NOOP
            self.multi_leg_market_exit_reason_bucket_total = _NOOP
            self.multi_leg_engine_bar_outcome_total = _NOOP
            self.strategy_symbol_bar_ohlc = _NOOP
            self.account_update_success = _NOOP
            self.account_update_age_seconds = _NOOP
            self.position_notional_usdt = _NOOP
            self.position_qty = _NOOP
            self.reconciliation_ok = _NOOP
            self.reconciliation_issue_count = _NOOP
            self.reconciliation_last_success_ts = _NOOP
            self.reconciliation_last_error_ts = _NOOP
            self.strategy_feature_value = _NOOP
            self.strategy_event_total = _NOOP
            self.strategy_event_price = _NOOP
            self.dashboard_catalog = _NOOP
            self._cpu_percent_primed = False
            self._last_dir_size_ts = 0.0
            return

        # ── Counters (累计值，只增不减) ──

        self.bars_processed = Counter(
            "mlbot_bars_processed_total",
            "Total bars processed across all symbols",
        )

        self.funnel_stage = Counter(
            "mlbot_funnel_total",
            "Signal funnel stage pass count",
            ["stage", "strategy"],
        )

        self.signals_total = Counter(
            "mlbot_signals_total",
            "Signals generated (evidence passed)",
            ["strategy"],
        )

        self.orders_total = Counter(
            "mlbot_orders_total",
            "Orders placed",
            ["strategy"],
        )

        self.multi_leg_bars_processed = Counter(
            "mlbot_multi_leg_bars_processed_total",
            "Multi-leg daemon processed bar ticks",
            ["strategy", "symbol"],
        )
        self.multi_leg_actions_total = Counter(
            "mlbot_multi_leg_actions_total",
            "Multi-leg engine actions submitted to orchestrator",
            ["strategy", "symbol"],
        )
        self.multi_leg_risk_rejected_total = Counter(
            "mlbot_multi_leg_risk_rejected_total",
            "Multi-leg actions rejected by risk governor",
            ["strategy", "symbol"],
        )
        self.multi_leg_execution_results_total = Counter(
            "mlbot_multi_leg_execution_results_total",
            "Multi-leg execution adapter results (orders/fills pipeline)",
            ["strategy", "symbol"],
        )
        self.multi_leg_reconciliation_issues_total = Counter(
            "mlbot_multi_leg_reconciliation_issues_total",
            "Multi-leg reconciliation failures (ok=false)",
            ["strategy"],
        )
        self.multi_leg_user_stream_events_total = Counter(
            "mlbot_multi_leg_user_stream_events_total",
            "Binance user data stream execution reports routed to multi-leg",
            ["strategy", "symbol"],
        )
        self.multi_leg_daemon_polls_total = Counter(
            "mlbot_multi_leg_daemon_polls_total",
            "Completed multi-leg daemon poll iterations (run_forever loop ticks)",
        )
        self.multi_leg_risk_reject_codes_total = Counter(
            "mlbot_multi_leg_risk_reject_codes_total",
            "Multi-leg portfolio risk vetoes by coarse reason bucket",
            ["strategy", "symbol", "code"],
        )
        self.multi_leg_cancel_reason_bucket_total = Counter(
            "mlbot_multi_leg_cancel_reason_bucket_total",
            "Multi-leg exchange cancel attempts grouped by coarse reason bucket",
            ["strategy", "symbol", "reason_bucket"],
        )
        self.multi_leg_market_exit_total = Counter(
            "mlbot_multi_leg_market_exit_total",
            "Multi-leg reduce-only market exits requested (shadow or live submit path)",
            ["strategy", "symbol"],
        )
        self.multi_leg_market_exit_reason_bucket_total = Counter(
            "mlbot_multi_leg_market_exit_reason_bucket_total",
            "market_exit intents by coarse reason label from action payload",
            ["strategy", "symbol", "reason_bucket"],
        )
        self.multi_leg_engine_bar_outcome_total = Counter(
            "mlbot_multi_leg_engine_bar_outcome_total",
            "Per-bar engine classification (why flat / open / hold / exit) for hedge engines",
            ["strategy", "symbol", "engine", "outcome"],
        )

        self.strategy_symbol_bar_ohlc = Gauge(
            "mlbot_strategy_symbol_bar_ohlc",
            "Latest strategy/symbol OHLC value for the configured feature timeframe",
            ["strategy", "symbol", "timeframe", "field"],
        )
        self.strategy_feature_value = Gauge(
            "mlbot_strategy_feature_value",
            "Selected strategy/symbol feature values for dashboard overlays",
            ["strategy", "symbol", "timeframe", "layer", "feature"],
        )
        self.strategy_event_total = Counter(
            "mlbot_strategy_event_total",
            "Strategy events for trade-map markers",
            ["scope", "strategy", "symbol", "event", "side"],
        )
        self.strategy_event_price = Gauge(
            "mlbot_strategy_event_price",
            "Last observed strategy event price (if available)",
            ["scope", "strategy", "symbol", "event", "side"],
        )
        self.dashboard_catalog = Gauge(
            "mlbot_dashboard_catalog",
            "Stable Strategy Map dropdown catalog (startup snapshot)",
            ["role", "name"],
        )

        # ── Gauges (当前值，可升可降) ──

        self.positions_active = Gauge(
            "mlbot_positions_active",
            "Currently active positions",
        )

        self.pnl_realized_total = Gauge(
            "mlbot_pnl_realized_total",
            "Cumulative realized PnL (percent of equity)",
        )

        self.drawdown = Gauge(
            "mlbot_drawdown",
            "Current drawdown (0.0 to 1.0)",
        )

        self.loss = Gauge(
            "mlbot_loss",
            "Loss by period (fraction of equity)",
            ["period"],  # daily / weekly / monthly
        )

        self.kill_switch_halted = Gauge(
            "mlbot_kill_switch_halted",
            "Kill switch state: 0=running, 1=halted",
        )

        self.last_bar_age = Gauge(
            "mlbot_last_bar_age_seconds",
            "Seconds since last processed decision bar became available "
            "(2h_close: now−(open+TF); not open-indexed wall age)",
            ["symbol"],
        )

        self.oms_db_age_seconds = Gauge(
            "mlbot_oms_db_age_seconds",
            "Seconds since trend OMS sqlite file was last modified",
        )

        self.external_file_age_seconds = Gauge(
            "mlbot_external_file_age_seconds",
            "Seconds since watched sqlite mtime (process-external; "
            "feature-bus/console — survives trend hang)",
            ["role"],
        )

        self.ws_connected = Gauge(
            "mlbot_ws_connected",
            "WebSocket connection state: 1=connected, 0=disconnected",
            ["symbol"],
        )

        self.cpu_percent = Gauge(
            "mlbot_cpu_percent",
            "CPU usage percent",
        )

        self.memory_mb = Gauge(
            "mlbot_memory_mb",
            "Memory usage in MB",
        )

        self.uptime_seconds = Gauge(
            "mlbot_uptime_seconds",
            "Process uptime in seconds",
        )

        self.gate_reject_rate = Gauge(
            "mlbot_gate_reject_rate",
            "Gate rejection rate (0.0 to 1.0)",
        )

        # ── Per-strategy gate rejection ──

        self.gate_rejected_total = Counter(
            "mlbot_gate_rejected_total",
            "Gate rejections by strategy",
            ["strategy"],
        )

        self.gate_reject_reasons_total = Counter(
            "mlbot_gate_reject_reasons_total",
            "Gate rejection reasons by strategy",
            ["strategy", "reason"],
        )

        self.direction_total = Counter(
            "mlbot_direction_total",
            "Direction assignments by strategy and side",
            ["strategy", "side"],  # side: long / short
        )

        self.memory_rss_mb = Gauge(
            "mlbot_memory_rss_mb",
            "Process RSS memory in MB",
        )

        self.disk_used_percent = Gauge(
            "mlbot_disk_used_percent",
            "Filesystem used percent for a monitored volume mount",
            ["volume"],
        )
        self.disk_free_gb = Gauge(
            "mlbot_disk_free_gb",
            "Filesystem free space in GB for a monitored volume mount",
            ["volume"],
        )
        self.dir_size_gb = Gauge(
            "mlbot_dir_size_gb",
            "On-disk directory size in GB (logs, warmup ticks/bars, feature bus, etc.)",
            ["volume"],
        )

        # ── Market Data (公开 API) ──

        self.funding_rate = Gauge(
            "mlbot_funding_rate",
            "Current funding rate per symbol",
            ["symbol"],
        )

        self.mark_price = Gauge(
            "mlbot_mark_price",
            "Current mark price per symbol",
            ["symbol"],
        )

        self.open_interest_usd = Gauge(
            "mlbot_open_interest_usd",
            "Open interest in USD per symbol",
            ["symbol"],
        )

        # ── Account Data (需要 API key) ──

        self.account_balance = Gauge(
            "mlbot_account_balance",
            "Account balance in USDT",
            ["type"],  # total / available / margin
        )

        self.account_margin_ratio = Gauge(
            "mlbot_account_margin_ratio",
            "Maintenance margin ratio (0-1, >1 = liquidation)",
        )

        self.unrealized_pnl_total = Gauge(
            "mlbot_unrealized_pnl_total",
            "Total unrealized PnL in USDT",
        )
        self.account_update_success = Gauge(
            "mlbot_account_update_success",
            "Account update state per scope: 1=ok, 0=error/missing config",
            ["scope"],
        )
        self.account_update_age_seconds = Gauge(
            "mlbot_account_update_age_seconds",
            "Seconds since last successful account update per scope (-1 if never)",
            ["scope"],
        )
        self.position_notional_usdt = Gauge(
            "mlbot_position_notional_usdt",
            "Open position notional in USDT by scope/strategy/symbol/side",
            ["scope", "strategy", "symbol", "side"],
        )
        self.position_qty = Gauge(
            "mlbot_position_qty",
            "Open position quantity by scope/strategy/symbol/side",
            ["scope", "strategy", "symbol", "side"],
        )
        self.reconciliation_ok = Gauge(
            "mlbot_reconciliation_ok",
            "Reconciliation state by scope/strategy/symbol: 1=ok 0=issue",
            ["scope", "strategy", "symbol"],
        )
        self.reconciliation_issue_count = Gauge(
            "mlbot_reconciliation_issue_count",
            "Current reconciliation issue counts by issue bucket",
            ["scope", "strategy", "symbol", "issue"],
        )
        self.reconciliation_last_success_ts = Gauge(
            "mlbot_reconciliation_last_success_timestamp_seconds",
            "Unix timestamp of last successful reconciliation",
            ["scope", "strategy", "symbol"],
        )
        self.reconciliation_last_error_ts = Gauge(
            "mlbot_reconciliation_last_error_timestamp_seconds",
            "Unix timestamp of last failed reconciliation",
            ["scope", "strategy", "symbol"],
        )

        # ── System Mode ──

        self.system_mode = Gauge(
            "mlbot_system_mode",
            "System operating mode: 0=OFFLINE, 1=DEGRADED, 2=NORMAL",
        )

        # ── Feature Health ──

        self.feature_total = Gauge(
            "mlbot_feature_total",
            "Total features computed (non-NaN) per symbol/timeframe",
            ["symbol", "timeframe"],
        )

        self.feature_expected = Gauge(
            "mlbot_feature_expected",
            "Expected features (from live_feature_set) per symbol/timeframe",
            ["symbol", "timeframe"],
        )

        self.feature_nan_count = Gauge(
            "mlbot_feature_nan_count",
            "Features in live_feature_set that are NaN or missing per symbol/timeframe",
            ["symbol", "timeframe"],
        )

        self.feature_nan_ratio = Gauge(
            "mlbot_feature_nan_ratio",
            "Fraction of expected features that are NaN/missing (0-1)",
            ["symbol", "timeframe"],
        )

        self.feature_critical_nan = Gauge(
            "mlbot_feature_critical_nan",
            "1 if critical features (atr, oi_*) are NaN, 0 otherwise",
            ["symbol", "timeframe"],
        )
        self.feature_bus_publish_skipped_total = Counter(
            "mlbot_feature_bus_publish_skipped_total",
            "Feature-bus snapshots withheld due to critical NaN or high nan_ratio",
            ["symbol", "timeframe"],
        )

        self.feature_bus_snapshot_age_seconds = Gauge(
            "mlbot_feature_bus_snapshot_age_seconds",
            "Age (seconds) of latest disk feature-bus snapshot per symbol",
            ["symbol"],
        )

        self.pipeline_data_age_seconds = Gauge(
            "mlbot_pipeline_data_age_seconds",
            "Seconds since last update for a disk data pipeline stage",
            ["pipeline", "symbol"],
        )
        self.pipeline_data_fresh = Gauge(
            "mlbot_pipeline_data_fresh",
            "1 if pipeline age is within stale threshold else 0",
            ["pipeline", "symbol"],
        )

        # Hold-regime (8h/480T) bus coverage — dual+hold needs continuous history.
        self.hold_regime_bus_rows = Gauge(
            "mlbot_hold_regime_bus_rows",
            "Row count of hold-regime feature-bus parquet",
            ["symbol", "timeframe"],
        )
        self.hold_regime_bus_span_hours = Gauge(
            "mlbot_hold_regime_bus_span_hours",
            "Hours from oldest to newest hold-regime bus timestamp",
            ["symbol", "timeframe"],
        )
        self.hold_regime_bus_max_gap_hours = Gauge(
            "mlbot_hold_regime_bus_max_gap_hours",
            "Largest gap (hours) between consecutive hold-regime bus bars",
            ["symbol", "timeframe"],
        )
        self.hold_regime_bus_latest_age_seconds = Gauge(
            "mlbot_hold_regime_bus_latest_age_seconds",
            "Age (seconds) of newest hold-regime bus row (-1 if missing)",
            ["symbol", "timeframe"],
        )
        self.hold_regime_bus_coverage_ok = Gauge(
            "mlbot_hold_regime_bus_coverage_ok",
            "1 if hold-regime bus meets min rows/span/gap/freshness else 0",
            ["symbol", "timeframe"],
        )
        self.hold_regime_inject_miss_total = Counter(
            "mlbot_hold_regime_inject_miss_total",
            "Times multileg failed to inject hold-regime column into 2h features",
            ["symbol", "reason"],
        )
        self.hold_regime_veto_total = Counter(
            "mlbot_hold_regime_veto_total",
            "Times 8h hold veto suppressed a 2h chop-only regime_exit decision",
            ["symbol", "outcome"],
        )

        self.feature_loader_errors = Counter(
            "mlbot_feature_loader_errors_total",
            "Feature loader/compute errors that were silently caught",
            ["symbol", "timeframe", "node"],
        )

        # ── P5 Non-Stationarity ──

        self.regime_state = Gauge(
            "mlbot_regime_state",
            "Regime state: 0=NORMAL, 1=HIGH_VOL, 2=HIGH_LEVERAGE",
            ["symbol", "timeframe"],
        )

        self.ood_score = Gauge(
            "mlbot_ood_score",
            "Out-of-distribution score: fraction of features outside training [q05, q95] (0-1)",
            ["symbol", "timeframe"],
        )

        # ── Retrain Monitor ──

        self.retrain_triggered = Gauge(
            "mlbot_retrain_triggered",
            "Whether retrain is triggered: 0=no, 1=yes",
            ["strategy"],
        )

        self.retrain_trigger_count = Gauge(
            "mlbot_retrain_trigger_count",
            "Number of retrain trigger conditions met (0-5)",
            ["strategy"],
        )

        self.sharpe_live_30d = Gauge(
            "mlbot_sharpe_live_30d",
            "Live rolling 30-day Sharpe ratio",
            ["strategy"],
        )

        self.sharpe_decay_ratio = Gauge(
            "mlbot_sharpe_decay_ratio",
            "Live Sharpe / baseline Sharpe ratio (< 0.5 = decay)",
            ["strategy"],
        )

        self.consecutive_losses = Gauge(
            "mlbot_consecutive_losses",
            "Current consecutive loss streak",
            ["strategy"],
        )

        self.days_since_last_train = Gauge(
            "mlbot_days_since_last_train",
            "Days elapsed since last research/training run",
            ["strategy"],
        )

        self.alpha_decay_max = Gauge(
            "mlbot_alpha_decay_max",
            "Max alpha decay from L4 gate rule / L5 evidence IC (0-1)",
            ["strategy"],
        )

        # ── Per-strategy Slot Metrics ──

        self.strategy_slots_active = Gauge(
            "mlbot_strategy_slots_active",
            "Currently active slot count per strategy",
            ["strategy"],
        )

        self.strategy_slots_max = Gauge(
            "mlbot_strategy_slots_max",
            "Max slot limit per strategy",
            ["strategy"],
        )

        # ── PCM Notional Risk Runtime ──
        self.pcm_notional_total = Gauge(
            "mlbot_pcm_notional_total_frac",
            "Current PCM estimated total notional fraction to equity",
        )
        self.pcm_notional_by_symbol = Gauge(
            "mlbot_pcm_notional_symbol_frac",
            "Current PCM estimated notional fraction by symbol",
            ["symbol"],
        )
        self.pcm_notional_by_family = Gauge(
            "mlbot_pcm_notional_family_frac",
            "Current PCM estimated notional fraction by strategy family",
            ["family"],
        )
        self.pcm_notional_reject_count = Gauge(
            "mlbot_pcm_notional_reject_count",
            "Cumulative reject count by PCM notional guard reason",
            ["reason"],
        )
        self.pcm_notional_soft_cap = Gauge(
            "mlbot_pcm_notional_soft_cap_frac",
            "Configured soft max total notional fraction",
        )
        self.pcm_notional_hard_cap = Gauge(
            "mlbot_pcm_notional_hard_cap_frac",
            "Configured hard max total notional fraction",
        )

        # ── B/C coin_m live (dapi pilot) ──

        self.bc_coin_active_mode_mismatch = Gauge(
            "mlbot_bc_coin_active_mode_mismatch",
            "1 when BC_COIN_ACTIVE_MODE env differs from process MLBOT_MARGIN_KIND",
            ["system"],
        )
        self.bc_coin_margin_kind = Gauge(
            "mlbot_bc_coin_margin_kind",
            "Process margin kind: 0=usd_m 1=coin_m",
            ["system"],
        )
        self.bc_coin_pilot_mode = Gauge(
            "mlbot_bc_coin_pilot_mode",
            "Pilot mode: 0=shadow 1=testnet 2=mainnet",
            ["system"],
        )
        self.bc_coin_guard_blocked_total = Counter(
            "mlbot_bc_coin_live_guard_blocked_total",
            "BC coin_m live guard blocks before order loop",
            ["system", "reason"],
        )
        self.exposure_consensus_state = Gauge(
            "mlbot_exposure_consensus_state",
            "Exposure flat consensus: 0=ambiguous 1=pending_flat 2=confirmed_flat 3=confirmed_exposure",
            ["system", "symbol", "side"],
        )
        self.exposure_lock_active = Gauge(
            "mlbot_exposure_lock_active",
            "1 when entry/destructive heal blocked by exposure consensus",
            ["system", "symbol", "side"],
        )

        # ── Info ──

        self.bot_info = Info(
            "mlbot",
            "Trading bot metadata",
        )
        self._account_last_success_ts: dict[str, float] = {}
        self._position_labelsets: Set[Tuple[str, str, str, str]] = set()
        self._cpu_percent_primed: bool = False
        self._last_dir_size_ts: float = 0.0

    def update_process_health(self) -> None:
        """Lightweight CPU / memory gauges only (safe before HTTP metrics server binds)."""
        if not _PROM_AVAILABLE:
            return
        try:
            import psutil

            proc = psutil.Process(os.getpid())
            # 必须使用本进程 cpu_percent：全局 psutil.cpu_percent 在容器内外常反映整台宿主，
            # 三进程 Grafana 重叠时会误判「谁吃了 CPU」。
            # psutil 约定：对同一 Process 的首次 cpu_percent(interval=None) 常为 0，需先 prime 一次。
            if not self._cpu_percent_primed:
                proc.cpu_percent(interval=None)
                self._cpu_percent_primed = True
            self.cpu_percent.set(proc.cpu_percent(interval=None))
            mem = psutil.virtual_memory()
            self.memory_mb.set(round(mem.used / 1024 / 1024, 1))
            self.memory_rss_mb.set(round(proc.memory_info().rss / 1024 / 1024, 1))
        except Exception:
            pass

    def update_system_health(self) -> None:
        """读取 psutil 更新 CPU / 内存 / uptime / 磁盘 / pipeline 新鲜度"""
        self.update_process_health()
        self.update_disk_health()
        self.update_pipeline_health_from_env()
        self.update_external_file_ages()

    def update_external_file_ages(self) -> None:
        """Stat OMS/monitor DBs on disk — independent of trend consumer liveness."""
        if not _PROM_AVAILABLE:
            return
        try:
            from src.monitoring.external_file_age import update_external_file_age_gauges

            update_external_file_age_gauges(self.external_file_age_seconds)
        except Exception as exc:
            logger.debug("external file age metrics skipped: %s", exc)

    def update_pipeline_health_from_env(self) -> None:
        """Publisher: ticks/bars/bus features/macro seed freshness (quant-feature-bus job)."""
        if not _PROM_AVAILABLE:
            return
        raw = os.getenv("MLBOT_LIVE_SYMBOLS", "")
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
        if not symbols:
            return
        try:
            from src.live_data_stream.pipeline_freshness import (
                update_pipeline_freshness_metrics,
            )

            live_base = Path(os.getenv("MLBOT_LIVE_BASE", "live/highcap"))
            storage = Path(
                os.getenv("MLBOT_LIVE_STORAGE_BASE", str(live_base / "data"))
            )
            bus = Path(os.getenv("MLBOT_FEATURE_BUS_ROOT", "live/shared_feature_bus"))
            seed_raw = os.getenv("MLBOT_WEEKLY_EMA_SEED_ROOT", "").strip()
            seed = seed_raw or str(storage / "macro" / "spot_weekly_ema200")
            tfs = [
                t.strip()
                for t in os.getenv("MLBOT_PIPELINE_FEATURE_TFS", "120T,240T").split(",")
                if t.strip()
            ]
            try:
                from src.live_data_stream.hold_regime_bus_health import (
                    hold_regime_timeframes_from_env,
                )

                for hold_tf in hold_regime_timeframes_from_env():
                    if hold_tf not in tfs:
                        tfs.append(hold_tf)
            except Exception:
                pass
            bar_only = [
                s.strip().upper()
                for s in os.getenv("MLBOT_BAR_ONLY_SYMBOLS", "").split(",")
                if s.strip()
            ]
            update_pipeline_freshness_metrics(
                symbols,
                storage_base=storage,
                bus_root=bus,
                seed_root=seed,
                feature_timeframes=tfs,
                bar_only_symbols=bar_only,
            )
            try:
                from src.live_data_stream.hold_regime_bus_health import (
                    update_hold_regime_bus_health_metrics,
                )

                update_hold_regime_bus_health_metrics(
                    [symbol for symbol in symbols if symbol not in set(bar_only)],
                    bus_root=bus,
                )
            except Exception as hold_exc:
                logger.debug("hold-regime bus health metrics skipped: %s", hold_exc)
        except Exception as exc:
            logger.debug("pipeline freshness metrics skipped: %s", exc)

    def update_disk_health(self) -> None:
        """更新根分区与各数据目录占用（供 Grafana 提醒清理日志 / warmup）。"""
        if not _PROM_AVAILABLE:
            return
        now = time.time()
        dir_interval = float(os.getenv("MLBOT_DISK_DIR_SIZE_INTERVAL_SECONDS", "300"))
        refresh_dir_sizes = (now - self._last_dir_size_ts) >= dir_interval
        if refresh_dir_sizes:
            self._last_dir_size_ts = now

        for volume, path in _disk_monitor_volumes():
            try:
                usage = shutil.disk_usage(path)
            except OSError:
                continue
            used_pct = (usage.used / max(usage.total, 1)) * 100.0
            self.disk_used_percent.labels(volume=volume).set(round(used_pct, 2))
            self.disk_free_gb.labels(volume=volume).set(
                round(usage.free / (1024**3), 2)
            )
            if refresh_dir_sizes and path.is_dir():
                size_bytes = _directory_size_bytes(path)
                if size_bytes is not None:
                    self.dir_size_gb.labels(volume=volume).set(
                        round(size_bytes / (1024**3), 3)
                    )

    # ── Market & Account Data ───────────────────────────────────

    def update_market_data(self, symbols: List[str]) -> None:
        """从 Binance 公开 REST API 获取资金费率 / 标记价格 / OI

        不需要 API Key，调用频率建议 ≤ 1次/30s。
        """
        try:
            import requests
        except ImportError:
            return

        session = self._get_http_session()
        base = "https://fapi.binance.com"
        sym_set = set(symbols)

        # ── premiumIndex (批量: 所有 symbol 一次请求) ──
        try:
            resp = session.get(f"{base}/fapi/v1/premiumIndex", timeout=10)
            if resp.ok:
                mark_prices = {}  # 临时缓存用于计算 OI USD
                for item in resp.json():
                    sym = item.get("symbol", "")
                    if sym not in sym_set:
                        continue
                    fr = float(item.get("lastFundingRate", 0))
                    mp = float(item.get("markPrice", 0))
                    self.funding_rate.labels(symbol=sym).set(fr)
                    self.mark_price.labels(symbol=sym).set(mp)
                    mark_prices[sym] = mp
        except Exception as exc:
            logger.debug("premiumIndex 获取失败: %s", exc)
            mark_prices = {}

        # ── openInterest (每个 symbol 单独请求) ──
        for sym in symbols:
            try:
                resp = session.get(
                    f"{base}/fapi/v1/openInterest",
                    params={"symbol": sym},
                    timeout=5,
                )
                if resp.ok:
                    oi_contracts = float(resp.json().get("openInterest", 0))
                    mp = mark_prices.get(sym, 0)
                    oi_usd = oi_contracts * mp if mp > 0 else oi_contracts
                    self.open_interest_usd.labels(symbol=sym).set(oi_usd)
            except Exception:
                pass

    def update_account_data(self) -> None:
        """从 Binance 私有 API 获取账户余额/保证金/未实现盈亏

        默认读取趋势账户 BINANCE_*；多腿进程设置 MLBOT_ACCOUNT_SCOPE=multi_leg
        后读取 MULTI_LEG_*。coin_m 进程（MLBOT_MARGIN_KIND=coin_m）走 dapi lane。
        如果未配置则静默跳过。
        """
        raw_scope = os.getenv("MLBOT_ACCOUNT_SCOPE", "trend").strip().lower()
        scope = self._normalize_scope(raw_scope)
        margin_kind = os.getenv("MLBOT_MARGIN_KIND", "usd_m").strip().lower()
        from order_management.live_lane_registry import (
            lane_for_metrics_account,
            read_lane_keys,
        )

        lane = lane_for_metrics_account(raw_scope, margin_kind=margin_kind)
        api_key, api_secret = read_lane_keys(lane)
        missing_msg = f"account data 跳过: lane {lane.value} credentials not configured"
        if not api_key or not api_secret:
            log_fn = logger.debug if margin_kind == "coin_m" else logger.warning
            log_fn(missing_msg)
            self._mark_account_update(scope=scope, success=False)
            return

        if margin_kind == "coin_m":
            self._update_dapi_account_data(scope=scope, lane=lane)
            return
        self._update_fapi_account_data(
            scope=scope, api_key=api_key, api_secret=api_secret
        )

    def _update_fapi_account_data(
        self,
        *,
        scope: str,
        api_key: str,
        api_secret: str,
    ) -> None:
        try:
            import hashlib
            import hmac
            import requests  # noqa: F811
        except ImportError:
            self._mark_account_update(scope=scope, success=False)
            return

        session = self._get_http_session()
        base = "https://fapi.binance.com"

        try:
            srv_resp = session.get(f"{base}/fapi/v1/time", timeout=5)
            if srv_resp.ok:
                server_ts = int(srv_resp.json().get("serverTime", 0))
            else:
                server_ts = int(time.time() * 1000)
        except Exception:
            server_ts = int(time.time() * 1000)

        query = f"timestamp={server_ts}"
        sig = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()

        try:
            resp = session.get(
                f"{base}/fapi/v2/account?{query}&signature={sig}",
                headers={"X-MBX-APIKEY": api_key},
                timeout=10,
            )
            if not resp.ok:
                logger.warning("account API %d: %s", resp.status_code, resp.text[:200])
                self._mark_account_update(scope=scope, success=False)
                return

            data = resp.json()
            self.account_balance.labels(type="total").set(
                float(data.get("totalWalletBalance", 0))
            )
            self.account_balance.labels(type="available").set(
                float(data.get("availableBalance", 0))
            )
            self.account_balance.labels(type="margin").set(
                float(data.get("totalMarginBalance", 0))
            )

            margin_bal = float(data.get("totalMarginBalance", 0))
            maint_margin = float(data.get("totalMaintMargin", 0))
            if margin_bal > 0:
                self.account_margin_ratio.set(round(maint_margin / margin_bal, 6))

            self.unrealized_pnl_total.set(float(data.get("totalUnrealizedProfit", 0)))
            self._mark_account_update(scope=scope, success=True)
            self.update_position_metrics(
                scope=scope,
                strategy="all",
                positions=data.get("positions"),
            )
        except Exception as exc:
            logger.warning("account data 获取失败: %s", exc)
            self._mark_account_update(scope=scope, success=False)

    def _update_dapi_account_data(self, *, scope: str, lane) -> None:
        from order_management.exchange.factory import build_exchange_port
        from order_management.exchange.types import MarginKind
        from order_management.live_lane_registry import get_lane_spec

        spec = get_lane_spec(lane)
        prefix = spec.dapi_prefix
        if not prefix:
            logger.warning("account data 跳过: lane %s has no dapi prefix", lane.value)
            self._mark_account_update(scope=scope, success=False)
            return

        testnet = os.getenv("MLBOT_ORDER_MANAGER_TESTNET", "").lower() in (
            "1",
            "true",
            "yes",
        )
        try:
            port = build_exchange_port(
                margin_kind=MarginKind.COIN_M,
                testnet=testnet,
                shadow=False,
                api_key_env_prefix=prefix,
            )
            if port is None:
                self._mark_account_update(scope=scope, success=False)
                return
            api = getattr(port, "underlying_api", port)
            account = api.get_account() if hasattr(api, "get_account") else {}
            risk_rows = (
                api.get_position_risk() if hasattr(api, "get_position_risk") else []
            )

            from mlbot_console.services.cms_exchange.dapi_reader import (
                _dapi_available_usdt,
                _dapi_equity_usdt,
                _dapi_margin_metrics,
            )

            mark_prices = self._mark_prices_for_dapi_account(account)
            equity_bits = _dapi_equity_usdt(
                account,
                mark_prices=mark_prices,
                open_positions=[],
            )
            available = _dapi_available_usdt(account, mark_prices=mark_prices)
            margin_bits = _dapi_margin_metrics(
                list(risk_rows or []),
                mark_prices=mark_prices,
                equity_usdt=float(equity_bits["equity_usdt"]),
            )

            wallet = float(equity_bits.get("wallet_balance_usdt") or 0.0)
            equity = float(equity_bits.get("equity_usdt") or 0.0)
            self.account_balance.labels(type="total").set(wallet)
            self.account_balance.labels(type="available").set(available)
            self.account_balance.labels(type="margin").set(equity)
            ratio = margin_bits.get("margin_ratio")
            if ratio is not None:
                self.account_margin_ratio.set(float(ratio))
            self.unrealized_pnl_total.set(
                float(equity_bits.get("unrealized_pnl_usdt") or 0.0)
            )
            self._mark_account_update(scope=scope, success=True)
            self.update_position_metrics(
                scope=scope,
                strategy="all",
                positions=risk_rows,
            )
        except Exception as exc:
            logger.warning("dapi account data 获取失败: %s", exc)
            self._mark_account_update(scope=scope, success=False)

    def _mark_prices_for_dapi_account(
        self, account: Mapping[str, Any]
    ) -> dict[str, float]:
        marks: Dict[str, float] = {"USDT": 1.0}
        assets: set[str] = set()
        for row in account.get("assets") or []:
            asset = str(row.get("asset") or "").upper()
            try:
                bal = float(row.get("walletBalance") or row.get("marginBalance") or 0.0)
            except (TypeError, ValueError):
                bal = 0.0
            if asset and bal > 0:
                assets.add(asset)
        if not assets:
            return marks
        session = self._get_http_session()
        try:
            resp = session.get(
                "https://fapi.binance.com/fapi/v1/premiumIndex", timeout=10
            )
            if not resp.ok:
                return marks
            for item in resp.json():
                sym = str(item.get("symbol") or "").upper()
                for asset in assets:
                    if sym == f"{asset}USDT":
                        marks[asset] = float(item.get("markPrice") or 0.0)
        except Exception:
            pass
        return marks

    @staticmethod
    def _get_http_session():
        """返回带代理配置的 requests Session (如果需要)"""
        import requests as _req

        session = _req.Session()
        if os.getenv("USE_SOCKS5_PROXY", "").lower() in ("1", "true", "yes"):
            host = os.getenv("SOCKS5_HOST", "127.0.0.1")
            port = os.getenv("SOCKS5_PORT", "7897")
            proxy = f"socks5h://{host}:{port}"
            session.proxies = {"http": proxy, "https": proxy}
        return session

    @staticmethod
    def _normalize_scope(scope: str) -> str:
        raw = str(scope or "").strip().lower()
        if raw in {"multi_leg", "multi-leg", "multileg", "hedge"}:
            return "hedge"
        if raw in {"spot", "spot_accum", "spot-accum"}:
            return "spot"
        # SA fade runs its own sub-account: keep its account/funnel series off `trend`.
        if raw in {"sa", "sa_fapi", "fade", "statistical_alpha"}:
            return "sa"
        return "trend"

    def _mark_account_update(self, *, scope: str, success: bool) -> None:
        now = time.time()
        scope_label = self._normalize_scope(scope)
        if success:
            self._account_last_success_ts[scope_label] = now
            self.account_update_success.labels(scope=scope_label).set(1)
            self.account_update_age_seconds.labels(scope=scope_label).set(0)
            return
        self.account_update_success.labels(scope=scope_label).set(0)
        last = self._account_last_success_ts.get(scope_label)
        age = -1 if last is None else max(0.0, now - last)
        self.account_update_age_seconds.labels(scope=scope_label).set(age)

    def update_position_metrics(
        self,
        *,
        scope: str,
        strategy: str,
        positions: Optional[Iterable[Mapping[str, Any]]],
    ) -> None:
        """Update per-position gauges from exchange/account payloads."""
        scope_label = self._normalize_scope(scope)
        strategy_label = str(strategy or "all")
        seen: Set[Tuple[str, str, str, str]] = set()
        for rec in list(positions or []):
            if not isinstance(rec, Mapping):
                continue
            symbol = str(rec.get("symbol", "") or "").upper()
            if not symbol:
                continue
            qty_raw = rec.get("positionAmt", rec.get("qty", rec.get("quantity", 0)))
            notional_raw = rec.get("notional", rec.get("notional_usdt", 0))
            try:
                qty_signed = float(qty_raw or 0.0)
            except (TypeError, ValueError):
                continue
            if abs(qty_signed) <= 1e-12:
                continue
            side = str(rec.get("side", "") or "").strip().lower()
            if side not in {"long", "short"}:
                side = "long" if qty_signed > 0 else "short"
            qty = abs(qty_signed)
            try:
                notional = abs(float(notional_raw or 0.0))
            except (TypeError, ValueError):
                notional = 0.0
            key = (scope_label, strategy_label, symbol, side)
            seen.add(key)
            self.position_qty.labels(
                scope=scope_label,
                strategy=strategy_label,
                symbol=symbol,
                side=side,
            ).set(qty)
            self.position_notional_usdt.labels(
                scope=scope_label,
                strategy=strategy_label,
                symbol=symbol,
                side=side,
            ).set(notional)

        stale = {
            key
            for key in self._position_labelsets
            if key[0] == scope_label and key[1] == strategy_label and key not in seen
        }
        for scope_l, strategy_l, symbol_l, side_l in stale:
            self.position_qty.labels(
                scope=scope_l,
                strategy=strategy_l,
                symbol=symbol_l,
                side=side_l,
            ).set(0)
            self.position_notional_usdt.labels(
                scope=scope_l,
                strategy=strategy_l,
                symbol=symbol_l,
                side=side_l,
            ).set(0)
        self._position_labelsets.difference_update(stale)
        self._position_labelsets.update(seen)

    def update_reconciliation_metrics(
        self,
        *,
        scope: str,
        strategy: str,
        symbol: str,
        ok: bool,
        issue_counts: Optional[Mapping[str, Any]] = None,
        ts_seconds: Optional[float] = None,
    ) -> None:
        """Update reconciliation health gauges with low-cardinality issue buckets."""
        scope_label = self._normalize_scope(scope)
        strategy_label = str(strategy or "all").lower()
        symbol_label = str(symbol or "all").upper()
        now_ts = float(ts_seconds or time.time())

        self.reconciliation_ok.labels(
            scope=scope_label, strategy=strategy_label, symbol=symbol_label
        ).set(1 if ok else 0)
        if ok:
            self.reconciliation_last_success_ts.labels(
                scope=scope_label, strategy=strategy_label, symbol=symbol_label
            ).set(now_ts)
        else:
            self.reconciliation_last_error_ts.labels(
                scope=scope_label, strategy=strategy_label, symbol=symbol_label
            ).set(now_ts)

        buckets = dict(issue_counts or {})
        for issue in RECONCILIATION_ISSUE_BUCKETS:
            raw = buckets.get(issue, 0)
            try:
                count = float(raw or 0)
            except (TypeError, ValueError):
                count = 0.0
            self.reconciliation_issue_count.labels(
                scope=scope_label,
                strategy=strategy_label,
                symbol=symbol_label,
                issue=issue,
            ).set(max(0.0, count))

    def record_strategy_event(
        self,
        *,
        scope: str,
        strategy: str,
        symbol: str,
        event: str,
        count: float = 1.0,
        side: str = "na",
        price: Optional[float] = None,
    ) -> None:
        """Record counter/gauge pair used by strategy-map marker panels."""
        if count <= 0:
            return
        scope_label = self._normalize_scope(scope)
        strategy_label = str(strategy or "unknown")
        symbol_label = str(symbol or "unknown").upper()
        event_label = str(event or "event").lower()
        side_label = str(side or "na").lower()
        self.strategy_event_total.labels(
            scope=scope_label,
            strategy=strategy_label,
            symbol=symbol_label,
            event=event_label,
            side=side_label,
        ).inc(float(count))
        if price is not None:
            try:
                self.strategy_event_price.labels(
                    scope=scope_label,
                    strategy=strategy_label,
                    symbol=symbol_label,
                    event=event_label,
                    side=side_label,
                ).set(float(price))
            except (TypeError, ValueError):
                return

    def publish_dashboard_catalog(
        self,
        *,
        strategies: Iterable[str],
        symbols: Iterable[str],
    ) -> None:
        """Expose low-cardinality strategy/symbol labels for Grafana template variables."""
        if not _PROM_AVAILABLE:
            return
        seen_s: Set[str] = set()
        for s in strategies:
            key = str(s or "").strip().lower()
            if not key or key in seen_s:
                continue
            seen_s.add(key)
            self.dashboard_catalog.labels(role="strategy", name=key).set(1.0)
        seen_sym: Set[str] = set()
        for sym in symbols:
            key = str(sym or "").strip().upper()
            if not key or key in seen_sym:
                continue
            seen_sym.add(key)
            self.dashboard_catalog.labels(role="symbol", name=key).set(1.0)

    def update_strategy_feature_values(
        self,
        *,
        strategy: str,
        symbol: str,
        timeframe: str,
        values: Mapping[str, Any],
        layer: str = "signal",
        feature_keys: Optional[Iterable[str]] = None,
    ) -> None:
        """Publish selected feature values to avoid unbounded metric cardinality."""
        if feature_keys is None:
            feature_keys = (
                "close",
                "atr",
                "atr14",
                "trend_confidence",
                "trend_direction",
                "semantic_chop",
                "bpc_semantic_chop",
                "box_prefilter",
                "regime_state",
                "ood_score",
            )
        strategy_label = str(strategy or "unknown")
        symbol_label = str(symbol or "unknown").upper()
        timeframe_label = str(timeframe or "unknown")
        layer_label = str(layer or "signal")
        for key in feature_keys:
            raw = values.get(key)
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                if isinstance(raw, str):
                    low = raw.strip().lower()
                    if low in {"up", "long", "true"}:
                        value = 1.0
                    elif low in {"down", "short", "false"}:
                        value = -1.0 if low in {"down", "short"} else 0.0
                    else:
                        continue
                elif isinstance(raw, bool):
                    value = 1.0 if raw else 0.0
                else:
                    continue
            self.strategy_feature_value.labels(
                strategy=strategy_label,
                symbol=symbol_label,
                timeframe=timeframe_label,
                layer=layer_label,
                feature=str(key),
            ).set(value)

    def update_from_flush(
        self,
        bars: int,
        direction: int,
        gate: int,
        entry_filter: int,
        evidence: int,
        pcm_selected: int,
        orders: int,
        by_strategy: dict,
        positions_count: int,
        symbol: str = "",
        scope: str = "trend",
    ) -> None:
        """StatsCollector.flush() 调用此方法同步更新 Prometheus 指标"""
        self.bars_processed.inc(bars)
        self.positions_active.set(positions_count)

        # Gate reject rate (基于 direction 而非 bars)
        total_dir = direction
        if total_dir > 0:
            self.gate_reject_rate.set(1.0 - gate / total_dir)

        # 按策略更新漏斗
        for strategy, stats in by_strategy.items():
            s = str(strategy)
            if stats.get("regime_passed", 0):
                self.funnel_stage.labels(stage="regime", strategy=s).inc(
                    stats["regime_passed"]
                )
            if stats.get("regime_denied", 0):
                self.funnel_stage.labels(stage="regime", strategy=s).inc(
                    stats["regime_denied"]
                )
            if stats.get("prefilter_passed", 0):
                self.funnel_stage.labels(stage="prefilter", strategy=s).inc(
                    stats["prefilter_passed"]
                )
            if stats.get("prefilter_denied", 0):
                self.funnel_stage.labels(stage="prefilter", strategy=s).inc(
                    stats["prefilter_denied"]
                )
            if stats.get("direction", 0):
                self.funnel_stage.labels(stage="direction", strategy=s).inc(
                    stats["direction"]
                )
            # 方向分布 (long/short)
            if stats.get("long", 0):
                self.direction_total.labels(strategy=s, side="long").inc(stats["long"])
            if stats.get("short", 0):
                self.direction_total.labels(strategy=s, side="short").inc(
                    stats["short"]
                )
            if stats.get("gate_passed", 0):
                self.funnel_stage.labels(stage="gate", strategy=s).inc(
                    stats["gate_passed"]
                )
            # Per-strategy gate rejection
            if stats.get("gate_rejected", 0):
                self.gate_rejected_total.labels(strategy=s).inc(stats["gate_rejected"])
            # Gate rejection reasons
            gate_reasons = stats.get("gate_reject_reasons")
            if isinstance(gate_reasons, dict):
                for reason, count in gate_reasons.items():
                    if isinstance(count, (int, float)) and count > 0:
                        self.gate_reject_reasons_total.labels(
                            strategy=s, reason=str(reason)
                        ).inc(count)
            if stats.get("entry_filter_passed", 0):
                self.funnel_stage.labels(stage="entry_filter", strategy=s).inc(
                    stats["entry_filter_passed"]
                )
            if stats.get("signals", 0):
                self.funnel_stage.labels(stage="evidence", strategy=s).inc(
                    stats["signals"]
                )
                self.signals_total.labels(strategy=s).inc(stats["signals"])
                self.record_strategy_event(
                    scope=scope,
                    strategy=s,
                    symbol=symbol,
                    event="signal",
                    count=float(stats["signals"]),
                )
            if stats.get("pcm_selected", 0):
                self.funnel_stage.labels(stage="pcm", strategy=s).inc(
                    stats["pcm_selected"]
                )
            if stats.get("orders", 0):
                self.funnel_stage.labels(stage="order", strategy=s).inc(stats["orders"])
                self.orders_total.labels(strategy=s).inc(stats["orders"])
                self.record_strategy_event(
                    scope=scope,
                    strategy=s,
                    symbol=symbol,
                    event="order",
                    count=float(stats["orders"]),
                )
            if stats.get("gate_rejected", 0):
                self.record_strategy_event(
                    scope=scope,
                    strategy=s,
                    symbol=symbol,
                    event="reject",
                    count=float(stats["gate_rejected"]),
                )

    def update_slot_metrics(
        self,
        runtime_state,
        per_strategy_limits: dict,
        global_max_slots: int = 2,
    ) -> None:
        """从 ConstitutionRuntimeState 更新 per-strategy slot Gauges

        Args:
            runtime_state: ConstitutionRuntimeState (包含 slots.active)
            per_strategy_limits: constitution.per_strategy_limits dict
            global_max_slots: 全局 slot 上限 (fallback)
        """
        if runtime_state is None:
            return
        # 统计每个 archetype 的 active slot 数
        archetype_counts: dict = {}
        try:
            for _pid, rec in (runtime_state.slots.active or {}).items():
                arch = getattr(rec, "archetype", None) or "unknown"
                arch = str(arch).lower()
                archetype_counts[arch] = archetype_counts.get(arch, 0) + 1
        except Exception:
            return

        known = {
            "bpc",
            "tpc",
            "fer",
            "me",
            "chop_grid",
            "dual_add_trend",
            "trend_scalp",
        }
        all_strategy_keys = sorted(
            known.union({str(k).lower() for k in archetype_counts.keys()})
        )
        for s in all_strategy_keys:
            self.strategy_slots_active.labels(strategy=s).set(
                archetype_counts.get(s, 0)
            )
            limits = (per_strategy_limits or {}).get(s) or {}
            max_s = int(limits.get("max_slots", global_max_slots))
            self.strategy_slots_max.labels(strategy=s).set(max_s)

    def update_strategy_symbol_ohlc(
        self,
        *,
        strategy: str,
        symbol: str,
        timeframe: str,
        values: Mapping[str, Any],
    ) -> None:
        """Publish the latest OHLC row for dashboard candlestick panels."""

        def _value_for(field: str) -> Optional[float]:
            candidates = {
                "open": ("open", "Open", "o"),
                "high": ("high", "High", "h"),
                "low": ("low", "Low", "l"),
                "close": ("close", "Close", "c", "price"),
            }[field]
            for key in candidates:
                raw = values.get(key)
                if raw is None:
                    continue
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    continue
            return None

        strategy_label = str(strategy or "unknown")
        symbol_label = str(symbol or "unknown").upper()
        timeframe_label = str(timeframe or "unknown")
        for field in ("open", "high", "low", "close"):
            val = _value_for(field)
            if val is None:
                continue
            self.strategy_symbol_bar_ohlc.labels(
                strategy=strategy_label,
                symbol=symbol_label,
                timeframe=timeframe_label,
                field=field,
            ).set(val)

    def update_pcm_notional_metrics(
        self,
        *,
        runtime: Optional[dict],
        policy: Optional[dict] = None,
    ) -> None:
        """Update PCM notional runtime gauges from LivePCM.get_stats().

        Args:
            runtime: stats["notional_runtime"]
            policy: stats["constitution"]["notional_policy"]
        """
        rt = dict(runtime or {})
        pol = dict(policy or {})
        self.pcm_notional_total.set(float(rt.get("total_notional_frac", 0.0) or 0.0))

        for sym, val in dict(rt.get("symbol_notional_frac") or {}).items():
            self.pcm_notional_by_symbol.labels(symbol=str(sym)).set(float(val or 0.0))
        for fam, val in dict(rt.get("family_notional_frac") or {}).items():
            self.pcm_notional_by_family.labels(family=str(fam)).set(float(val or 0.0))
        for reason, val in dict(rt.get("reject_counts") or {}).items():
            self.pcm_notional_reject_count.labels(reason=str(reason)).set(
                float(val or 0.0)
            )

        self.pcm_notional_soft_cap.set(
            float(pol.get("soft_max_total_notional_pct", 0.0) or 0.0)
        )
        self.pcm_notional_hard_cap.set(
            float(pol.get("hard_max_total_notional_pct", 0.0) or 0.0)
        )


def _disk_monitor_volumes() -> List[Tuple[str, Path]]:
    """Return (volume_label, path) pairs for disk / directory monitoring."""
    raw = (os.getenv("MLBOT_DISK_MONITOR_VOLUMES") or "").strip()
    if raw:
        out: List[Tuple[str, Path]] = []
        for part in raw.split(","):
            piece = part.strip()
            if not piece or ":" not in piece:
                continue
            label, path_str = piece.split(":", 1)
            label = label.strip()
            path_str = path_str.strip()
            if not label or not path_str:
                continue
            out.append((label, Path(path_str)))
        if out:
            return out

    live_base = Path(os.getenv("MLBOT_LIVE_BASE", "live/highcap"))
    storage_base = Path(os.getenv("MLBOT_LIVE_STORAGE_BASE", str(live_base / "data")))
    return [
        ("root", Path("/")),
        ("logs", live_base / "logs"),
        ("ticks", storage_base / "ticks"),
        ("bars", storage_base / "bars"),
        ("macro_seed", storage_base / "macro"),
        (
            "feature_bus",
            Path(os.getenv("MLBOT_FEATURE_BUS_ROOT", "live/shared_feature_bus")),
        ),
        ("engine_data", Path(os.getenv("MLBOT_ENGINE_DATA_ROOT", "data"))),
    ]


def _directory_size_bytes(path: Path) -> Optional[float]:
    """Best-effort directory size; prefers ``du -sb`` on Linux."""
    try:
        proc = subprocess.run(
            ["du", "-sb", str(path)],
            capture_output=True,
            text=True,
            timeout=int(os.getenv("MLBOT_DISK_DU_TIMEOUT_SECONDS", "120")),
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return float(proc.stdout.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError):
        pass

    total = 0.0
    try:
        for root, _dirs, files in os.walk(path, followlinks=False):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    continue
    except OSError:
        return None
    return total


# ── 全局单例 ──────────────────────────────────────────────────
# Tests (and a few call sites) import this module as both
# ``src.time_series_model.live.metrics_exporter`` and
# ``time_series_model.live.metrics_exporter``. Those are two module objects
# but one Prometheus REGISTRY — constructing Metrics twice raises
# ``Duplicated timeseries``. Park the instance on ``sys`` so both paths share it.

_PROCESS_METRICS_ATTR = "_mlbot_prometheus_metrics_singleton"


def _process_metrics() -> Metrics:
    existing = getattr(sys, _PROCESS_METRICS_ATTR, None)
    if existing is not None:
        return existing
    created = Metrics()
    setattr(sys, _PROCESS_METRICS_ATTR, created)
    return created


METRICS = _process_metrics()

# 启动时间 (用于计算 uptime)
_START_TIME: Optional[float] = None


def start_metrics_server(port: int = 9090) -> bool:
    """启动 Prometheus HTTP metrics server

    Returns:
        True if started, False if prometheus_client not available
    """
    global _START_TIME

    if not _PROM_AVAILABLE:
        logger.warning(
            "prometheus_client 未安装，跳过 metrics server。"
            "安装: pip install prometheus_client"
        )
        return False

    _START_TIME = time.time()

    # 设置 bot info
    METRICS.bot_info.info(
        {
            "version": "1.0",
            "strategies": "bpc,me,fer",
        }
    )
    _initialize_default_series()

    # 注册 uptime callback
    if hasattr(METRICS.uptime_seconds, "set_function"):
        METRICS.uptime_seconds.set_function(
            lambda: time.time() - _START_TIME if _START_TIME else 0
        )

    try:
        start_http_server(port)
        logger.info(
            "✅ Prometheus metrics server 启动: http://0.0.0.0:%d/metrics", port
        )
        return True
    except OSError as e:
        logger.error("❌ Prometheus metrics server 启动失败 (端口 %d): %s", port, e)
        return False


def _initialize_default_series() -> None:
    """Expose baseline series so Grafana overview panels do not start empty."""

    try:
        METRICS.kill_switch_halted.set(0)
        METRICS.positions_active.set(0)
        METRICS.pnl_realized_total.set(0)
        METRICS.drawdown.set(0)
        METRICS.gate_reject_rate.set(0)
        # system_mode：由 run_live 在启动时与 mode_manager 对齐，不设为 0 以免长期误显 OFFLINE。
        METRICS.account_margin_ratio.set(0)
        METRICS.unrealized_pnl_total.set(0)
        METRICS.pcm_notional_total.set(0)
        METRICS.pcm_notional_soft_cap.set(0)
        METRICS.pcm_notional_hard_cap.set(0)
        for scope in ("trend", "hedge", "spot"):
            METRICS.account_update_success.labels(scope=scope).set(0)
            METRICS.account_update_age_seconds.labels(scope=scope).set(-1)
            METRICS.reconciliation_ok.labels(
                scope=scope, strategy="all", symbol="ALL"
            ).set(1)
            METRICS.reconciliation_last_success_ts.labels(
                scope=scope, strategy="all", symbol="ALL"
            ).set(0)
            METRICS.reconciliation_last_error_ts.labels(
                scope=scope, strategy="all", symbol="ALL"
            ).set(0)
            for issue in (
                "missing_exchange_order",
                "orphan_exchange_order",
                "stale_local_order",
                "position_mismatch",
                "api_error",
            ):
                METRICS.reconciliation_issue_count.labels(
                    scope=scope, strategy="all", symbol="ALL", issue=issue
                ).set(0)
        for period in ("daily", "weekly", "monthly"):
            METRICS.loss.labels(period=period).set(0)
        for balance_type in ("total", "available", "margin"):
            METRICS.account_balance.labels(type=balance_type).set(0)
        from src.live_data_stream.constitution_config import (
            strategies_for_slot_metrics_from_constitution,
        )

        for strategy in strategies_for_slot_metrics_from_constitution():
            METRICS.strategy_slots_active.labels(strategy=strategy).set(0)
            METRICS.strategy_slots_max.labels(strategy=strategy).set(0)
        METRICS.update_process_health()
    except Exception:
        logger.debug("default metrics initialization skipped", exc_info=True)
