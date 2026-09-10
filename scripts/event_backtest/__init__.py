"""Event-driven multi-strategy backtest (PCM + 1m bar simulation).

Subpackages:
  - ``spot/``: spot_accum budget, deploy decay, inventory KPI, BH benchmarks
  - ``simulator/``: PositionSimulator, OM bridge, add-position rehydrate helpers
  - ``reporting/``: audit merge, JSON export, trading map HTML
  - ``features/``: bar/timeframe alignment and feature as-of helpers
  - ``modes/``: classify runs as SPOT vs TREND
"""

from scripts.event_backtest.engine import (
    BacktestResult,
    ClosedTrade,
    EventBacktester,
    PositionSimulator,
    _save_json,
    generate_trading_map_html,
    main,
)
from scripts.event_backtest.features.timeline import (
    _align_feature_index_to_bar_close,
    _feature_asof_from_sym_tf_features,
    _feature_row_asof_from_sym_tf_features,
    _get_bar_minutes,
    _get_timeframe,
    _iter_update_bars_1min,
    _iter_update_bars_primary_tf,
    _next_open_entry_price,
    _ohlc_dict_from_bar_row,
    _sync_ema_1200_from_feature_row,
    _sync_macro_tp_vwap_from_feature_row,
    _sync_weekly_ema_200_from_feature_row,
    _timeframe_from_strategy_meta,
    _timeframe_to_timedelta,
    row_to_features,
)
from scripts.event_backtest.reporting.trading_map import _rolling_tp_vwap
from scripts.event_backtest.simulator.position import (
    _collect_open_parent_pids,
    _filter_add_position_dict_for_open_parents,
    _load_add_position_runtime_from_resume,
    _merge_add_position_runtime_with_open_legs,
    _prune_stale_add_position_records,
    _rehydrate_add_position_runtime_from_simulator,
)
from scripts.event_backtest.spot.budget import spot_regime_unit_multiplier
from scripts.event_backtest.spot.metrics import (
    compute_spot_buy_hold_benchmarks,
    compute_spot_inventory_metrics,
)
from src.data_tools.data_handler import DataHandler

_spot_regime_unit_multiplier = spot_regime_unit_multiplier
_compute_spot_buy_hold_benchmarks = compute_spot_buy_hold_benchmarks
_compute_spot_inventory_metrics = compute_spot_inventory_metrics
_bucket_spot_accum_funnel_row = __import__(
    "scripts.event_backtest.spot.metrics", fromlist=["bucket_spot_accum_funnel_row"]
).bucket_spot_accum_funnel_row
_compute_spot_accum_accumulation_audit = __import__(
    "scripts.event_backtest.spot.metrics",
    fromlist=["compute_spot_accum_accumulation_audit"],
).compute_spot_accum_accumulation_audit
_compute_deploy_quote_pct_series = __import__(
    "scripts.event_backtest.spot.metrics",
    fromlist=["compute_deploy_quote_pct_series"],
).compute_deploy_quote_pct_series

__all__ = [
    "BacktestResult",
    "ClosedTrade",
    "DataHandler",
    "EventBacktester",
    "PositionSimulator",
    "_save_json",
    "generate_trading_map_html",
    "main",
    "row_to_features",
    "spot_regime_unit_multiplier",
]
