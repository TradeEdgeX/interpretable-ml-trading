"""Event backtest implementation (split across submodules; import from here)."""

from scripts.event_backtest._bootstrap import logger
from scripts.event_backtest.backtester import EventBacktester
from scripts.event_backtest.cli import main
from scripts.event_backtest.features.timeline import row_to_features
from scripts.event_backtest.reporting.json_export import (
    save_json,
    save_path_efficiency_sidecar,
)
from scripts.event_backtest.reporting.trading_map import (
    _rolling_tp_vwap,
    generate_trading_map_html,
)
from scripts.event_backtest.results import BacktestResult
from scripts.event_backtest.simulator.om_bridge import OMBridge
from scripts.event_backtest.simulator.position import (
    PositionSimulator,
    _collect_open_parent_pids,
    _filter_add_position_dict_for_open_parents,
    _load_add_position_runtime_from_resume,
    _merge_add_position_runtime_with_open_legs,
    _prune_stale_add_position_records,
    _rehydrate_add_position_runtime_from_simulator,
)
from scripts.event_backtest.spot.budget import _spot_regime_unit_multiplier
from scripts.event_backtest.spot.metrics import (
    _bucket_spot_accum_funnel_row,
    _compute_deploy_quote_pct_series,
    _compute_spot_accum_accumulation_audit,
    _compute_spot_buy_hold_benchmarks,
    _compute_spot_inventory_metrics,
    _ts_utc,
)
from scripts.event_backtest.types.trade import ClosedTrade

__all__ = [
    "BacktestResult",
    "ClosedTrade",
    "EventBacktester",
    "OMBridge",
    "PositionSimulator",
    "generate_trading_map_html",
    "logger",
    "main",
    "row_to_features",
    "save_json",
]

_save_json = save_json
_spot_regime_unit_multiplier = _spot_regime_unit_multiplier
_compute_spot_buy_hold_benchmarks = _compute_spot_buy_hold_benchmarks
_compute_spot_inventory_metrics = _compute_spot_inventory_metrics
_compute_spot_accum_accumulation_audit = _compute_spot_accum_accumulation_audit
_compute_deploy_quote_pct_series = _compute_deploy_quote_pct_series
_bucket_spot_accum_funnel_row = _bucket_spot_accum_funnel_row

if __name__ == "__main__":
    import sys

    sys.exit(main())
