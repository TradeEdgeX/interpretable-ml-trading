"""
增量特征计算器（事件驱动）

用于实盘和回测，支持流式处理 tick 和 bar 数据，计算特征增量更新。
与批处理版本共享核心算法，但维护状态以支持增量计算。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, Deque, Set, List, Mapping
from collections import deque
from datetime import datetime, timedelta
import time
import os

# Feature extractors (baseline/talib/pywt/sklearn) are imported lazily inside
# compute paths so lean CI (requirements-ci.txt) can import OrderFlowListener /
# IncrementalFeatureComputer without the research stack.
from src.time_series_model.live.multileg_runtime_features import (
    enrich_multileg_runtime_features,
    live_feature_satisfied,
)

# Nautilus Trader types (optional — dict format is always supported)
try:
    from nautilus_trader.model import TradeTick, Bar
    from nautilus_trader.model.enums import AggressorSide

    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False
    TradeTick = None
    Bar = None
    AggressorSide = None


class IncrementalFeatureComputer:
    """
    增量特征计算器

    支持：
    1. Tick 级特征（VPIN、订单流不平衡等）
    2. Bar 级特征（技术指标、时间框架特征等）
    3. 状态维护（滚动窗口、跨月连续性等）
    """

    def __init__(
        self,
        tick_window_minutes: int = 30,
        bar_window_size: int = 1000,
        vpin_bucket_volume: Optional[float] = None,
        vpin_bucket_volume_usd: Optional[float] = None,
        vpin_n_buckets: int = 50,
        live_feature_plan_path: Optional[str] = None,
        primary_timeframe: Optional[str] = None,
        archetypes_dir: Optional[str] = None,
        weekly_ema_seed_root: Optional[str] = None,
    ):
        """
        Args:
            tick_window_minutes: Tick 数据保留窗口（分钟）
            bar_window_size: Bar 数据保留窗口（条数）
            vpin_bucket_volume: VPIN bucket volume（数量）
            vpin_bucket_volume_usd: VPIN bucket volume（USD）
            vpin_n_buckets: VPIN 滚动窗口大小
        """
        self.tick_window_minutes = tick_window_minutes
        self.bar_window_size = bar_window_size

        # Tick 缓冲区
        self.tick_buffer: Deque[Dict[str, Any]] = deque(maxlen=20000)

        # CVD per-bar accumulation (tick-driven)
        self._cvd_bar_delta: float = 0.0
        self._cvd_bar_total_flow: float = 0.0
        self._cvd_cum: float = 0.0
        self._cvd_change_hist: Deque[float] = deque(maxlen=20)
        self._cvd_total_flow_hist: Deque[float] = deque(maxlen=20)

        # Bar 缓冲区
        self.bar_buffer: Deque[Dict[str, Any]] = deque(maxlen=bar_window_size)

        # VPIN 状态
        self.vpin_bucket_volume = vpin_bucket_volume
        self.vpin_bucket_volume_usd = vpin_bucket_volume_usd
        self.vpin_n_buckets = vpin_n_buckets
        self.vpin_buckets: Deque[tuple] = deque(
            maxlen=vpin_n_buckets * 2
        )  # (timestamp, vpin_value)
        self.vpin_bucket_state = {
            "current_buy": 0.0,
            "current_sell": 0.0,
            "filled_value": 0.0,
        }

        # 当前特征值
        self.current_features: Dict[str, float] = {}

        # 时间框架特征（按时间框架分组）
        self.timeframe_features: Dict[str, Dict[str, float]] = {}

        # Live feature plan (optional)
        self.primary_timeframe = str(primary_timeframe) if primary_timeframe else None
        self.live_feature_set: Set[str] = set()
        self.live_feature_nodes: List[str] = []

        # Live bounded-tail compute: expensive per-bar-loop features (SR strength)
        # recompute the full history every cycle although live only reads the last
        # row. Computing only the last N bars is bit-identical for the last row and
        # cuts that loop ~10x. Default 300 bars (> any downstream lookback on the SR
        # columns); set MLBOT_LIVE_FEATURE_TAIL_BARS=0 to disable (full recompute).
        try:
            self._live_feature_tail: Optional[int] = int(
                os.getenv("MLBOT_LIVE_FEATURE_TAIL_BARS", "300")
            )
        except (TypeError, ValueError):
            self._live_feature_tail = 300
        if self._live_feature_tail is not None and self._live_feature_tail <= 0:
            self._live_feature_tail = None

        # Persist for live cross-symbol helpers (e.g. macro_tp_vwap anchor overlay).
        self.archetypes_dir_path: Optional[str] = (
            str(archetypes_dir) if archetypes_dir else None
        )
        self.weekly_ema_seed_root: Optional[str] = (
            str(weekly_ema_seed_root).strip()
            if weekly_ema_seed_root and str(weekly_ema_seed_root).strip()
            else os.getenv("MLBOT_WEEKLY_EMA_SEED_ROOT", "").strip() or None
        )

        # Strategy A: archetypes auto-detect (preferred — no NN dependency)
        if archetypes_dir is not None:
            try:
                from src.time_series_model.live.live_feature_plan import (
                    extract_features_from_archetypes,
                )

                feat_set, feat_nodes = extract_features_from_archetypes(
                    archetypes_dir=archetypes_dir,
                )
                self.live_feature_set = feat_set
                self.live_feature_nodes = feat_nodes
                print(
                    f"   \U0001f4cb Archetypes auto-detect: "
                    f"{len(feat_set)} columns, {len(feat_nodes)} nodes "
                    f"(from {archetypes_dir})"
                )
            except Exception as e:
                print(f"   \u26a0\ufe0f Archetypes auto-detect failed: {e}")
                self.live_feature_set = set()
                self.live_feature_nodes = []
        else:
            # Strategy B: legacy live_feature_plan.yaml (NN tier-based)
            try:
                from src.time_series_model.live.live_feature_plan import (
                    load_live_feature_plan,
                    load_live_feature_nodes,
                )

                plan_path = (
                    live_feature_plan_path
                    if live_feature_plan_path is not None
                    else os.getenv(
                        "MLBOT_LIVE_FEATURE_PLAN_YAML",
                        "config/live/live_feature_plan.yaml",
                    )
                )
                self.live_feature_set = (
                    load_live_feature_plan(plan_path=plan_path) or set()
                )
                self.live_feature_nodes = (
                    load_live_feature_nodes(plan_path=plan_path) or []
                )
            except Exception:
                self.live_feature_set = set()
                self.live_feature_nodes = []

        self._feature_loader = None
        self._feature_deps = None
        self._last_missing_log_ts: Optional[float] = None
        self._last_skipped_nodes: List[str] = []
        self._warmup_mode: bool = False  # warmup 期间跳过重型特征计算
        self._batch_features: Dict[str, Any] = {}  # 批量计算结果缓存
        if self.live_feature_nodes:
            try:
                from src.features.loader.strategy_feature_loader import (
                    StrategyFeatureLoader,
                )

                from src.time_series_model.live.feature_deps_path import (
                    resolve_feature_deps_path,
                )

                _deps_path = resolve_feature_deps_path(archetypes_dir)
                self._feature_loader = StrategyFeatureLoader(
                    feature_deps_path=_deps_path,
                    strategy_config_path=None,
                    cache_dir=None,
                    use_disk_cache=False,
                    use_memory_cache=False,
                    use_monthly_cache=False,
                    max_workers=None,
                    parallel_backend="process",
                    normalization_contract_mode="warn",
                    verbose=False,  # 实盘模式：只打印摘要+异常
                )
                self._feature_deps = self._feature_loader.feature_deps or {}

                # P5: inject baseline_path for ood_score_f from strategy config dir
                self._inject_ood_baseline_path(archetypes_dir)
            except Exception:
                self._feature_loader = None
                self._feature_deps = None

    def _want(self, key: str) -> bool:
        return (not self.live_feature_set) or (key in self.live_feature_set)

    def _inject_ood_baseline_path(self, archetypes_dir: Optional[str]) -> None:
        """Inject training_baseline.json path into ood_score_f compute_params.

        Derives strategy config dir from archetypes_dir:
          archetypes_dir = "config/strategies/tpc/archetypes"
          → strategy_dir = "config/strategies/tpc/"
          → baseline = "config/strategies/tpc/training_baseline.json"

        If file doesn't exist, ood_score safely returns 0.
        """
        if not self._feature_deps or not archetypes_dir:
            return
        feats = self._feature_deps.get("features") or {}
        ood_node = feats.get("ood_score_f")
        if not isinstance(ood_node, dict):
            return
        cp = ood_node.get("compute_params")
        if not isinstance(cp, dict):
            cp = {}
            ood_node["compute_params"] = cp
        if "baseline_path" in cp:
            return  # already set
        from pathlib import Path

        strategy_dir = Path(archetypes_dir).parent
        bl_path = strategy_dir / "training_baseline.json"
        cp["baseline_path"] = str(bl_path)

    def _compute_cvd_from_ticks(
        self,
        bars_tf: pd.DataFrame,
        ticks_1min: pd.DataFrame,
        ptf: str,
        _logger,
    ) -> bool:
        """Tick 数据补算 CVD。

        当磁盘 1min bars 没有 buy_volume/sell_volume 时，
        从 tick 数据中按 primary timeframe 重采样计算 buy/sell volume，
        然后计算 CVD。确保 fer_failure_signals_f 等节点有 CVD 输入。

        Returns:
            True if CVD columns were successfully added.
        """
        ticks = ticks_1min.copy()

        # 统一 timestamp
        if not isinstance(ticks.index, pd.DatetimeIndex):
            if "timestamp" in ticks.columns:
                ticks.index = pd.to_datetime(ticks["timestamp"], utc=True)
            else:
                _logger.debug("CVD补算: ticks 没有 timestamp 列")
                return False
        if ticks.index.tz is None:
            ticks.index = ticks.index.tz_localize("UTC")

        # 需要 side 列 (1=buy, -1=sell 或 0=buy, 1=sell)
        if "side" not in ticks.columns:
            _logger.debug("CVD补算: ticks 没有 side 列")
            return False

        # 使用 volume 或 size 列
        vol_col = "volume" if "volume" in ticks.columns else "size"
        if vol_col not in ticks.columns:
            _logger.debug("CVD补算: ticks 没有 volume/size 列")
            return False

        # 判断 side 编码: 1=buy/-1=sell 还是 0=buy/1=sell
        side = ticks["side"]
        if set(side.dropna().unique()).issubset({0, 1}):
            is_buy = side == 0  # Binance: 0=buy, 1=sell
        else:
            is_buy = side == 1  # 通用: 1=buy, -1=sell

        ticks["_buy_vol"] = ticks[vol_col].where(is_buy, 0.0)
        ticks["_sell_vol"] = ticks[vol_col].where(~is_buy, 0.0)

        # 按 primary timeframe 重采样
        agg = ticks.resample(ptf).agg({"_buy_vol": "sum", "_sell_vol": "sum"})
        agg = agg.reindex(bars_tf.index, fill_value=0.0)

        buy_vol = agg["_buy_vol"].fillna(0.0)
        sell_vol = agg["_sell_vol"].fillna(0.0)
        delta = buy_vol - sell_vol
        total_flow = buy_vol + sell_vol

        bars_tf["buy_volume"] = buy_vol
        bars_tf["sell_volume"] = sell_vol
        bars_tf["buy_qty"] = buy_vol
        bars_tf["sell_qty"] = sell_vol
        bars_tf["cvd_change_1"] = delta
        bars_tf["cvd_change_5"] = delta.rolling(window=5, min_periods=1).sum()
        bars_tf["cvd_change_20"] = delta.rolling(window=20, min_periods=1).sum()
        bars_tf["cvd_roll20"] = delta.rolling(window=20, min_periods=1).sum()
        bars_tf["cvd_roll60"] = delta.rolling(window=60, min_periods=1).sum()
        bars_tf["cvd_roll288"] = delta.rolling(window=288, min_periods=1).sum()
        bars_tf["cvd"] = delta.cumsum()
        bars_tf["cvd_normalized"] = (delta / total_flow.replace(0, np.nan)).fillna(0)
        total_flow_5 = total_flow.rolling(window=5, min_periods=1).sum()
        bars_tf["cvd_change_5_normalized"] = (
            bars_tf["cvd_change_5"] / total_flow_5.replace(0, np.nan)
        ).fillna(0)
        bars_tf["taker_buy_ratio"] = (buy_vol / total_flow.replace(0, np.nan)).fillna(
            0.5
        )

        _n_nonzero = int((delta != 0).sum())
        _logger.info(
            "  CVD 从 tick 补算: %d/%d bars 有有效 CVD",
            _n_nonzero,
            len(bars_tf),
        )
        return _n_nonzero > 0

    def _detect_tick_dependent_nodes(self, feats_cfg: dict) -> set:
        """从 feature_dependencies 动态检测需要 tick 数据的特征节点。

        通过 inspect.signature 检查 compute_func 是否接受 `ticks` 或
        `ticks_loader_json` 参数来判断，与 train_strategy_pipeline.py 中
        的检测逻辑保持一致，避免硬编码节点名。
        """
        if hasattr(self, "_tick_dependent_nodes_cache"):
            return self._tick_dependent_nodes_cache

        import inspect

        tick_nodes: set = set()
        try:
            from src.features.registry import get_compute_func
        except ImportError:
            self._tick_dependent_nodes_cache = tick_nodes
            return tick_nodes

        for node_name, node_cfg in feats_cfg.items():
            if not isinstance(node_cfg, dict):
                continue
            compute_func_name = node_cfg.get("compute_func")
            if not compute_func_name:
                continue
            try:
                cfn = get_compute_func(compute_func_name)
                if cfn is None:
                    continue
                sig = inspect.signature(cfn)
                if "ticks" in sig.parameters or "ticks_loader_json" in sig.parameters:
                    tick_nodes.add(node_name)
            except Exception:
                continue

        self._tick_dependent_nodes_cache = tick_nodes
        return tick_nodes

    @staticmethod
    def _ticks_1min_as_trades(ticks: pd.DataFrame) -> pd.DataFrame:
        """Normalize live 1min tick buffer/disk rows for reflexivity OFCI inputs."""
        if ticks is None or ticks.empty:
            return pd.DataFrame()
        t = ticks.copy()
        if not isinstance(t.index, pd.DatetimeIndex):
            if "timestamp" in t.columns:
                t.index = pd.to_datetime(t["timestamp"], utc=True)
            else:
                return pd.DataFrame()
        if t.index.tz is None:
            t.index = t.index.tz_localize("UTC")
        if "side" not in t.columns:
            return pd.DataFrame()
        return t

    @staticmethod
    def _normalize_result_index_tz(
        result: pd.Series | pd.DataFrame,
        base_index: pd.Index,
    ) -> pd.Series | pd.DataFrame:
        """Align result index timezone to ``base_index`` (offline loader policy)."""
        if not isinstance(base_index, pd.DatetimeIndex):
            return result
        if not isinstance(result, (pd.Series, pd.DataFrame)):
            return result
        if not isinstance(result.index, pd.DatetimeIndex):
            return result
        base_tz = base_index.tz
        res_tz = result.index.tz
        if base_tz is not None and res_tz is None:
            out = result.copy()
            out.index = out.index.tz_localize(base_tz)
            return out
        if base_tz is None and res_tz is not None:
            out = result.copy()
            out.index = out.index.tz_localize(None)
            return out
        return result

    @staticmethod
    def _join_loader_result_to_bars(
        bars_tf: pd.DataFrame,
        result: Any,
        info: dict,
    ) -> pd.DataFrame:
        """Merge a feature node result into resampled bars (shared by loader passes).

        Tick-derived results (OFCI etc.) are indexed by 1min aggregated trades
        with duplicate labels (buy + sell share a minute timestamp). Dedup
        ``keep="last"``, normalize tz to bar index, then ``ffill`` reindex so
        bar-close values match offline loader semantics on tz-aware live data.
        """
        output_cols = info.get("output_columns") or []
        if isinstance(result, tuple):
            if len(result) == len(output_cols):
                result = pd.DataFrame(dict(zip(output_cols, result)))
        if isinstance(result, pd.DataFrame):
            result = IncrementalFeatureComputer._normalize_result_index_tz(
                result, bars_tf.index
            )
            if result.index.has_duplicates:
                result = result[~result.index.duplicated(keep="last")]
            new_cols = [c for c in result.columns if c not in bars_tf.columns]
            if new_cols:
                aligned = result[new_cols].reindex(bars_tf.index, method="ffill")
                bars_tf = pd.concat([bars_tf, aligned], axis=1)
        elif isinstance(result, pd.Series):
            result = IncrementalFeatureComputer._normalize_result_index_tz(
                result, bars_tf.index
            )
            if result.index.has_duplicates:
                result = result[~result.index.duplicated(keep="last")]
            name = result.name or (output_cols[0] if output_cols else None)
            if name and name not in bars_tf.columns:
                bars_tf[name] = result.reindex(bars_tf.index, method="ffill")
        return bars_tf

    def _compute_ofci_pct_from_trades(
        self,
        bars_tf: pd.DataFrame,
        trades: pd.DataFrame,
        info: dict,
    ) -> pd.DataFrame:
        """Compute ``ofci_pct`` on the trade index (same percentile as offline).

        Offline ``data/parquet_data`` ticks are 1min-aggregated (clean minute
        timestamps); live recent ticks come from the WS stream with millisecond
        timestamps that do not land on 2h bar boundaries. Percentile runs on the
        trade series; bar alignment (tz-normalize + ffill) happens in
        ``_join_loader_result_to_bars`` so the current bar takes its bar-open
        value instead of NaN.
        """
        from src.features.time_series.reflexivity_features import (
            compute_ofci_from_trades,
            compute_ofci_pct_from_series,
        )
        from src.features.live_tail_context import get_live_tick_tail_rows

        params = info.get("compute_params") or {}
        ofci_window = int(params.get("ofci_window", 100))
        percentile_window = int(params.get("percentile_window", 540))
        shift = int(params.get("shift", 1))

        tick_tail = get_live_tick_tail_rows(
            ofci_window=ofci_window,
            percentile_window=percentile_window,
        )
        trades_work = trades
        if tick_tail is not None and len(trades) > tick_tail:
            trades_work = trades.iloc[-tick_tail:]

        ofci_tick = compute_ofci_from_trades(
            trades=trades_work,
            window=ofci_window,
            df=bars_tf,
        )
        if ofci_tick.empty or "ofci" not in ofci_tick.columns:
            return pd.DataFrame(index=bars_tf.index, columns=["ofci_pct"], dtype=float)
        return compute_ofci_pct_from_series(
            ofci=ofci_tick["ofci"],
            percentile_window=percentile_window,
            shift=shift,
        )

    def _compute_tick_dependent_loader_nodes(
        self,
        bars_tf: pd.DataFrame,
        ticks_1min: pd.DataFrame,
        *,
        skipped: List[str],
        tick_dependent_nodes: set,
        feats_cfg: dict,
        primary_timeframe: str,
        logger: Any,
    ) -> pd.DataFrame:
        """Run tick-dependent loader nodes skipped by step 3 when ticks are in memory."""
        import inspect

        from src.features.loader.feature_computer import _build_call_args
        from src.features.registry import get_compute_func

        trades = self._ticks_1min_as_trades(ticks_1min)
        if trades.empty:
            return bars_tf

        tick_nodes = [n for n in skipped if n in tick_dependent_nodes]
        if not tick_nodes:
            return bars_tf

        ticks = ticks_1min.copy()
        if (
            not isinstance(ticks.index, pd.DatetimeIndex)
            and "timestamp" in ticks.columns
        ):
            ticks.index = pd.to_datetime(ticks["timestamp"], utc=True)

        for n in tick_nodes:
            info = feats_cfg.get(n)
            if not isinstance(info, dict):
                continue
            output_cols = info.get("output_columns") or []
            if output_cols and all(c in bars_tf.columns for c in output_cols):
                continue
            func_name = str(info.get("compute_func") or n)
            try:
                if func_name == "compute_ofci_pct_from_series":
                    result = self._compute_ofci_pct_from_trades(bars_tf, trades, info)
                else:
                    cfn = get_compute_func(func_name)
                    if cfn is None:
                        continue
                    call_args, call_kwargs = _build_call_args(info, bars_tf, n)
                    sig = inspect.signature(cfn)
                    if "ticks" in sig.parameters:
                        call_kwargs["ticks"] = ticks
                    elif "trades" in sig.parameters:
                        call_kwargs["trades"] = trades
                        call_kwargs["df"] = bars_tf
                    else:
                        continue
                    if call_kwargs:
                        allowed = set(sig.parameters.keys())
                        accepts_var_kw = any(
                            p.kind == inspect.Parameter.VAR_KEYWORD
                            for p in sig.parameters.values()
                        )
                        if not accepts_var_kw:
                            call_kwargs = {
                                k: v for k, v in call_kwargs.items() if k in allowed
                            }
                    result = cfn(*call_args, **call_kwargs)

                bars_tf = self._join_loader_result_to_bars(bars_tf, result, info)
                logger.info("  Tick pass: %s → OK", n)
            except Exception as exc:
                logger.warning("  Tick pass failed for %s: %s", n, exc)
                self._record_loader_error(n, primary_timeframe, exc)
        return bars_tf

    def on_tick(self, tick: Any) -> None:
        """
        处理 tick 数据

        仅维护 CVD 累加器（用于 1min bar 的 cvd_change_1/cvd 列）。
        VPIN / 订单流特征已改为磁盘批量计算（compute_features_batch），
        不再在 tick 级增量更新。

        Args:
            tick: dict 或 TradeTick 对象
        """
        # 转换为统一格式
        if isinstance(tick, dict):
            tick_data = tick
        elif (
            NAUTILUS_AVAILABLE and TradeTick is not None and isinstance(tick, TradeTick)
        ):
            tick_data = {
                "ts": tick.ts_event,
                "price": float(tick.price),
                "volume": float(tick.size),
                "side": 1 if tick.aggressor_side == AggressorSide.BUYER else -1,
            }
        else:
            return

        # 添加到缓冲区（保留用于 get_orderflow_features 实时查询）
        self.tick_buffer.append(tick_data)

        # CVD accumulation for the current bar
        side = tick_data.get("side")
        volume = tick_data.get("volume")
        if side in (1, -1) and volume is not None:
            vol = float(volume)
            self._cvd_bar_delta += vol if side == 1 else -vol
            self._cvd_bar_total_flow += abs(vol)
            self._cvd_cum += vol if side == 1 else -vol

        # NOTE: _update_vpin / _update_orderflow_features 已移除
        # VPIN 和订单流特征现在通过 compute_features_batch() 从磁盘批量计算

    def on_bar(self, bar: Any, timeframe: str = "1H") -> None:
        """
        处理 bar 数据

        Args:
            bar: dict 或 Bar 对象
            timeframe: 时间框架（如 "15T", "1H")
        """
        # 转换为统一格式
        if isinstance(bar, dict):
            bar_data = bar
        elif NAUTILUS_AVAILABLE and Bar is not None and isinstance(bar, Bar):
            bar_data = {
                "ts": bar.ts_event,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
        else:
            return

        # Attach tick-driven orderflow to bar (if available)
        bar_data["cvd_change_1"] = float(self._cvd_bar_delta)
        bar_data["cvd"] = float(self._cvd_cum)
        self._cvd_change_hist.append(float(self._cvd_bar_delta))
        self._cvd_total_flow_hist.append(float(self._cvd_bar_total_flow))
        if len(self._cvd_change_hist) > 0:
            cvd_change_5 = float(np.sum(list(self._cvd_change_hist)[-5:]))
        else:
            cvd_change_5 = 0.0
        if len(self._cvd_total_flow_hist) > 0:
            total_flow_5 = float(np.sum(list(self._cvd_total_flow_hist)[-5:]))
        else:
            total_flow_5 = 0.0
        bar_data["cvd_change_5"] = cvd_change_5
        if total_flow_5 > 0:
            bar_data["cvd_change_5_normalized"] = cvd_change_5 / total_flow_5
        else:
            bar_data["cvd_change_5_normalized"] = 0.0
        # cvd_change_20: sum of last 20 bars' delta (maxlen=20)
        if len(self._cvd_change_hist) > 0:
            bar_data["cvd_change_20"] = float(np.sum(list(self._cvd_change_hist)))
        else:
            bar_data["cvd_change_20"] = 0.0
        # cvd_normalized: per-bar delta / total_flow
        if self._cvd_bar_total_flow > 0:
            bar_data["cvd_normalized"] = float(self._cvd_bar_delta) / float(
                self._cvd_bar_total_flow
            )
        else:
            bar_data["cvd_normalized"] = 0.0
        # reset accumulators for next bar
        self._cvd_bar_delta = 0.0
        self._cvd_bar_total_flow = 0.0

        # 添加到缓冲区（用于 get_recent_bars() — BPC 执行规则需要）
        self.bar_buffer.append(bar_data)

        # NOTE: 特征计算已改为磁盘批量模式 (compute_features_batch)
        # on_bar 只维护 bar_buffer + CVD，不再触发 _update_timeframe_features

    def _update_vpin(self, tick_data: Dict[str, Any]) -> None:
        """更新 VPIN（增量计算）"""
        if self.vpin_bucket_volume is None and self.vpin_bucket_volume_usd is None:
            return

        price = tick_data["price"]
        volume = tick_data["volume"]
        side = tick_data["side"]

        # 确定 bucket volume
        if self.vpin_bucket_volume_usd is not None:
            tick_value = price * volume
            target_bucket = self.vpin_bucket_volume_usd
        else:
            tick_value = volume
            target_bucket = self.vpin_bucket_volume

        # 更新 bucket 状态
        remaining = tick_value
        while remaining > 0:
            space_left = target_bucket - self.vpin_bucket_state["filled_value"]
            trade_value = min(remaining, space_left)

            if side == 1:
                self.vpin_bucket_state["current_buy"] += trade_value
            else:
                self.vpin_bucket_state["current_sell"] += trade_value

            self.vpin_bucket_state["filled_value"] += trade_value
            remaining -= trade_value

            # 检查 bucket 是否填满
            BUCKET_COMPLETION_TOLERANCE = 1e-6
            if (
                self.vpin_bucket_state["filled_value"]
                >= target_bucket - BUCKET_COMPLETION_TOLERANCE
            ):
                # 计算 VPIN
                imbalance = abs(
                    self.vpin_bucket_state["current_buy"]
                    - self.vpin_bucket_state["current_sell"]
                )
                vpin_value = imbalance / target_bucket

                # 添加到 buckets
                self.vpin_buckets.append((tick_data["ts"], vpin_value))

                # 重置 bucket 状态
                self.vpin_bucket_state = {
                    "current_buy": 0.0,
                    "current_sell": 0.0,
                    "filled_value": 0.0,
                }

        # 更新当前 VPIN（滚动平均）
        if len(self.vpin_buckets) > 0:
            recent_buckets = list(self.vpin_buckets)[-self.vpin_n_buckets :]
            if recent_buckets:
                vpin_values = [v for _, v in recent_buckets]
                self.current_features["vpin"] = float(np.mean(vpin_values))
            else:
                self.current_features["vpin"] = 0.0
        else:
            self.current_features["vpin"] = 0.0

    def _update_orderflow_features(self) -> None:
        """更新订单流特征（基于最近 N 分钟的 tick）"""
        if not self.tick_buffer:
            return

        # 获取最近 N 分钟的 tick
        cutoff_ns = (
            self.tick_buffer[-1]["ts"] - self.tick_window_minutes * 60 * 1_000_000_000
        )
        recent_ticks = [t for t in self.tick_buffer if t["ts"] >= cutoff_ns]

        if not recent_ticks:
            return

        # 计算买卖量
        buy_vol = sum(t["volume"] for t in recent_ticks if t["side"] == 1)
        sell_vol = sum(t["volume"] for t in recent_ticks if t["side"] == -1)
        total_vol = buy_vol + sell_vol

        # 计算不平衡度
        if total_vol > 0:
            imbalance = (buy_vol - sell_vol) / total_vol
            self.current_features["orderflow_imbalance"] = float(imbalance)
            self.current_features["orderflow_total_vol"] = float(total_vol)
        else:
            self.current_features["orderflow_imbalance"] = 0.0
            self.current_features["orderflow_total_vol"] = 0.0

    def _update_timeframe_features(
        self, bar_data: Dict[str, Any], timeframe: str
    ) -> None:
        """更新时间框架特征（技术指标等）"""
        if timeframe not in self.timeframe_features:
            self.timeframe_features[timeframe] = {}

        # 转换为 DataFrame 用于计算
        bars_df = pd.DataFrame(list(self.bar_buffer))
        if len(bars_df) < 2:
            return
        bars_df_indexed = bars_df.copy()
        bars_df_indexed.index = pd.to_datetime(
            bars_df_indexed["ts"], unit="ns", utc=True
        )

        # 计算简单技术指标（示例）
        closes = bars_df["close"].values

        # RSI（简化版，需要至少 14 根 bar）
        if len(closes) >= 14:
            delta = np.diff(closes)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)

            avg_gain = np.mean(gain[-14:])
            avg_loss = np.mean(loss[-14:])

            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                if self._want("rsi"):
                    self.timeframe_features[timeframe]["rsi"] = float(rsi)

        # ATR（简化版）
        if len(bars_df) >= 14:
            high = bars_df["high"].values[-14:]
            low = bars_df["low"].values[-14:]
            close = bars_df["close"].values[-14:]
            close_prev = (
                bars_df["close"].values[-15:-1] if len(bars_df) > 14 else close[:-1]
            )

            # 确保数组长度一致
            min_len = min(len(high), len(low), len(close), len(close_prev))
            if min_len > 0:
                high = high[-min_len:]
                low = low[-min_len:]
                close = close[-min_len:]
                close_prev = (
                    close_prev[-min_len:] if len(close_prev) >= min_len else close[:-1]
                )

                tr1 = high - low
                tr2 = np.abs(high - close_prev)
                tr3 = np.abs(low - close_prev)

                tr = np.maximum(tr1, np.maximum(tr2, tr3))
                atr = np.mean(tr)
                if self._want("atr"):
                    self.timeframe_features[timeframe]["atr"] = float(atr)

        # 价格位置
        if self._want("open"):
            self.timeframe_features[timeframe]["open"] = float(bar_data["open"])
        if self._want("high"):
            self.timeframe_features[timeframe]["high"] = float(bar_data["high"])
        if self._want("low"):
            self.timeframe_features[timeframe]["low"] = float(bar_data["low"])
        if self._want("close"):
            self.timeframe_features[timeframe]["close"] = float(bar_data["close"])
        if self._want("volume"):
            self.timeframe_features[timeframe]["volume"] = float(bar_data["volume"])

        # Wick ratios and price range symmetry
        try:
            from src.features.time_series.baseline_features import (
                compute_price_range_symmetry_from_series,
                compute_wick_ratios_from_series,
            )

            if self._want("price_range_symmetry"):
                sym = compute_price_range_symmetry_from_series(
                    high=bars_df["high"],
                    low=bars_df["low"],
                    close=bars_df["close"],
                )
                self.timeframe_features[timeframe]["price_range_symmetry"] = float(
                    sym.iloc[-1]
                )
                bars_df_indexed["price_range_symmetry"] = sym
            if self._want("wick_upper_ratio") or self._want("wick_lower_ratio"):
                wick_df = compute_wick_ratios_from_series(
                    open=bars_df["open"],
                    high=bars_df["high"],
                    low=bars_df["low"],
                    close=bars_df["close"],
                )
                if self._want("wick_upper_ratio"):
                    self.timeframe_features[timeframe]["wick_upper_ratio"] = float(
                        wick_df["wick_upper_ratio"].iloc[-1]
                    )
                if self._want("wick_lower_ratio"):
                    self.timeframe_features[timeframe]["wick_lower_ratio"] = float(
                        wick_df["wick_lower_ratio"].iloc[-1]
                    )
                bars_df_indexed["wick_upper_ratio"] = wick_df["wick_upper_ratio"]
                bars_df_indexed["wick_lower_ratio"] = wick_df["wick_lower_ratio"]
        except Exception:
            pass

        # Tick-driven orderflow and footprint features (bar-aligned)
        need_orderflow = any(
            k.startswith("vpin") or k.startswith("trade_cluster")
            for k in (self.live_feature_set or [])
        ) or any(
            "vpin" in str(n) or "trade_cluster" in str(n)
            for n in (self.live_feature_nodes or [])
        )
        need_trade_cluster = any(
            k.startswith("trade_cluster") for k in (self.live_feature_set or [])
        ) or any("trade_cluster" in str(n) for n in (self.live_feature_nodes or []))
        need_vpin_derived = any(
            (
                k.startswith("vpin_")
                and k
                not in {
                    "vpin",
                    "vpin_signed_imbalance",
                    "vpin_last",
                    "vpin_max",
                    "vpin_min",
                    "vpin_std",
                    "vpin_count",
                    "vpin_skewness",
                    "vpin_trend",
                    "vpin_signed_imbalance_last",
                    "vpin_signed_imbalance_max",
                }
            )
            for k in (self.live_feature_set or [])
        ) or any(
            n
            in {
                "vpin_derived_features_f",
                "vpin_block_features_f",
                "vpin_features_f",
                "vpin_ma_max_features_f",
                "vpin_change_features_f",
                "vpin_zscore_features_f",
                "vpin_quantile_features_f",
                "vpin_volatility_features_f",
                "vpin_spike_features_f",
                "vpin_momentum_features_f",
                "vpin_signed_zscore_features_f",
                "vpin_scene_semantic_scores_f",
            }
            for n in (self.live_feature_nodes or [])
        )
        need_footprint = any(
            k.startswith("fp_") for k in (self.live_feature_set or [])
        ) or any("footprint" in str(n) for n in (self.live_feature_nodes or []))
        if (need_orderflow or need_footprint) and self.tick_buffer:
            try:
                from src.features.time_series.utils_order_flow_features import (
                    extract_order_flow_features,
                    compute_trade_cluster_derived_features_from_base,
                    compute_vpin_derived_features_from_base,
                )
                from src.features.loader.feature_wrappers import (
                    compute_footprint_features,
                )

                ticks_df = pd.DataFrame(list(self.tick_buffer))
                if not ticks_df.empty and "ts" in ticks_df.columns:
                    ticks_df.index = pd.to_datetime(ticks_df["ts"], unit="ns", utc=True)
                if need_orderflow and not ticks_df.empty:
                    of_df = extract_order_flow_features(
                        bars_df_indexed,
                        ticks=ticks_df,
                        freq=pd.infer_freq(bars_df_indexed.index),
                        include_trade_clustering=need_trade_cluster,
                        compute_vpin_derived=need_vpin_derived,
                        vpin_bucket_volume=self.vpin_bucket_volume,
                        vpin_bucket_volume_usd=self.vpin_bucket_volume_usd,
                        vpin_n_buckets=self.vpin_n_buckets,
                    )
                    bars_df_indexed = bars_df_indexed.join(of_df, how="left")
                    if need_vpin_derived:
                        try:
                            vpin_derived = compute_vpin_derived_features_from_base(
                                bars_df_indexed
                            )
                            for c in vpin_derived.columns:
                                if c not in bars_df_indexed.columns:
                                    bars_df_indexed[c] = vpin_derived[c]
                        except Exception:
                            pass
                    if need_trade_cluster:
                        try:
                            tc_derived = (
                                compute_trade_cluster_derived_features_from_base(
                                    bars_df_indexed
                                )
                            )
                            for c in tc_derived.columns:
                                if c not in bars_df_indexed.columns:
                                    bars_df_indexed[c] = tc_derived[c]
                        except Exception:
                            pass
                    last = of_df.iloc[-1].to_dict()
                    for k, v in last.items():
                        if self._want(str(k)):
                            try:
                                self.timeframe_features[timeframe][str(k)] = float(v)
                            except Exception:
                                continue
                if need_footprint and not ticks_df.empty:
                    fp_df = compute_footprint_features(
                        bars_df_indexed.tail(200),
                        ticks=ticks_df,
                        persist_monthly=False,
                    )
                    bars_df_indexed = bars_df_indexed.join(fp_df, how="left")
                    last = fp_df.iloc[-1].to_dict()
                    for k, v in last.items():
                        if self._want(str(k)):
                            try:
                                self.timeframe_features[timeframe][str(k)] = float(v)
                            except Exception:
                                continue
                    # Ensure footprint columns are available for downstream nodes
                    for c in fp_df.columns:
                        if c not in bars_df_indexed.columns:
                            bars_df_indexed[c] = fp_df[c]
            except Exception:
                pass

        # BB width ratio (needed by compression_score_f)
        try:
            from src.features.time_series.utils_interaction_features import (
                compute_bb_width_ratio_from_price_from_series,
                compute_compression_score_from_series,
            )

            if "bb_width_ratio" not in bars_df_indexed.columns:
                bb_ratio = compute_bb_width_ratio_from_price_from_series(
                    close=bars_df_indexed["close"]
                )
                bars_df_indexed["bb_width_ratio"] = bb_ratio
            if "compression_score" not in bars_df_indexed.columns:
                comp = compute_compression_score_from_series(
                    bb_width_ratio=bars_df_indexed["bb_width_ratio"]
                )
                bars_df_indexed["compression_score"] = comp
                if self._want("compression_score"):
                    self.timeframe_features[timeframe]["compression_score"] = float(
                        comp.iloc[-1]
                    )
        except Exception:
            pass

        # Range ratio (5-bar)
        try:
            from src.features.time_series.baseline_features import (
                compute_range_ratio_5bar_from_series,
            )

            if "range_ratio_5bar" not in bars_df_indexed.columns:
                rr = compute_range_ratio_5bar_from_series(
                    high=bars_df_indexed["high"], low=bars_df_indexed["low"]
                )
                bars_df_indexed["range_ratio_5bar"] = rr
                if self._want("range_ratio_5bar"):
                    self.timeframe_features[timeframe]["range_ratio_5bar"] = float(
                        rr.iloc[-1]
                    )
        except Exception:
            pass

        # Volume profile volatility features (vp_* entropy/skewness etc.)
        try:
            if any(k.startswith("vp_") for k in (self.live_feature_set or [])):
                from src.features.time_series.utils_volatility_features import (
                    extract_volume_profile_volatility_features_from_series,
                )

                vp_df = extract_volume_profile_volatility_features_from_series(
                    close=bars_df_indexed["close"],
                    volume=bars_df_indexed["volume"],
                )
                bars_df_indexed = bars_df_indexed.join(vp_df, how="left")
                last = vp_df.iloc[-1].to_dict()
                for k, v in last.items():
                    if self._want(str(k)):
                        try:
                            self.timeframe_features[timeframe][str(k)] = float(v)
                        except Exception:
                            continue
        except Exception:
            pass

        # CVD change columns + normalized, if underlying flow columns exist
        _need_cvd = (
            self._want("cvd_change_5")
            or self._want("cvd_change_5_normalized")
            or self._want("cvd_change_1")
            or self._want("cvd_change_20")
            or self._want("cvd_normalized")
        )
        if _need_cvd:
            net_buy = None
            total_flow = None
            if "buy_qty" in bars_df.columns and "sell_qty" in bars_df.columns:
                buy = pd.to_numeric(bars_df["buy_qty"], errors="coerce").fillna(0.0)
                sell = pd.to_numeric(bars_df["sell_qty"], errors="coerce").fillna(0.0)
                net_buy = buy - sell
                total_flow = (buy + sell).replace(0, np.nan)
            elif "cvd_change_1" in bars_df.columns:
                net_buy = pd.to_numeric(
                    bars_df["cvd_change_1"], errors="coerce"
                ).fillna(0.0)
                total_flow = pd.to_numeric(
                    bars_df.get("volume", 0.0), errors="coerce"
                ).fillna(0.0)
                total_flow = total_flow.replace(0, np.nan)
            elif "cvd" in bars_df.columns:
                cvd = pd.to_numeric(bars_df["cvd"], errors="coerce").fillna(0.0)
                net_buy = cvd.diff().fillna(0.0)
                total_flow = pd.to_numeric(
                    bars_df.get("volume", 0.0), errors="coerce"
                ).fillna(0.0)
                total_flow = total_flow.replace(0, np.nan)

            if net_buy is not None:
                # cvd_change_1 (per-bar delta)
                if "cvd_change_1" not in bars_df_indexed.columns:
                    bars_df_indexed["cvd_change_1"] = net_buy.values
                # cvd_change_5
                cvd_change_5 = net_buy.rolling(window=5, min_periods=1).sum()
                if self._want("cvd_change_5"):
                    self.timeframe_features[timeframe]["cvd_change_5"] = float(
                        cvd_change_5.iloc[-1]
                    )
                bars_df_indexed["cvd_change_5"] = cvd_change_5
                # cvd_change_20
                cvd_change_20 = net_buy.rolling(window=20, min_periods=1).sum()
                if self._want("cvd_change_20"):
                    self.timeframe_features[timeframe]["cvd_change_20"] = float(
                        cvd_change_20.iloc[-1]
                    )
                bars_df_indexed["cvd_change_20"] = cvd_change_20
                # cvd_normalized (per-bar delta / total_flow)
                if (
                    total_flow is not None
                    and "cvd_normalized" not in bars_df_indexed.columns
                ):
                    cvd_normalized = (
                        (net_buy / total_flow)
                        .replace([np.inf, -np.inf], np.nan)
                        .fillna(0.0)
                    )
                    bars_df_indexed["cvd_normalized"] = cvd_normalized
                    if self._want("cvd_normalized"):
                        self.timeframe_features[timeframe]["cvd_normalized"] = float(
                            cvd_normalized.iloc[-1]
                        )
                # cvd_change_5_normalized
                if self._want("cvd_change_5_normalized"):
                    if total_flow is None:
                        cvd_norm = 0.0
                    else:
                        total_flow_5 = total_flow.rolling(window=5, min_periods=1).sum()
                        cvd_norm = (
                            (cvd_change_5 / total_flow_5)
                            .replace([np.inf, -np.inf], np.nan)
                            .fillna(0.0)
                        )
                    self.timeframe_features[timeframe]["cvd_change_5_normalized"] = (
                        float(
                            cvd_norm.iloc[-1] if hasattr(cvd_norm, "iloc") else cvd_norm
                        )
                    )
                    if hasattr(cvd_norm, "iloc"):
                        bars_df_indexed["cvd_change_5_normalized"] = cvd_norm
                # cvd_change_20
                cvd_change_20 = net_buy.rolling(window=20, min_periods=1).sum()
                if self._want("cvd_change_20"):
                    self.timeframe_features[timeframe]["cvd_change_20"] = float(
                        cvd_change_20.iloc[-1]
                    )
                bars_df_indexed["cvd_change_20"] = cvd_change_20
                # cvd_normalized (per-bar delta / total_flow)
                if (
                    total_flow is not None
                    and "cvd_normalized" not in bars_df_indexed.columns
                ):
                    cvd_normalized = (
                        (net_buy / total_flow)
                        .replace([np.inf, -np.inf], np.nan)
                        .fillna(0.0)
                    )
                    bars_df_indexed["cvd_normalized"] = cvd_normalized
                    if self._want("cvd_normalized"):
                        self.timeframe_features[timeframe]["cvd_normalized"] = float(
                            cvd_normalized.iloc[-1]
                        )

        # Extra live features from base training plan
        try:
            from src.features.time_series.baseline_features import (
                compute_atr_percentile,
                compute_bb_width_features_from_series,
                compute_trend_r2_20_from_series,
                compute_volume_ratio_from_series,
            )
            from src.features.time_series.utils_volume_profile import (
                compute_volume_profile_vpvr_from_series,
            )

            df = bars_df.copy()
            if self._want("trend_r2_20"):
                out = compute_trend_r2_20_from_series(close=df["close"])
                val = out.iloc[-1]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                self.timeframe_features[timeframe]["trend_r2_20"] = float(val)
                bars_df_indexed["trend_r2_20"] = out
            if self._want("volume_ratio"):
                out = compute_volume_ratio_from_series(volume=df["volume"])
                self.timeframe_features[timeframe]["volume_ratio"] = float(
                    out["volume_ratio"].iloc[-1]
                )
                bars_df_indexed["volume_ratio"] = out["volume_ratio"]
            if self._want("bb_width_normalized") or self._want("bb_position"):
                out = compute_bb_width_features_from_series(
                    close=df["close"],
                    high=df["high"],
                    low=df["low"],
                )
                if self._want("bb_width_normalized"):
                    self.timeframe_features[timeframe]["bb_width_normalized"] = float(
                        out["bb_width_normalized"].iloc[-1]
                    )
                    bars_df_indexed["bb_width_normalized"] = out["bb_width_normalized"]
                if self._want("bb_position"):
                    self.timeframe_features[timeframe]["bb_position"] = float(
                        out["bb_position"].iloc[-1]
                    )
                    bars_df_indexed["bb_position"] = out["bb_position"]
            if self._want("atr_percentile"):
                out = compute_atr_percentile(df)
                self.timeframe_features[timeframe]["atr_percentile"] = float(
                    out["atr_percentile"].iloc[-1]
                )
                bars_df_indexed["atr_percentile"] = out["atr_percentile"]
            if any(k.startswith("vpvr_") for k in self.live_feature_set) or self._want(
                "vpvr_pvp"
            ):
                out = compute_volume_profile_vpvr_from_series(
                    close=df["close"],
                    high=df["high"],
                    low=df["low"],
                    volume=df["volume"],
                )
                last = out.iloc[-1].to_dict()
                for k, v in last.items():
                    if self._want(str(k)):
                        self.timeframe_features[timeframe][str(k)] = float(v)
                for k in out.columns:
                    bars_df_indexed[str(k)] = out[k]
        except Exception:
            pass

        # Batch fallback: compute tier features using FeatureComputer on bar buffer.
        if self._feature_loader is not None and self.live_feature_nodes:
            try:
                req = list(self.live_feature_nodes)
                # Filter nodes that require columns not present in bars_df.
                feats_cfg = (self._feature_deps or {}).get("features") or {}
                bar_cols = set(bars_df_indexed.columns)
                filtered = []
                skipped = []
                for n in req:
                    info = feats_cfg.get(n)
                    if isinstance(info, dict):
                        req_cols = set(info.get("required_columns") or [])
                        deps = info.get("dependencies") or []
                        if req_cols and not req_cols.issubset(bar_cols) and not deps:
                            skipped.append(str(n))
                            continue
                    filtered.append(n)
                self._last_skipped_nodes = skipped
                df2 = bars_df_indexed.copy()
                df2 = self._feature_loader.load_features_from_requested(
                    df2, requested_features=filtered, fit=False
                )
                last = df2.iloc[-1].to_dict()
                for k, v in last.items():
                    if self._want(str(k)):
                        try:
                            self.timeframe_features[timeframe][str(k)] = float(v)
                        except Exception:
                            continue
            except Exception:
                pass

        # Final backfill: include any computed columns in the live feature set
        try:
            last_row = bars_df_indexed.iloc[-1].to_dict()
            for k in self.live_feature_set:
                if k in last_row and k not in self.timeframe_features[timeframe]:
                    v = last_row.get(k)
                    if v is None:
                        continue
                    try:
                        if np.isscalar(v):
                            self.timeframe_features[timeframe][str(k)] = float(v)
                    except Exception:
                        continue
        except Exception:
            pass

    def get_features(self) -> Dict[str, float]:
        """
        获取当前所有特征（兼容接口）

        注意：实盘主路径已改为 compute_features_batch()，
        此方法保留用于后向兼容。

        Returns:
            特征字典
        """
        # 优先返回批量计算结果（如果有）
        if self._batch_features:
            return dict(self._batch_features)

        features = {}

        # Tick 级特征
        for k, v in (self.current_features or {}).items():
            if self._want(str(k)):
                features[str(k)] = float(v)

        # 时间框架特征（扁平化）
        for tf, tf_features in self.timeframe_features.items():
            for key, value in tf_features.items():
                pref = f"{tf}_{key}"
                if self._want(pref):
                    features[pref] = value
                if self.primary_timeframe and str(tf) == str(self.primary_timeframe):
                    if self._want(str(key)):
                        features[str(key)] = value

        self._log_missing_features(features)

        return features

    # ── 共享核心：步骤 1-3 的单一实现 ──────────────────────────
    def _compute_features_core(
        self,
        bars_1min: pd.DataFrame,
        ticks_1min: pd.DataFrame,
        primary_timeframe: Optional[str] = None,
        fp_max_bars: Optional[int] = 200,
    ) -> Optional[pd.DataFrame]:
        """共享的特征计算核心 (steps 1-3).

        compute_features_batch 和 compute_features_dataframe 的唯一入口，
        消除代码重复，确保逻辑一致。

        Returns:
            bars_tf DataFrame (所有计算完的列)，或 None（数据不足）。
        """
        import logging

        _logger = logging.getLogger(__name__)

        ptf = primary_timeframe or self.primary_timeframe or "240T"

        if bars_1min is None or bars_1min.empty:
            _logger.warning("_compute_features_core: bars_1min is empty")
            return None

        # ── 1. 重采样 1min bars → primary_timeframe (4h) ──
        bars = bars_1min.copy()

        # 确保 DatetimeIndex
        if not isinstance(bars.index, pd.DatetimeIndex):
            if "timestamp" in bars.columns:
                bars.index = pd.to_datetime(bars["timestamp"], utc=True)
            else:
                _logger.warning("_compute_features_core: no timestamp column")
                return None

        # 确保 tz-aware (UTC)
        if bars.index.tz is None:
            bars.index = bars.index.tz_localize("UTC")

        # 重采样
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        for col, agg in [
            ("buy_volume", "sum"),
            ("sell_volume", "sum"),
            ("buy_count", "sum"),
            ("sell_count", "sum"),
            ("trade_count", "sum"),
            ("delta", "sum"),
        ]:
            if col in bars.columns:
                agg_dict[col] = agg

        bars_tf = bars.resample(ptf).agg(agg_dict).dropna(subset=["close"])

        # 保留 _symbol 列 (OI join 等特征需要)
        if "_symbol" in bars.columns:
            bars_tf["_symbol"] = bars["_symbol"].iloc[0]

        # 添加 buy_qty / sell_qty（研发格式，用于 CVD 等特征节点）
        if "buy_volume" in bars_tf.columns and "sell_volume" in bars_tf.columns:
            bars_tf["buy_qty"] = bars_tf["buy_volume"]
            bars_tf["sell_qty"] = bars_tf["sell_volume"]

            # 计算 CVD 相关特征（研发格式）
            delta = bars_tf["buy_volume"] - bars_tf["sell_volume"]
            total_flow = bars_tf["buy_volume"] + bars_tf["sell_volume"]

            bars_tf["cvd_change_1"] = delta
            bars_tf["cvd_change_5"] = delta.rolling(window=5, min_periods=1).sum()
            bars_tf["cvd_change_20"] = delta.rolling(window=20, min_periods=1).sum()
            bars_tf["cvd_roll20"] = delta.rolling(window=20, min_periods=1).sum()
            bars_tf["cvd_roll60"] = delta.rolling(window=60, min_periods=1).sum()
            bars_tf["cvd_roll288"] = delta.rolling(window=288, min_periods=1).sum()
            bars_tf["cvd"] = delta.cumsum()
            bars_tf["cvd_normalized"] = (delta / total_flow.replace(0, np.nan)).fillna(
                0
            )
            total_flow_5 = total_flow.rolling(window=5, min_periods=1).sum()
            bars_tf["cvd_change_5_normalized"] = (
                bars_tf["cvd_change_5"] / total_flow_5.replace(0, np.nan)
            ).fillna(0)
            bars_tf["taker_buy_ratio"] = (
                bars_tf["buy_volume"] / total_flow.replace(0, np.nan)
            ).fillna(0.5)
        elif ticks_1min is not None and not ticks_1min.empty:
            # 🐛 Fix: 磁盘 1min bars 没有 buy_volume/sell_volume 时，
            #   从 tick 数据中计算 CVD，确保 fer_failure_signals_f 有输入。
            #   磁盘 bars 可能来自 warmup 下载，不含买卖量分列。
            try:
                _cvd_computed = self._compute_cvd_from_ticks(
                    bars_tf, ticks_1min, ptf, _logger
                )
                if _cvd_computed:
                    _logger.info("  CVD 从 tick 数据补算成功")
            except Exception as _cvd_e:
                _logger.warning("  CVD 从 tick 补算失败: %s", _cvd_e)

        _logger.info(
            "_compute_features_core: %d 1min bars → %d %s bars",
            len(bars_1min),
            len(bars_tf),
            ptf,
        )

        if len(bars_tf) < 10:
            _logger.warning(
                "_compute_features_core: too few %s bars (%d) from %d 1min bars",
                ptf,
                len(bars_tf),
                len(bars_1min),
            )
            return None

        # ── 2. 计算订单流特征 (VPIN / Trade Clustering / Footprint) ──
        # Gate trade_cluster + derived VPIN on live plan (soft_phase only needs
        # raw `vpin`; avoid always-on cluster compute for TPC/BPC slim plans).
        if ticks_1min is not None and not ticks_1min.empty:
            try:
                from src.features.live_tail_context import live_feature_tail

                need_trade_cluster = any(
                    k.startswith("trade_cluster") for k in (self.live_feature_set or [])
                ) or any(
                    "trade_cluster" in str(n) for n in (self.live_feature_nodes or [])
                )
                need_vpin_derived = any(
                    (
                        k.startswith("vpin_")
                        and k
                        not in {
                            "vpin",
                            "vpin_signed_imbalance",
                            "vpin_last",
                            "vpin_max",
                            "vpin_min",
                            "vpin_std",
                            "vpin_count",
                            "vpin_skewness",
                            "vpin_trend",
                            "vpin_signed_imbalance_last",
                            "vpin_signed_imbalance_max",
                        }
                    )
                    for k in (self.live_feature_set or [])
                ) or any(
                    n
                    in {
                        "vpin_derived_features_f",
                        "vpin_block_features_f",
                        "vpin_features_f",
                        "vpin_ma_max_features_f",
                        "vpin_change_features_f",
                        "vpin_zscore_features_f",
                        "vpin_quantile_features_f",
                        "vpin_volatility_features_f",
                        "vpin_spike_features_f",
                        "vpin_momentum_features_f",
                        "vpin_signed_zscore_features_f",
                        "vpin_scene_semantic_scores_f",
                    }
                    for n in (self.live_feature_nodes or [])
                )
                # soft_phase / ofci / any vpin*/trade_cluster node → run orderflow
                need_orderflow = (
                    need_trade_cluster
                    or need_vpin_derived
                    or any(
                        k.startswith("vpin") or k.startswith("trade_cluster")
                        for k in (self.live_feature_set or [])
                    )
                    or any(
                        "vpin" in str(n) or "trade_cluster" in str(n)
                        for n in (self.live_feature_nodes or [])
                    )
                )

                ticks = ticks_1min.copy()
                if not isinstance(ticks.index, pd.DatetimeIndex):
                    if "timestamp" in ticks.columns:
                        ticks.index = pd.to_datetime(ticks["timestamp"], utc=True)
                if ticks.index.tz is None:
                    ticks.index = ticks.index.tz_localize("UTC")

                from src.features.time_series.utils_order_flow_features import (
                    extract_order_flow_features,
                    compute_trade_cluster_derived_features_from_base,
                    compute_vpin_derived_features_from_base,
                )
                from src.features.loader.feature_wrappers import (
                    compute_footprint_features,
                )

                # Bounded-tail: extract_order_flow_features slices bars + ticks when
                # live_feature_tail is active (same env as loader DAG).
                if need_orderflow:
                    with live_feature_tail(self._live_feature_tail):
                        of_df = extract_order_flow_features(
                            bars_tf,
                            ticks=ticks,
                            freq=ptf,
                            include_trade_clustering=need_trade_cluster,
                            compute_vpin_derived=need_vpin_derived,
                        )
                    existing_cols = set(bars_tf.columns)
                    new_cols = [c for c in of_df.columns if c not in existing_cols]
                    if new_cols:
                        bars_tf = bars_tf.join(of_df[new_cols], how="left")

                    if need_vpin_derived:
                        try:
                            vpin_derived = compute_vpin_derived_features_from_base(
                                bars_tf
                            )
                            for c in vpin_derived.columns:
                                if c not in bars_tf.columns:
                                    bars_tf[c] = vpin_derived[c]
                        except Exception:
                            pass

                    if need_trade_cluster:
                        try:
                            tc_derived = (
                                compute_trade_cluster_derived_features_from_base(
                                    bars_tf
                                )
                            )
                            for c in tc_derived.columns:
                                if c not in bars_tf.columns:
                                    bars_tf[c] = tc_derived[c]
                        except Exception:
                            pass

                try:
                    # 诊断: 检查 bars_tf 和 ticks 的时间范围是否匹配
                    _bars_tz = bars_tf.index.tz
                    _ticks_tz = ticks.index.tz
                    if _bars_tz != _ticks_tz:
                        _logger.warning(
                            "FP timezone mismatch: bars_tf.tz=%s, ticks.tz=%s",
                            _bars_tz,
                            _ticks_tz,
                        )
                        # 统一 timezone
                        if _bars_tz is None and _ticks_tz is not None:
                            bars_tf.index = bars_tf.index.tz_localize(_ticks_tz)
                        elif _ticks_tz is None and _bars_tz is not None:
                            ticks.index = ticks.index.tz_localize(_bars_tz)

                    _fp_input = bars_tf.tail(fp_max_bars) if fp_max_bars else bars_tf
                    fp_df = compute_footprint_features(
                        _fp_input,
                        ticks=ticks,
                        persist_monthly=False,
                    )
                    fp_new_cols = [c for c in fp_df.columns if c not in bars_tf.columns]
                    if fp_new_cols:
                        bars_tf = bars_tf.join(fp_df[fp_new_cols], how="left")
                except Exception as _fp_exc:
                    _logger.warning("Footprint 计算失败: %s", _fp_exc)

                if need_orderflow:
                    _logger.info(
                        "  OF features: %d cols from %d ticks "
                        "(trade_cluster=%s vpin_derived=%s)",
                        len(of_df.columns),
                        len(ticks_1min),
                        need_trade_cluster,
                        need_vpin_derived,
                    )
                else:
                    _logger.info(
                        "  OF features skipped (not in live plan); %d ticks available",
                        len(ticks_1min),
                    )
            except Exception as e:
                _logger.warning("  OF features failed: %s", e)

        # ── 3. StrategyFeatureLoader 批量计算 ──
        if self._feature_loader is not None and self.live_feature_nodes:
            try:
                req = list(self.live_feature_nodes)
                feats_cfg = (self._feature_deps or {}).get("features") or {}
                bar_cols = set(bars_tf.columns)

                tick_dependent_nodes = self._detect_tick_dependent_nodes(feats_cfg)

                def _has_tick_dependency(node_name: str, visited: set = None) -> bool:
                    if visited is None:
                        visited = set()
                    if node_name in visited:
                        return False
                    visited.add(node_name)
                    if node_name in tick_dependent_nodes:
                        return True
                    info = feats_cfg.get(node_name)
                    if isinstance(info, dict):
                        deps = info.get("dependencies") or []
                        for dep in deps:
                            if _has_tick_dependency(dep, visited):
                                return True
                    return False

                filtered = []
                skipped = []
                for n in req:
                    if _has_tick_dependency(n):
                        skipped.append(str(n))
                        continue
                    info = feats_cfg.get(n)
                    if isinstance(info, dict):
                        req_cols = set(info.get("required_columns") or [])
                        deps = info.get("dependencies") or []
                        if req_cols and not req_cols.issubset(bar_cols) and not deps:
                            skipped.append(str(n))
                            continue
                    filtered.append(n)

                # ── 3a-fix. skipped 节点的非-tick 依赖加入 filtered ──
                filtered_set = set(filtered)
                for n in skipped:
                    if n in tick_dependent_nodes:
                        continue
                    info = feats_cfg.get(n)
                    if isinstance(info, dict):
                        for dep in info.get("dependencies") or []:
                            if dep not in filtered_set and not _has_tick_dependency(
                                dep
                            ):
                                filtered.append(dep)
                                filtered_set.add(dep)

                def _collect_deferred_bar_nodes(node_name: str, out: list[str]) -> None:
                    """Collect skipped DAG nodes that can run once bar columns exist."""
                    if node_name in out or node_name in tick_dependent_nodes:
                        return
                    info = feats_cfg.get(node_name)
                    if not isinstance(info, dict):
                        return
                    for dep in info.get("dependencies") or []:
                        _collect_deferred_bar_nodes(str(dep), out)
                    if node_name not in filtered_set:
                        out.append(node_name)

                deferred_second_pass: list[str] = []
                for n in skipped:
                    _collect_deferred_bar_nodes(str(n), deferred_second_pass)

                filtered = self._ensure_atr_f_in_loader_request(filtered)
                from src.features.live_tail_context import live_feature_tail

                with live_feature_tail(self._live_feature_tail):
                    bars_tf = self._feature_loader.load_features_from_requested(
                        bars_tf, requested_features=filtered, fit=False
                    )
                _logger.info(
                    "  Loader: %d nodes (%d skipped tick-dep) → %d cols",
                    len(filtered),
                    len(skipped),
                    len(bars_tf.columns),
                )

                # ── 3b. Iterative second pass ──
                # Some requested nodes depend on tick-derived columns that were already
                # materialized in step 2 (VPIN/footprint). Run deferred bar-only nodes
                # as their required columns become available, e.g. vpin_scene -> dual_*.
                # Keep the same live_feature_tail context so percentile / soft-phase
                # helpers that honor the context stay bounded.
                if deferred_second_pass:
                    from src.features.registry import get_compute_func
                    from src.features.loader.feature_computer import (
                        _build_call_args,
                    )
                    import inspect

                    pending = list(dict.fromkeys(deferred_second_pass))
                    with live_feature_tail(self._live_feature_tail):
                        while pending:
                            progressed = False
                            for n in list(pending):
                                info = feats_cfg.get(n)
                                if not isinstance(info, dict):
                                    pending.remove(n)
                                    continue
                                bar_cols_updated = set(bars_tf.columns)
                                req_cols = set(info.get("required_columns") or [])
                                if not req_cols.issubset(bar_cols_updated):
                                    continue

                                try:
                                    compute_func_name = info.get("compute_func", n)
                                    cfn = get_compute_func(compute_func_name)
                                    if cfn is None:
                                        pending.remove(n)
                                        continue
                                    info_filtered = dict(info)
                                    raw_mappings = info.get("column_mappings") or {}
                                    if raw_mappings:
                                        avail_mappings = {}
                                        for param, src in raw_mappings.items():
                                            if (
                                                isinstance(src, str)
                                                and src in bar_cols_updated
                                            ):
                                                avail_mappings[param] = src
                                            elif isinstance(src, list) and all(
                                                s in bar_cols_updated for s in src
                                            ):
                                                avail_mappings[param] = src
                                        info_filtered["column_mappings"] = (
                                            avail_mappings
                                        )
                                    call_args, call_kwargs = _build_call_args(
                                        info_filtered, bars_tf, n
                                    )
                                    sig = inspect.signature(cfn)
                                    accepts_var_kw = any(
                                        p.kind == inspect.Parameter.VAR_KEYWORD
                                        for p in sig.parameters.values()
                                    )
                                    if not accepts_var_kw and call_kwargs:
                                        allowed = set(sig.parameters.keys())
                                        call_kwargs = {
                                            k: v
                                            for k, v in call_kwargs.items()
                                            if k in allowed
                                        }
                                    result = cfn(*call_args, **call_kwargs)
                                    output_cols = info.get("output_columns", [n])
                                    if isinstance(result, tuple):
                                        if len(result) == len(output_cols):
                                            result = pd.DataFrame(
                                                dict(zip(output_cols, result))
                                            )
                                    bars_tf = self._join_loader_result_to_bars(
                                        bars_tf, result, info
                                    )
                                    pending.remove(n)
                                    progressed = True
                                    _logger.info("  Second pass: %s → OK", n)
                                except Exception as e:
                                    pending.remove(n)
                                    _logger.warning(
                                        "  Second pass failed for %s: %s", n, e
                                    )
                                    self._record_loader_error(n, ptf, e)

                            if not progressed:
                                break

                if (
                    ticks_1min is not None
                    and not ticks_1min.empty
                    and tick_dependent_nodes
                    and skipped
                ):
                    # Must run inside the same bounded-tail context as the loader DAG
                    # so ``get_live_tick_tail_rows()`` sees the active bar tail and the
                    # OFCI tick path can slice; otherwise it silently computes full history.
                    with live_feature_tail(self._live_feature_tail):
                        bars_tf = self._compute_tick_dependent_loader_nodes(
                            bars_tf,
                            ticks_1min,
                            skipped=skipped,
                            tick_dependent_nodes=tick_dependent_nodes,
                            feats_cfg=feats_cfg,
                            primary_timeframe=ptf,
                            logger=_logger,
                        )
            except Exception as e:
                _logger.warning("⚠️ Loader failed: %s", e)
                import traceback

                _logger.warning("  Loader traceback: %s", traceback.format_exc())

        return self._ensure_baseline_atr_column(bars_tf)

    def _ensure_atr_f_in_loader_request(self, filtered: List[str]) -> List[str]:
        """Prepend ``atr_f`` so the loader materializes ``atr`` before dependents."""
        if not self._want("atr"):
            return filtered
        nodes = list(filtered)
        if "atr_f" in nodes:
            return nodes
        feats = (self._feature_deps or {}).get("features") or {}
        if "atr_f" in feats:
            nodes.insert(0, "atr_f")
        return nodes

    def _ensure_baseline_atr_column(
        self, bars_tf: Optional[pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """Guarantee ``atr`` on resampled bars when the live plan requires it.

        StrategyFeatureLoader can skip ``atr_f`` when nodes are filtered; execution
        and health checks still need a finite ``atr`` on the last bar.
        """
        if bars_tf is None or bars_tf.empty or not self._want("atr"):
            return bars_tf
        need_fill = "atr" not in bars_tf.columns
        if not need_fill and len(bars_tf) > 0:
            last = bars_tf["atr"].iloc[-1]
            need_fill = last is None or (np.isscalar(last) and pd.isna(last))
        if not need_fill:
            return bars_tf
        try:
            from src.features.time_series.baseline_features import (
                compute_atr_from_series,
            )

            out = compute_atr_from_series(
                high=pd.to_numeric(bars_tf["high"], errors="coerce"),
                low=pd.to_numeric(bars_tf["low"], errors="coerce"),
                close=pd.to_numeric(bars_tf["close"], errors="coerce"),
            )
            if "atr" in bars_tf.columns:
                bars_tf = bars_tf.drop(columns=["atr"])
            bars_tf = bars_tf.join(out, how="left")
        except Exception:
            pass
        return bars_tf

    # ── 元数据驱动的 warmup 特征识别 ─────────────────────────
    def _get_warmup_check_features(self) -> frozenset:
        """从 feature_dependencies.yaml 元数据识别需要大窗口 warmup 的特征。

        判定规则 (基于配置元数据，不猜后缀):
          1. compute_params 包含 percentile_window >= 100
          2. 或 compute_func 名含 "percentile" (如 compute_atr_percentile_from_series)

        这些特征使用大窗口 rolling + min_periods=window，
        warmup 数据不足时最后一行必然为 NaN。
        """
        if not self._feature_deps or not self.live_feature_set:
            return frozenset()

        feats_cfg = (self._feature_deps or {}).get("features") or {}
        check_cols: set = set()

        for node_name, info in feats_cfg.items():
            if not isinstance(info, dict):
                continue
            params = info.get("compute_params") or {}
            func_name = str(info.get("compute_func", "")).lower()

            # 规则 1: 有 percentile_window >= 100 的节点
            has_large_pct_window = params.get("percentile_window", 0) >= 100
            # 规则 2: compute_func 名含 "percentile"
            is_pct_func = "percentile" in func_name

            if has_large_pct_window or is_pct_func:
                for col in info.get("output_columns") or []:
                    check_cols.add(col)

        return frozenset(check_cols & self.live_feature_set)

    def _validate_warmup(
        self,
        bars_tf: pd.DataFrame,
        features: Optional[Dict[str, float]] = None,
    ) -> None:
        """校验大窗口百分位特征是否因 warmup 不足而为 NaN。

        在 compute_features_batch (实盘) 和 compute_features_dataframe (事件回测)
        中共享调用，确保两条路径的校验逻辑一致。

        Args:
            bars_tf: 计算完成的特征 DataFrame
            features: 最后一行特征字典 (batch 模式提供);
                      为 None 时自动从 bars_tf 最后一行提取。
        """
        import logging

        _logger = logging.getLogger(__name__)

        warmup_features = self._get_warmup_check_features()
        if not warmup_features:
            return

        # 如果没传 features dict，从 DataFrame 最后一行构建
        if features is None and len(bars_tf) > 0:
            last_row = bars_tf.iloc[-1]
            features = {}
            for k, v in last_row.items():
                if v is None or not np.isscalar(v):
                    continue
                if isinstance(v, (str, bytes)):
                    continue
                if pd.isna(v):
                    continue
                try:
                    features[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
        elif features is None:
            features = {}

        nan_features = []
        for k in warmup_features:
            if k not in features:
                in_cols = k in bars_tf.columns
                last_val = bars_tf[k].iloc[-1] if in_cols else "N/A"
                nonnull = int(bars_tf[k].notna().sum()) if in_cols else 0
                _logger.warning(
                    "  ⚠️ warmup_check: %s missing from features "
                    "(in_cols=%s, last_val=%s, nonnull_rows=%d/%d)",
                    k,
                    in_cols,
                    last_val,
                    nonnull,
                    len(bars_tf),
                )
                nan_features.append(k)

        if nan_features:
            raise RuntimeError(
                f"Warmup 不足: {len(nan_features)} 个大窗口百分位特征为 NaN "
                f"(rolling window 未满足 min_periods). "
                f"缺失特征: {nan_features}. "
                f"当前 bars: {len(bars_tf)}. "
                f"请运行: bash live/scripts/prepare_warmup_ticks.sh highcap 6"
            )

    def _apply_weekly_ema_seed_override(
        self,
        bars_tf: pd.DataFrame,
        features: Dict[str, float],
    ) -> Dict[str, float]:
        """Override weekly_ema_200_position from Vision spot macro seed when configured."""
        col = "weekly_ema_200_position"
        if col not in self.live_feature_set:
            return features
        root = self.weekly_ema_seed_root
        if not root:
            return features
        sym = getattr(self, "_current_symbol", "") or ""
        if not sym or bars_tf is None or bars_tf.empty:
            return features
        try:
            from src.live_data_stream.spot_weekly_ema_seed import (
                weekly_ema_position_from_seed,
            )

            last = bars_tf.iloc[-1]
            close = float(last.get("close", np.nan))
            if not np.isfinite(close):
                return features
            bar_ts = bars_tf.index[-1]
            pos = weekly_ema_position_from_seed(
                close=close,
                bar_ts=bar_ts,
                seed_root=root,
                symbol=sym,
            )
            out = dict(features)
            if pos is None:
                # Fail closed: do not keep short-buffer EMA guess; spot prefilter needs NaN.
                out[col] = float("nan")
                return out
            out[col] = float(pos)
            return out
        except Exception:
            return features

    @staticmethod
    def _drop_forming_primary_bar(
        bars_tf: pd.DataFrame,
        *,
        primary_timeframe: str,
        now: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """In ``2h_close`` mode, drop the still-open last resample bucket.

        Trigger fires when a new primary bucket *opens*; pandas ``resample`` then
        already contains a partial row for that forming bucket. Publishing
        ``iloc[-1]`` would freeze incomplete close/chop for the whole bar (never
        rewritten at true close). Keep only bars with ``index < open_bucket``.
        """
        if bars_tf is None or bars_tf.empty:
            return bars_tf
        mode = os.getenv("MLBOT_FEATURE_COMPUTE_MODE", "2h_close").strip().lower()
        if mode != "2h_close":
            return bars_tf
        ptf = str(primary_timeframe or "120T").strip()
        # "120T" / "120min" / "2h"
        try:
            if ptf.lower().endswith("t"):
                bar_minutes = int(ptf[:-1])
            elif ptf.lower().endswith("min"):
                bar_minutes = int(ptf[:-3])
            elif ptf.lower().endswith("h"):
                bar_minutes = int(float(ptf[:-1]) * 60)
            else:
                bar_minutes = int(pd.Timedelta(ptf).total_seconds() // 60)
        except Exception:
            bar_minutes = 120
        ts_now = now if now is not None else pd.Timestamp.now(tz="UTC")
        if ts_now.tzinfo is None:
            ts_now = ts_now.tz_localize("UTC")
        else:
            ts_now = ts_now.tz_convert("UTC")
        open_bucket = ts_now.floor(f"{bar_minutes}min")
        closed = bars_tf[bars_tf.index < open_bucket]
        # Fail-closed: never fall back to the forming row (that reintroduces the bug).
        return closed

    # ── 公开接口 ─────────────────────────────────────────────
    def compute_features_batch(
        self,
        bars_1min: pd.DataFrame,
        ticks_1min: pd.DataFrame,
        primary_timeframe: Optional[str] = None,
    ) -> Dict[str, float]:
        """实盘入口: 返回最后一行特征字典。

        每 15 分钟调用一次 (order_flow_listener)。
        内部调用 _compute_features_core (共享步骤 1-3)，
        然后提取最后一行 + warmup 校验。
        """
        import logging

        _logger = logging.getLogger(__name__)

        ptf = primary_timeframe or self.primary_timeframe or "240T"
        bars_tf = self._compute_features_core(
            bars_1min,
            ticks_1min,
            primary_timeframe,
            fp_max_bars=200,
        )
        if bars_tf is None or bars_tf.empty:
            return {}

        bars_tf = self._drop_forming_primary_bar(bars_tf, primary_timeframe=ptf)
        if bars_tf is None or bars_tf.empty:
            return {}

        # ── 4. 提取最后一行 ──
        features: Dict[str, Any] = {}
        last_row = bars_tf.iloc[-1].to_dict()
        # Closed-bar index for 2h_close bus labeling (listener prefers this over now.floor).
        try:
            features["_bar_timestamp"] = pd.Timestamp(bars_tf.index[-1]).isoformat()
        except Exception:
            pass
        for k, v in last_row.items():
            key = str(k)
            if not self._want(key):
                continue
            if v is None or (np.isscalar(v) and pd.isna(v)):
                continue
            if key == "trend_direction" and isinstance(v, str):
                # Coerced to ±1 in enrich_multileg_runtime_features for parquet.
                continue
            try:
                if np.isscalar(v):
                    features[key] = float(v)
            except (ValueError, TypeError):
                continue

        if self.live_feature_set:
            enrich_multileg_runtime_features(features, wanted=self.live_feature_set)

        # ── 5. 校验大窗口百分位特征 (基于 feature_dependencies 元数据) ──
        self._validate_warmup(bars_tf, features)

        # ── 5b. Live macro seed override for weekly_ema_200_position ──
        features = self._apply_weekly_ema_seed_override(bars_tf, features)

        # 缓存结果（用于 get_features() 兼容接口）
        self._batch_features = features

        ptf = primary_timeframe or self.primary_timeframe or "240T"
        _logger.info("  Final: %d features", len(features))
        self._log_missing_features(features)

        # ── 6. 特征健康报告 (实盘 + Prometheus) ──
        sym = getattr(self, "_current_symbol", "")
        self.report_feature_health(
            features,
            symbol=sym,
            timeframe=ptf,
            update_prometheus=True,
        )

        return features

    def compute_features_dataframe(
        self,
        bars_1min: pd.DataFrame,
        ticks_1min: pd.DataFrame,
        primary_timeframe: Optional[str] = None,
    ) -> pd.DataFrame:
        """事件回测入口: 返回完整 DataFrame（所有行）。

        与 compute_features_batch 共享相同的 _compute_features_core，
        保证逻辑一致。用于一次性计算全量特征后逐行喂给 decide()。
        """
        import logging

        _logger = logging.getLogger(__name__)

        bars_tf = self._compute_features_core(
            bars_1min,
            ticks_1min,
            primary_timeframe,
            fp_max_bars=None,
        )
        if bars_tf is None or bars_tf.empty:
            return pd.DataFrame()

        # ── 4. 过滤列，只保留 live_feature_set 需要的 ──
        if self.live_feature_set:
            keep = [c for c in bars_tf.columns if self._want(str(c))]
            bars_tf = bars_tf[keep]

        # ── 5. warmup 校验 (与实盘一致) ──
        self._validate_warmup(bars_tf)

        _logger.info(
            "  compute_features_dataframe done: %d rows × %d cols",
            len(bars_tf),
            len(bars_tf.columns),
        )
        return bars_tf

    def _log_missing_features(self, features: Mapping[str, Any]) -> None:
        if not self.live_feature_set:
            return
        missing = [
            k for k in self.live_feature_set if not live_feature_satisfied(k, features)
        ]
        now = time.time()
        if self._last_missing_log_ts is None or now - self._last_missing_log_ts >= 60:
            if missing:
                preview = ", ".join(missing[:10])
                print(f"⚠️ live_feature_missing ({len(missing)}): {preview}")
            if self._last_skipped_nodes:
                preview = ", ".join(self._last_skipped_nodes[:10])
                print(
                    f"⚠️ live_feature_nodes_skipped ({len(self._last_skipped_nodes)}): {preview}"
                )
            self._last_missing_log_ts = now

    # ── 特征健康报告 ───────────────────────────────

    def _record_loader_error(self, node: str, timeframe: str, exc: Exception) -> None:
        """Record a feature loader error for Prometheus tracking."""
        try:
            from src.time_series_model.live.metrics_exporter import METRICS

            sym = getattr(self, "_current_symbol", "unknown")
            METRICS.feature_loader_errors.labels(
                symbol=sym, timeframe=timeframe, node=node
            ).inc()
        except Exception:
            pass

    # 关键特征前缀 — 这些特征缺失通常意味着上游数据链路断裂
    _CRITICAL_PREFIXES = ("atr", "oi_", "funding_oi_")
    # Side-gated features intentionally NaN on the inactive regime leg (see tpc_macro_pullback_pct_f).
    _HIGH_NAN_OK_COLUMNS = frozenset(
        {
            "tpc_macro_pullback_pct_long",
            "tpc_macro_pullback_pct_short",
        }
    )

    @staticmethod
    def _is_critical(name: str) -> bool:
        """Is this a critical feature that should never be NaN."""
        for p in IncrementalFeatureComputer._CRITICAL_PREFIXES:
            if name == p or name.startswith(p):
                return True
        return False

    def _missing_or_nan_features(self, features: Dict[str, Any]) -> List[str]:
        """Expected live features that are absent or hold NaN/None."""
        if not self.live_feature_set:
            return []
        bad: List[str] = []
        for key in sorted(self.live_feature_set):
            if key not in features:
                bad.append(key)
                continue
            val = features.get(key)
            try:
                if val is None or (np.isscalar(val) and pd.isna(val)):
                    bad.append(key)
            except (TypeError, ValueError):
                bad.append(key)
        return bad

    def report_feature_health(
        self,
        features: Dict[str, float],
        symbol: str = "",
        timeframe: str = "",
        *,
        update_prometheus: bool = False,
    ) -> Dict[str, Any]:
        """Compute and log a feature health report.

        Returns a dict with:
          total, expected, missing_count, nan_ratio,
          critical_nan (list of critical features missing),
          missing_names (first 20).

        If update_prometheus=True, also updates Prometheus gauges.
        """
        import logging

        _logger = logging.getLogger(__name__)

        expected = len(self.live_feature_set) if self.live_feature_set else 0
        total = len(features)
        missing = self._missing_or_nan_features(features)
        missing_count = len(missing)
        nan_ratio = missing_count / expected if expected > 0 else 0.0

        # Critical features check
        critical_nan = [m for m in missing if self._is_critical(m)]

        report = {
            "symbol": symbol,
            "timeframe": timeframe,
            "total": total,
            "expected": expected,
            "missing_count": missing_count,
            "nan_ratio": round(nan_ratio, 4),
            "critical_nan": critical_nan,
            "missing_names": missing[:20],
        }

        # ── 日志输出 ──
        if critical_nan:
            _logger.error(
                "🚨 [%s][%s] CRITICAL features NaN! %d critical missing: %s  "
                "(total=%d, expected=%d, nan_ratio=%.1f%%)",
                symbol,
                timeframe,
                len(critical_nan),
                critical_nan,
                total,
                expected,
                nan_ratio * 100,
            )
        elif nan_ratio > 0.2:
            _logger.warning(
                "⚠️ [%s][%s] Feature health WARNING: %d/%d missing (%.1f%%). "
                "First 10: %s",
                symbol,
                timeframe,
                missing_count,
                expected,
                nan_ratio * 100,
                missing[:10],
            )
        elif nan_ratio > 0.05:
            _logger.info(
                "🟡 [%s][%s] Feature health: %d/%d missing (%.1f%%)",
                symbol,
                timeframe,
                missing_count,
                expected,
                nan_ratio * 100,
            )
        else:
            _logger.info(
                "✅ [%s][%s] Feature health OK: %d/%d computed (%.1f%% coverage)",
                symbol,
                timeframe,
                total,
                expected,
                (1 - nan_ratio) * 100,
            )

        # ── Prometheus ──
        if update_prometheus:
            try:
                from src.time_series_model.live.metrics_exporter import METRICS

                sym = symbol or "unknown"
                tf = timeframe or "unknown"
                METRICS.feature_total.labels(symbol=sym, timeframe=tf).set(total)
                METRICS.feature_expected.labels(symbol=sym, timeframe=tf).set(expected)
                METRICS.feature_nan_count.labels(symbol=sym, timeframe=tf).set(
                    missing_count
                )
                METRICS.feature_nan_ratio.labels(symbol=sym, timeframe=tf).set(
                    nan_ratio
                )
                METRICS.feature_critical_nan.labels(symbol=sym, timeframe=tf).set(
                    1 if critical_nan else 0
                )
                # P5: regime_state / ood_score
                if "regime_state" in features:
                    METRICS.regime_state.labels(symbol=sym, timeframe=tf).set(
                        features["regime_state"]
                    )
                if "ood_score" in features:
                    METRICS.ood_score.labels(symbol=sym, timeframe=tf).set(
                        features["ood_score"]
                    )
            except Exception:
                pass  # Prometheus optional

        return report

    def report_feature_health_df(
        self,
        features_df: pd.DataFrame,
        symbol: str = "",
        timeframe: str = "",
    ) -> Dict[str, Any]:
        """Health report for DataFrame output (event backtest / dataframe mode).

        Checks the LAST row's NaN columns against live_feature_set.
        Also reports columns with >50% NaN across all rows.
        """
        import logging

        _logger = logging.getLogger(__name__)

        if features_df is None or features_df.empty:
            _logger.warning("🚨 [%s][%s] features_df is EMPTY", symbol, timeframe)
            return {"symbol": symbol, "timeframe": timeframe, "empty": True}

        # Last row check
        last_row = features_df.iloc[-1]
        nan_cols = sorted(last_row.index[last_row.isna()].tolist())
        total_cols = len(features_df.columns)

        # High NaN-rate columns (>50% NaN across all rows)
        nan_rates = features_df.isna().mean()
        high_nan_cols = sorted(
            c
            for c in nan_rates[nan_rates > 0.5].index.tolist()
            if c not in self._HIGH_NAN_OK_COLUMNS
        )

        # Critical features missing in last row
        critical_nan = [c for c in nan_cols if self._is_critical(c)]

        report = {
            "symbol": symbol,
            "timeframe": timeframe,
            "total_cols": total_cols,
            "last_row_nan_count": len(nan_cols),
            "last_row_nan_ratio": (
                round(len(nan_cols) / total_cols, 4) if total_cols else 0
            ),
            "high_nan_cols": high_nan_cols[:20],
            "critical_nan": critical_nan,
        }

        if critical_nan:
            _logger.error(
                "🚨 [%s][%s] CRITICAL features NaN in last row! %s  "
                "(total_cols=%d, nan_in_last=%d, high_nan_cols=%d)",
                symbol,
                timeframe,
                critical_nan,
                total_cols,
                len(nan_cols),
                len(high_nan_cols),
            )
        elif high_nan_cols:
            _logger.warning(
                "⚠️ [%s][%s] %d columns >50%% NaN: %s",
                symbol,
                timeframe,
                len(high_nan_cols),
                high_nan_cols[:10],
            )
        else:
            _logger.info(
                "✅ [%s][%s] DataFrame health OK: %d cols, %d NaN in last row",
                symbol,
                timeframe,
                total_cols,
                len(nan_cols),
            )

        return report

    def get_orderflow_features(self, window_minutes: int = 15) -> Dict[str, float]:
        """
        获取订单流特征（指定时间窗口）

        Args:
            window_minutes: 时间窗口（分钟）

        Returns:
            订单流特征字典
        """
        if not self.tick_buffer:
            return {"vpin": 0.0, "imbalance": 0.0, "total_vol": 0.0}

        cutoff_ns = self.tick_buffer[-1]["ts"] - window_minutes * 60 * 1_000_000_000
        recent_ticks = [t for t in self.tick_buffer if t["ts"] >= cutoff_ns]

        if not recent_ticks:
            return {"vpin": 0.0, "imbalance": 0.0, "total_vol": 0.0}

        buy_vol = sum(t["volume"] for t in recent_ticks if t["side"] == 1)
        sell_vol = sum(t["volume"] for t in recent_ticks if t["side"] == -1)
        total_vol = buy_vol + sell_vol

        vpin = self.current_features.get("vpin", 0.0)
        imbalance = (buy_vol - sell_vol) / (total_vol + 1e-12) if total_vol > 0 else 0.0

        return {
            "vpin": float(vpin),
            "imbalance": float(imbalance),
            "total_vol": float(total_vol),
        }

    def get_recent_bars(self, n: int = 200) -> list[Dict[str, Any]]:
        """
        Return recent bar records (oldest -> newest).

        Used by heuristic execution rules to compute simple structure signals
        without introducing full batch feature dependencies.
        """
        if n <= 0:
            return []
        xs = list(self.bar_buffer)
        if not xs:
            return []
        return xs[-int(n) :]

    def get_last_tick_ts_ns(self) -> Optional[int]:
        if not self.tick_buffer:
            return None
        try:
            return int(self.tick_buffer[-1]["ts"])
        except Exception:
            return None

    def reset(self) -> None:
        """重置所有状态"""
        self.tick_buffer.clear()
        self.bar_buffer.clear()
        self.vpin_buckets.clear()
        self.vpin_bucket_state = {
            "current_buy": 0.0,
            "current_sell": 0.0,
            "filled_value": 0.0,
        }
        self.current_features.clear()
        self.timeframe_features.clear()
        self._batch_features.clear()
