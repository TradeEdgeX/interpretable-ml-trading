"""
GenericLiveStrategy - 配置驱动的通用策略解析引擎

将策略逻辑完全从代码中解耦，通过 YAML 配置文件驱动决策流程。
支持任意策略的统一实现，只需提供对应的 archetype 配置。

核心组件：
1. DirectionEvaluator: 解析 direction.yaml 规则
2. GateEvaluator: 评估 gate.yaml 条件
3. EntryFilterChecker: 检查 entry_filters.yaml
4. Evidence（可选）: archetype.evidence 计算综合分，用于 PCM 仲裁/confidence；不调仓位倍数
5. ExecutionParamGenerator: 生成 execution.yaml 参数（含 regime_execution.size_multiplier）
"""

from __future__ import annotations

from collections import deque
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import yaml

from src.config.strategy_layout import resolve_strategy_package_under_root
from src.time_series_model.core.trade_intent import TradeIntent
from src.time_series_model.archetype.loader import (
    StrategyArchetype,
    load_strategy_archetype,
)
from src.time_series_model.execution.confidence_sizing import (
    resolve_confidence_size_scale,
)
from src.time_series_model.execution.entry_filter import (
    DerivedEntryFeatureState,
    check_entry_filters_or_single,
    entry_filter_applies_to_direction,
    load_entry_filters_config,
)
from src.time_series_model.live.execution_profile_apply import (
    rr_constraints_from_exec_params,
)
from src.time_series_model.live.fer_diagnostics import record_fer_entry_eval
from src.time_series_model.live.adverse_tree_gate_veto import AdverseTreeGateVeto
from src.time_series_model.live.srb_regime import (
    pick_srb_true_sr_level,
    resolve_srb_opposite_sr_level,
    should_reject_srb_wide_entry,
)
from src.time_series_model.live.direction_rule_ops import (
    dual_position_agree_deadband_scalar,
    is_direction_rule_enabled,
    parse_dual_rule,
    parse_signal_match_position_band_rule,
    parse_single_position_band_rule,
    single_position_band_scalar,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 1. 方向规则解析器
# =============================================================================


class DirectionEvaluator:
    """解析 direction.yaml 规则，确定交易方向"""

    def __init__(self, direction_config: Dict[str, Any]):
        self.config = direction_config
        self.rules = direction_config.get("direction_rules", [])
        self._prev_tree_score: Optional[float] = None
        self._prev_tree_score_by_symbol: Dict[str, float] = {}
        self._prev_dual_long: Optional[float] = None
        self._prev_dual_short: Optional[float] = None
        self._prev_dual_long_by_symbol: Dict[str, float] = {}
        self._prev_dual_short_by_symbol: Dict[str, float] = {}
        # fixed_direction: long/short → 忽略 direction_rules，强制固定方向
        _fd = direction_config.get("fixed_direction", None)
        if _fd == "long":
            self._fixed = 1
        elif _fd == "short":
            self._fixed = -1
        else:
            self._fixed = None
        # direction_filter: long/short → 方向模型正常运行，但只接受指定方向
        # 方向模型说 SHORT 时返回 0（跳过），不强制反向
        _df = direction_config.get("direction_filter", None)
        if _df == "long":
            self._filter = 1
        elif _df == "short":
            self._filter = -1
        else:
            self._filter = None
        # invert_side: after rules (+ optional direction_filter), flip ±1.
        # Research: fade / reverse of the same detector clock (e.g. SRB L3 fade).
        self._invert = bool(direction_config.get("invert_side", False))

    def evaluate(
        self, features: Dict[str, Any], *, symbol: str = ""
    ) -> Tuple[int, Optional[str]]:
        """
        评估方向规则

        Returns:
            (direction: int, matched_rule_id: Optional[str])
            direction: +1(多) / -1(空) / 0(无方向)
        """
        direction, rule_id = self._evaluate_core(features, symbol=symbol)
        if direction != 0 and self._invert:
            direction = -int(direction)
            if rule_id:
                rule_id = f"{rule_id}|invert_side"
        return direction, rule_id

    def _evaluate_core(
        self, features: Dict[str, Any], *, symbol: str = ""
    ) -> Tuple[int, Optional[str]]:
        """Rule stack + fixed_direction / direction_filter (no invert_side)."""
        # fixed_direction 优先 — 跳过所有规则直接返回固定方向
        if self._fixed is not None:
            return self._fixed, "fixed_direction"

        dual_cfg = self.config.get("dual_head") or {}
        if bool(dual_cfg.get("enabled")):
            direction, rule_id = self._evaluate_dual_head_direction(
                features, symbol=symbol
            )
            if (
                direction != 0
                and self._filter is not None
                and direction != self._filter
            ):
                return 0, None
            return direction, rule_id

        tree_hit = self._evaluate_tree_score_direction(features, symbol=symbol)
        if tree_hit is not None:
            direction, rule_id = tree_hit
            if (
                direction != 0
                and self._filter is not None
                and direction != self._filter
            ):
                return 0, None
            return direction, rule_id

        if not self.rules:
            return 0, None

        for rule in self.rules:
            if not is_direction_rule_enabled(rule):
                continue
            rule_id = rule.get("id", "unknown")
            compound = parse_signal_match_position_band_rule(rule)
            if compound is not None:
                consensus = compound.get("consensus_mode", "first")
                candidate = 0
                if consensus == "all":
                    votes = []
                    for sr in compound["signal_rules"]:
                        if not isinstance(sr, dict):
                            continue
                        if not is_direction_rule_enabled(sr):
                            continue
                        d_atom = self._evaluate_atomic_direction_rule(sr, features)
                        if d_atom != 0:
                            votes.append(d_atom)
                    if votes and all(v == votes[0] for v in votes):
                        candidate = votes[0]
                else:
                    for sr in compound["signal_rules"]:
                        if not isinstance(sr, dict):
                            continue
                        if not is_direction_rule_enabled(sr):
                            continue
                        d_atom = self._evaluate_atomic_direction_rule(sr, features)
                        if d_atom != 0:
                            candidate = d_atom
                            break
                if candidate == 0:
                    continue
                band_dir = single_position_band_scalar(
                    features.get(compound["band_feature"]),
                    float(compound["inner_abs"]),
                    float(compound["outer_abs"]),
                )
                if band_dir != candidate:
                    continue
                rsa = compound.get("require_sign_agreement")
                if rsa:
                    rsa_feat = str(rsa.get("feature", "")).strip()
                    try:
                        rsa_dead = float(rsa.get("deadband", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        rsa_dead = 0.0
                    if not rsa_feat:
                        continue
                    raw_sv = features.get(rsa_feat)
                    try:
                        sv = float(raw_sv)
                    except (TypeError, ValueError):
                        continue
                    if sv != sv or abs(sv) <= rsa_dead:
                        continue
                    if int(np.sign(sv)) != int(candidate):
                        continue
                logger.debug(
                    "方向匹配: rule=%s signal_match_position_band → direction=%s",
                    rule_id,
                    candidate,
                )
                if self._filter is not None and candidate != self._filter:
                    return 0, None
                return candidate, str(rule_id)

            dual = parse_dual_rule(rule)
            if dual is not None:
                col_a, col_b, eps = dual
                direction = dual_position_agree_deadband_scalar(
                    features.get(col_a), features.get(col_b), eps
                )
                if direction != 0:
                    logger.debug(
                        "方向匹配: rule=%s dual_deadband %s/%s eps=%s → direction=%s",
                        rule_id,
                        col_a,
                        col_b,
                        eps,
                        direction,
                    )
                    if self._filter is not None and direction != self._filter:
                        return 0, None
                    return direction, str(rule_id)
                continue

            band = parse_single_position_band_rule(rule)
            if band is not None:
                col, inner_abs, outer_abs = band
                direction = single_position_band_scalar(
                    features.get(col), inner_abs, outer_abs
                )
                if direction != 0:
                    if self._filter is not None and direction != self._filter:
                        return 0, None
                    return direction, str(rule_id)
                continue

            # threshold_compare: feature <= long_if_below → +1; feature >= short_if_above → -1。
            # 专为 CRF/ 边缘回归类策略设计（box_pos_120 ∈ [0,1]）。
            if str(rule.get("method", "")).strip().lower() == "threshold_compare":
                feat_name = str(rule.get("feature", "")).strip()
                thr = rule.get("thresholds") or {}
                raw_v = features.get(feat_name)
                if raw_v is None or feat_name == "":
                    continue
                try:
                    v = float(raw_v)
                except (TypeError, ValueError):
                    continue
                if v != v:  # NaN
                    continue
                long_if_below = thr.get("long_if_below")
                short_if_above = thr.get("short_if_above")
                direction = 0
                try:
                    if long_if_below is not None and v <= float(long_if_below):
                        direction = 1
                    elif short_if_above is not None and v >= float(short_if_above):
                        direction = -1
                except (TypeError, ValueError):
                    direction = 0
                if direction != 0:
                    logger.debug(
                        "方向匹配: rule=%s threshold_compare %s=%.4f → direction=%s",
                        rule_id,
                        feat_name,
                        v,
                        direction,
                    )
                    if self._filter is not None and direction != self._filter:
                        return 0, None
                    return direction, str(rule_id)
                continue

            feature_name = rule.get("feature", "")
            transform = rule.get("transform", "raw")

            value = features.get(feature_name)
            if value is None:
                continue

            try:
                value = float(value)
            except (TypeError, ValueError):
                continue

            direction = self._apply_transform(value, transform)

            if direction != 0:
                logger.debug(
                    f"方向匹配: rule={rule_id}, feature={feature_name}, "
                    f"value={value:.4f}, transform={transform} → direction={direction}"
                )
                if self._filter is not None and direction != self._filter:
                    return 0, None
                return direction, str(rule_id)

        return 0, None

    def _evaluate_dual_head_direction(
        self, features: Dict[str, Any], *, symbol: str = ""
    ) -> Tuple[int, Optional[str]]:
        """Independent P(long_win) / P(short_win) columns (injected or live)."""
        dual = self.config.get("dual_head") or {}
        long_cfg = dual.get("long") if isinstance(dual.get("long"), dict) else {}
        short_cfg = dual.get("short") if isinstance(dual.get("short"), dict) else {}
        long_col = str(long_cfg.get("score_column") or "score_long").strip()
        short_col = str(short_cfg.get("score_column") or "score_short").strip()
        long_thr = long_cfg.get("entry_threshold")
        short_thr = short_cfg.get("entry_threshold")
        if long_thr is None or short_thr is None:
            return 0, "dual_head_missing_threshold"

        def _prob(col: str) -> Optional[float]:
            raw = features.get(col)
            if raw is None:
                return None
            try:
                v = float(raw)
            except (TypeError, ValueError):
                return None
            if v != v:
                return None
            return v

        lp = _prob(long_col)
        sp = _prob(short_col)
        if lp is None or sp is None:
            return 0, "dual_head_missing_score"

        thr_block = self.config.get("thresholds") or {}
        entry_mode = str(thr_block.get("entry_mode", "level")).lower()
        long_raw = lp >= float(long_thr)
        short_raw = sp >= float(short_thr)

        sym_key = str(symbol or "").strip().upper()
        prev_l = self._prev_dual_long_by_symbol.get(sym_key)
        prev_s = self._prev_dual_short_by_symbol.get(sym_key)
        if sym_key:
            self._prev_dual_long_by_symbol[sym_key] = lp
            self._prev_dual_short_by_symbol[sym_key] = sp
        else:
            self._prev_dual_long = lp
            self._prev_dual_short = sp

        if entry_mode == "cross":
            if prev_l is None or prev_s is None or prev_l != prev_l or prev_s != prev_s:
                return 0, "dual_head_dead_zone"
            long_hit = long_raw and prev_l < float(long_thr)
            short_hit = short_raw and prev_s < float(short_thr)
        else:
            long_hit = long_raw
            short_hit = short_raw

        if bool(dual.get("reject_if_both_high")) and long_hit and short_hit:
            return 0, "dual_head_both_high"
        if bool(dual.get("reject_if_contradiction")) and long_hit and short_hit:
            return 0, "dual_head_contradiction"
        if long_hit:
            return 1, "dual_head_long"
        if short_hit:
            return -1, "dual_head_short"
        return 0, "dual_head_dead_zone"

    def _evaluate_tree_score_direction(
        self, features: Dict[str, Any], *, symbol: str = ""
    ) -> Optional[Tuple[int, Optional[str]]]:
        """Tree regression slug: direction.yaml ``source`` + ``thresholds`` block."""
        thr = self.config.get("thresholds")
        if not isinstance(thr, dict) or not thr:
            return None
        src = self.config.get("source") or {}
        col = str(src.get("score_column") or "score").strip()
        raw = features.get(col)
        if raw is None:
            return 0, None
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return 0, None
        if v != v:  # NaN
            return 0, None
        sym_key = str(symbol or "").strip().upper()
        long_entry = thr.get("long_entry")
        short_entry = thr.get("short_entry")
        ps = self.config.get("per_symbol_thresholds") or {}
        if sym_key and isinstance(ps.get(sym_key), dict):
            sym_thr = ps[sym_key]
            if sym_thr.get("long_entry") is not None:
                long_entry = sym_thr.get("long_entry")
            if sym_thr.get("short_entry") is not None:
                short_entry = sym_thr.get("short_entry")
        entry_mode = str(thr.get("entry_mode", "level")).lower()
        long_raw = long_entry is not None and v >= float(long_entry)
        short_raw = short_entry is not None and v <= float(short_entry)
        prev = (
            self._prev_tree_score_by_symbol.get(sym_key)
            if sym_key
            else self._prev_tree_score
        )
        if sym_key:
            self._prev_tree_score_by_symbol[sym_key] = v
        else:
            self._prev_tree_score = v
        if entry_mode == "cross":
            if prev is None or prev != prev:
                return 0, "tree_dead_zone"
            long_hit = long_raw and prev < float(long_entry)
            short_hit = short_raw and prev > float(short_entry)
        else:
            long_hit = long_raw
            short_hit = short_raw
        if long_hit:
            return 1, "tree_score_long"
        if short_hit:
            return -1, "tree_score_short"
        return 0, "tree_dead_zone"

    def _evaluate_atomic_direction_rule(
        self, rule: Dict[str, Any], features: Dict[str, Any]
    ) -> int:
        """单条子规则（不含 signal_match_position_band 嵌套）→ ±1 或 0。"""
        if not isinstance(rule, dict):
            return 0
        if str(rule.get("method", "")).strip().lower() == "signal_match_position_band":
            return 0
        dual = parse_dual_rule(rule)
        if dual is not None:
            col_a, col_b, eps = dual
            return dual_position_agree_deadband_scalar(
                features.get(col_a), features.get(col_b), eps
            )
        band = parse_single_position_band_rule(rule)
        if band is not None:
            col, inner_abs, outer_abs = band
            return single_position_band_scalar(features.get(col), inner_abs, outer_abs)
        feature_name = rule.get("feature", "")
        transform = rule.get("transform", "raw")
        value = features.get(feature_name)
        if value is None:
            return 0
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0
        return self._apply_transform(value, transform)

    def _apply_transform(self, value: float, transform: str) -> int:
        """应用变换函数"""
        if transform == "raw":
            return int(value)
        elif transform == "sign":
            return int(np.sign(value))
        elif transform == "negate_sign":
            return int(-np.sign(value))
        elif transform == "center_sign":
            return int(np.sign(value - 0.5))
        elif transform == "threshold":
            # 需要额外参数
            threshold = 0.0
            return 1 if value > threshold else -1
        else:
            return int(value)  # 默认 raw


# =============================================================================
# 2. Gate 条件评估引擎
# =============================================================================


class GateEvaluator:
    """评估 gate.yaml 条件，进行结构性过滤"""

    def __init__(
        self,
        archetype: StrategyArchetype,
        *,
        tree_gate_veto: Optional[AdverseTreeGateVeto] = None,
    ):
        self.archetype = archetype
        self._tree_gate_veto = tree_gate_veto

    def evaluate(
        self, features: Dict[str, Any], quantiles: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str], float]:
        """
        评估 Gate 条件

        Returns:
            (passed: bool, reasons: List[str], weight: float)
        """
        if self.archetype is None:
            return True, [], 1.0

        passed, reasons, weight = self.archetype.apply_gate(features, quantiles)
        if not passed:
            return passed, reasons, weight
        if self._tree_gate_veto is not None:
            ok, veto_reasons = self._tree_gate_veto.evaluate(features)
            if not ok:
                return False, list(reasons) + list(veto_reasons), 0.0
        return passed, reasons, weight


# =============================================================================
# 3. Entry Filter 检查器
# =============================================================================


class EntryFilterChecker:
    """检查 entry_filters.yaml 条件"""

    def __init__(self, entry_config: Dict[str, Any]):
        self.config = entry_config
        self.ef_state = DerivedEntryFeatureState()
        self._last_merged_features: Dict[str, Any] = {}
        self._last_direction: Optional[int] = None

    def check(
        self,
        features: Dict[str, Any],
        *,
        direction: Optional[int] = None,
    ) -> bool:
        """检查是否满足入场条件"""
        if not self.config:
            return True

        # 更新派生特征
        ef_features = self.ef_state.update(features)
        merged = {**features, **ef_features}
        self._last_merged_features = merged
        self._last_direction = direction

        return check_entry_filters_or_single(merged, self.config, direction=direction)

    def explain(self, features: Dict[str, Any]) -> List[str]:
        """Return compact failure reasons for the enabled entry filters."""
        if not self.config:
            return []

        merged = self._last_merged_features or dict(features)
        reasons: List[str] = []
        direction = self._last_direction
        for filt in self.config.get("filters", []):
            if not filt.get("enabled", False):
                continue
            if not entry_filter_applies_to_direction(filt, direction):
                continue
            fid = str(filt.get("id") or "entry_filter")
            failed: List[str] = []
            for cond in filt.get("conditions", []) or []:
                feat = str(cond.get("feature") or "")
                op = str(cond.get("operator") or "")
                threshold = cond.get("value")
                val = merged.get(feat)
                try:
                    v = float(val)
                    t = float(threshold)
                    passed = {
                        ">": v > t,
                        ">=": v >= t,
                        "<": v < t,
                        "<=": v <= t,
                        "==": v == t,
                        "!=": v != t,
                    }.get(op, False)
                    if not passed:
                        failed.append(f"{feat}={v:.4g} not {op} {t:.4g}")
                except (TypeError, ValueError):
                    failed.append(f"{feat}=missing")
            if failed:
                reasons.append(f"{fid}: " + "; ".join(failed[:3]))
        return reasons[:8]


# =============================================================================
# 4. Evidence 评分 — 由 archetype.evidence_config.compute_composite_score() 计算
#    在 GenericLiveStrategy.decide() 和 check_signal() 中内联调用
# =============================================================================


# =============================================================================
# 5. Execution 参数生成器
# =============================================================================


class ExecutionParamGenerator:
    """根据 execution.yaml（及可选 regime_execution 补丁）生成执行参数。

    ``evidence_score`` 参数为历史签名保留；当前不参与 stop/TP/size 的档位选择。
    """

    def __init__(self, execution_config: Dict[str, Any]):
        self.config = execution_config

    def generate_params(
        self,
        evidence_score: float,
        features: Optional[Dict[str, Any]] = None,
        direction: Optional[int] = None,
        regime_label: str = "neutral",
    ) -> Dict[str, Any]:
        """生成执行参数 — 统一使用全局参数（grid search 优化的）

        可选 ``features`` / ``direction``：SRB 实验用（regime_execution / sr_structural_exit），
        由事件回测注入 ``srb_regime_*`` / ``srb_sr_*`` 后传入；缺省行为与旧版完全一致。
        """
        features = features or {}
        sl_cfg = self.config.get("stop_loss", {})
        trail_cfg = sl_cfg.get("trailing", {}) or {}
        guardrails = sl_cfg.get("guardrails", {}) or {}
        breakeven_cfg = sl_cfg.get("breakeven", {}) or {}
        exec_constraints = self.config.get("execution_constraints", {}) or {}
        trailing_enabled = bool(trail_cfg.get("enabled", True))

        # take_profit: 必须检查 enabled 标志，读 target_r (与向量回测一致)
        tp_cfg = self.config.get("take_profit", {})
        tp_enabled = tp_cfg.get("enabled", False)
        take_profit_r = float(tp_cfg.get("target_r", 0.0)) if tp_enabled else 0.0
        stop_loss_type = str(sl_cfg.get("type", "fixed") or "fixed").strip().lower()
        take_profit_type = str(tp_cfg.get("type", "fixed") or "fixed").strip().lower()
        box_window = int(sl_cfg.get("box_window", tp_cfg.get("box_window", 120)) or 120)

        # time_stop_bars: 0 表示禁用时间止损 (fat tail 模式)
        # 注意: 不能用 `or 50`，因为 Python 中 0 or 50 = 50
        holding = self.config.get("holding", {}) or {}
        _raw_tsb = holding.get("time_stop_bars")
        _raw_mhb = holding.get("max_holding_bars")
        # time_stop_bars: 0 的常用语义是「不单独用 time_stop 名」而仍要尊重 max_holding_bars
        # （CRF / BPC）；若显式 0 且没有 max，则 0=禁用时间出场（ME 等长持模式）。
        if _raw_tsb is not None and int(_raw_tsb) == 0:
            if _raw_mhb is not None and int(_raw_mhb) > 0:
                _tsb = int(_raw_mhb)
            else:
                _tsb = 0
        elif _raw_tsb is not None and int(_raw_tsb) > 0:
            _tsb = int(_raw_tsb)
        elif _raw_mhb is not None and int(_raw_mhb) > 0:
            _tsb = int(_raw_mhb)
        else:
            _tsb = 0
        # ------------------------------------------------------------------
        # Unified breakeven lock（2026-04-22 重构，统一 mother_breakeven）
        # ------------------------------------------------------------------
        # 语义：MFE ≥ trigger_r × R 时，SL 抬到 entry + lock_level_r × R；
        #       tighten-only（硬编码：永不回退）；子仓通过 inherit_parent_stop 跟随。
        #   measure: "initial_risk"（默认）| "atr"
        #   lock_level_r: 0 = 纯保本；>0 = 锁进 N R 利润
        # 加仓前需要利润锁定；默认对 allow_add_on 策略启用。
        breakeven_enabled = bool(
            breakeven_cfg.get(
                "enabled",
                bool(exec_constraints.get("allow_add_on", False)),
            )
        )
        breakeven_trigger_r = float(breakeven_cfg.get("trigger_r", 1.0) or 1.0)
        breakeven_lock_level_r = float(breakeven_cfg.get("lock_level_r", 0.0) or 0.0)
        breakeven_measure = (
            str(breakeven_cfg.get("measure", "initial_risk") or "initial_risk")
            .strip()
            .lower()
        )
        if breakeven_measure not in {"initial_risk", "atr"}:
            breakeven_measure = "initial_risk"

        activation_r = (
            float(trail_cfg.get("activation_r", 1.0)) if trailing_enabled else None
        )
        trail_r = float(trail_cfg.get("trail_r", 1.5)) if trailing_enabled else None
        trail_expand_primary_atr = bool(trail_cfg.get("expand_with_primary_atr", False))

        # L3 dynamic trailing：当反向 L3 距离 < l3_near_threshold_atr × ATR 时使用 trail_r_near，
        # 否则使用 trail_r_far；两者都缺失时回退到 trail_r。
        def _opt_float(key: str) -> Optional[float]:
            v = trail_cfg.get(key)
            if v is None or not trailing_enabled:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        trail_r_far = _opt_float("trail_r_far")
        trail_r_near = _opt_float("trail_r_near")
        l3_near_threshold_atr = _opt_float("l3_near_threshold_atr")

        # E1 (2026-04-23): time_stop 分层解除 —— MFE ≥ 阈值时跳过 max_holding_bars
        _uncap_raw = holding.get("time_stop_uncap_mfe_r")
        try:
            time_stop_uncap_mfe_r = (
                float(_uncap_raw)
                if _uncap_raw is not None and float(_uncap_raw) > 0
                else None
            )
        except (TypeError, ValueError):
            time_stop_uncap_mfe_r = None

        # E2 (2026-04-23): L3 大级别结构化退出
        _l3_exit_cfg = self.config.get("l3_structural_exit") or {}
        l3_structural_exit_enabled = bool(_l3_exit_cfg.get("enabled", False))
        try:
            l3_structural_exit_buffer_atr = float(
                _l3_exit_cfg.get("buffer_atr", 0.25) or 0.25
            )
        except (TypeError, ValueError):
            l3_structural_exit_buffer_atr = 0.25

        result: Dict[str, Any] = {
            "initial_r": float(sl_cfg.get("initial_r", 2.0)),
            "activation_r": activation_r,
            "trail_r": trail_r,
            "trail_r_far": trail_r_far,
            "trail_r_near": trail_r_near,
            "l3_near_threshold_atr": l3_near_threshold_atr,
            "take_profit_r": take_profit_r,
            "stop_loss_type": stop_loss_type,
            "take_profit_type": take_profit_type,
            "box_window": box_window,
            "box_stop_buffer_frac": float(
                sl_cfg.get("box_buffer_frac", sl_cfg.get("stop_buffer_frac", 0.25))
                or 0.25
            ),
            "box_target_edge_frac": float(
                tp_cfg.get("edge_frac", tp_cfg.get("target_edge_frac", 0.15)) or 0.15
            ),
            "time_stop_bars": _tsb,
            "max_holding_bars": _tsb,
            "size_multiplier": 1.0,
            "structural_exit": sl_cfg.get("structural_exit"),
            "regime_exit_min_score": sl_cfg.get("regime_exit_min_score"),
            "regime_lifecycle_exit": sl_cfg.get("regime_lifecycle_exit") or {},
            "regime_exit": sl_cfg.get("regime_exit") or {},
            "cycle_close": sl_cfg.get("cycle_close") or {},
            "profit_take_ladder": sl_cfg.get("profit_take_ladder") or {},
            "min_stop_pct": guardrails.get("min_stop_pct"),
            "max_stop_pct": guardrails.get("max_stop_pct"),
            "breakeven_enabled": breakeven_enabled,
            "breakeven_trigger_r": breakeven_trigger_r,
            "breakeven_lock_level_r": breakeven_lock_level_r,
            "breakeven_measure": breakeven_measure,
            "allow_trailing": trailing_enabled and activation_r is not None,
            "trail_expand_primary_atr": trail_expand_primary_atr,
            "time_stop_uncap_mfe_r": time_stop_uncap_mfe_r,
            "l3_structural_exit_enabled": l3_structural_exit_enabled,
            "l3_structural_exit_buffer_atr": l3_structural_exit_buffer_atr,
            # SRB 结构化 SL（若 execution.yaml 下配置），透传到 rr_constraints
            "structural_sl": sl_cfg.get("structural_sl") or {},
            # Research trail overlay (default off when omitted)
            "trail_overlay": sl_cfg.get("trail_overlay") or {},
            # SRB scale-out bank (research; default off) — see scale_out_bank.py
            "scale_out_bank": self.config.get("scale_out_bank") or {},
        }

        # CRF/box-aware execution: capture causal box boundaries at signal time.
        # ``position_logic`` turns these into actual price SL/TP if execution.yaml
        # requests box_edge / opposite_edge. Missing/invalid values safely fall back
        # to ATR-based fixed R behavior.
        if stop_loss_type == "box_edge" or take_profit_type in {
            "opposite_edge",
            "box_mid",
        }:
            _box_map = {
                "box_hi": f"box_hi_{box_window}",
                "box_lo": f"box_lo_{box_window}",
                "box_width_pct": f"box_width_pct_{box_window}",
                "box_pos": f"box_pos_{box_window}",
            }
            for _dst, _src in _box_map.items():
                if _src in features and features.get(_src) is not None:
                    result[_dst] = features.get(_src)
            for _k in (
                f"box_hi_{box_window}",
                f"box_lo_{box_window}",
                f"box_width_pct_{box_window}",
                f"box_pos_{box_window}",
            ):
                if _k in features and features.get(_k) is not None:
                    result[_k] = features.get(_k)

        re_cfg = self.config.get("regime_execution") or {}
        if re_cfg.get("enabled"):
            bucket = str(features.get("srb_regime_bucket", "unknown"))
            buckets = re_cfg.get("buckets") or {}
            patch = buckets.get(bucket) or buckets.get("default") or {}
            for k in ("initial_r", "activation_r", "trail_r", "take_profit_r"):
                if k in patch and patch[k] is not None:
                    try:
                        result[k] = float(patch[k])
                    except (TypeError, ValueError):
                        pass
            if "size_multiplier" in patch and patch["size_multiplier"] is not None:
                try:
                    result["size_multiplier"] = float(patch["size_multiplier"])
                except (TypeError, ValueError):
                    pass
            if "allow_trailing" in patch:
                at = bool(patch["allow_trailing"])
                result["allow_trailing"] = at and result.get("activation_r") is not None

        # Confidence → size (entry filters unchanged). Multiplies current size.
        _cs = resolve_confidence_size_scale(
            self.config.get("confidence_sizing"), features
        )
        if _cs is not None:
            try:
                result["size_multiplier"] = float(
                    result.get("size_multiplier", 1.0) or 1.0
                ) * float(_cs)
            except (TypeError, ValueError):
                pass

        se_cfg = self.config.get("sr_structural_exit") or {}
        if se_cfg.get("enabled") and direction is not None:
            buf = float(se_cfg.get("buffer_atr", 0.25) or 0.25)
            sp: Optional[float] = None
            if direction == 1:
                sp = features.get("srb_sr_support")
            elif direction == -1:
                sp = features.get("srb_sr_resistance")
            if sp is not None:
                try:
                    fp = float(sp)
                    if np.isfinite(fp):
                        result["structural_exit"] = "sr_break_level"
                        result["sr_exit_price"] = fp
                        result["sr_exit_buffer_atr"] = buf
                except (TypeError, ValueError):
                    pass

        # ── Regime 自适应退出 (2026-06-10, extended 2026-06-23) ──
        # exit_by_regime 支持 per-regime 覆盖：
        #   bull/bear/neutral.trailing.enabled: bool     — 是否启用 trailing
        #   bull/bear/neutral.initial_r: float           — 覆盖初始止损 (ATR 倍数)
        #   bull/bear/neutral.trailing.activation_r: float — 覆盖 trailing 激活阈值
        #   bull/bear/neutral.trailing.trail_r: float      — 覆盖 trail 距离
        # 优先级：exit_by_regime (新) > regime_adaptive_exit.indicator (旧)
        exit_rg = sl_cfg.get("exit_by_regime") or {}
        ra_cfg = sl_cfg.get("regime_adaptive_exit") or {}

        if exit_rg:
            rg_action = exit_rg.get(regime_label) or exit_rg.get("neutral") or {}
            if rg_action:
                # ── Per-regime initial_r override ──
                rg_init = rg_action.get("initial_r")
                if rg_init is not None:
                    try:
                        result["initial_r"] = float(rg_init)
                    except (TypeError, ValueError):
                        pass

                # ── Per-regime trailing ──
                rg_trail = rg_action.get("trailing") or {}
                if isinstance(rg_trail, dict):
                    if rg_trail.get("enabled") is False:
                        result["allow_trailing"] = False
                        result["activation_r"] = None
                        result["trail_r"] = None
                    else:
                        # Per-regime trailing params override
                        rg_act = rg_trail.get("activation_r")
                        if rg_act is not None:
                            try:
                                result["activation_r"] = float(rg_act)
                            except (TypeError, ValueError):
                                pass
                        rg_tr = rg_trail.get("trail_r")
                        if rg_tr is not None:
                            try:
                                result["trail_r"] = float(rg_tr)
                            except (TypeError, ValueError):
                                pass
                        # trail_r_far / trail_r_near overrides (for L3 dynamic trailing)
                        for k in ("trail_r_far", "trail_r_near"):
                            v = rg_trail.get(k)
                            if v is not None:
                                try:
                                    result[k] = float(v)
                                except (TypeError, ValueError):
                                    pass

                # ── Per-regime breakeven override ──
                rg_be = rg_action.get("breakeven")
                if isinstance(rg_be, dict):
                    rg_be_tr = rg_be.get("trigger_r")
                    if rg_be_tr is not None:
                        try:
                            result["breakeven_trigger_r"] = float(rg_be_tr)
                        except (TypeError, ValueError):
                            pass
        elif ra_cfg.get("enabled") and features:
            # ── 旧风格：直接读特征判断（向后兼容）──
            indicator = str(ra_cfg.get("indicator", "ema_1200_position"))
            bull_thr = float(
                ra_cfg.get("bull_threshold", 0.18 if indicator != "adx" else 25)
            )
            bull_ov = ra_cfg.get("bull_override") or {}
            bull_trail = bull_ov.get("trailing") or {}
            disable_trailing = bull_trail.get("enabled") is False

            if indicator == "adx":
                adx_val = features.get("adx")
                if adx_val is not None:
                    try:
                        if float(adx_val) > bull_thr and disable_trailing:
                            result["allow_trailing"] = False
                            result["activation_r"] = None
                            result["trail_r"] = None
                    except (TypeError, ValueError):
                        pass
            else:
                ema_pos = features.get("ema_1200_position")
                if ema_pos is not None:
                    try:
                        if float(ema_pos) > bull_thr and disable_trailing:
                            result["allow_trailing"] = False
                            result["activation_r"] = None
                            result["trail_r"] = None
                    except (TypeError, ValueError):
                        pass

        return result


# =============================================================================
# 6. 通用 LiveStrategy 主类
# =============================================================================


class GenericLiveStrategy:
    """
    配置驱动的通用 LiveStrategy 解析引擎

    通过加载策略的 archetype 配置文件，自动构建决策管线。
    支持任意策略，只需提供标准的配置文件结构。
    """

    def __init__(
        self,
        strategy_name: str,
        strategies_root: str = "config/strategies",
        holding_yaml_path: Optional[str] = None,
        trade_size: float = 1.0,
        primary_timeframe: str = "240T",
        bar_minutes: int = 240,
    ):
        self.strategy_name = strategy_name
        self.strategies_root = strategies_root
        self.trade_size = trade_size
        self.primary_timeframe = primary_timeframe
        self.bar_minutes = bar_minutes
        self.holding_yaml_path = holding_yaml_path

        # 配置组件
        self.archetype: Optional[StrategyArchetype] = None
        self.direction_evaluator: Optional[DirectionEvaluator] = None
        self.gate_evaluator: Optional[GateEvaluator] = None
        self.entry_filter_checker: Optional[EntryFilterChecker] = None
        self.execution_generator: Optional[ExecutionParamGenerator] = None

        # 状态
        self._quantiles: Dict[str, Dict[str, float]] = {}
        self._last_tier_params: Optional[Dict[str, Any]] = None
        self._last_funnel: Dict[str, Any] = (
            {}
        )  # 上次 decide() 的漏斗结果 (含丰富元数据)
        self._decision_alignment: Dict[str, Any] = {
            "enabled": False,
            "mode": "prefilter_recent_window",
            "window_bars": 0,
            "layers_allow_recent_window": [],
            "layers_required_same_bar": [],
        }
        self._prefilter_recent_state: Dict[str, deque] = {}
        self._accumulation_bull_seen_by_symbol: Dict[str, bool] = {}

        # 加载配置
        self.load_configs()

    def _strategy_package_root(self) -> Path:
        """Live tree resolution only: never probe ``bad-candidates/`` (research archive)."""

        return resolve_strategy_package_under_root(
            Path(self.strategies_root),
            self.strategy_name,
            allow_bad_candidates=False,
        )

    def load_configs(self) -> None:
        """加载所有配置文件"""
        try:
            # 1. 加载 Archetype (Gate + Evidence + Execution)
            self.archetype = load_strategy_archetype(
                self.strategy_name, self.strategies_root, live_layout=True
            )
            _rg_rules = len(self.archetype.regime.rules)
            _rg_empty = self.archetype.regime.is_empty
            logger.info(
                f"\u2705 Archetype loaded: "
                f"{_rg_rules} regime rules (empty={_rg_empty}, "
                f"allowed_sides={list(self.archetype.regime.allowed_sides)}), "
                f"{len(self.archetype.prefilter.rules)} prefilter rules, "
                f"{len(self.archetype.gate.all_rules)} gate rules"
            )

            # 2. 加载 Direction 配置
            dir_path = self._strategy_package_root() / "archetypes" / "direction.yaml"
            if dir_path.exists():
                with open(dir_path, "r", encoding="utf-8") as f:
                    direction_cfg = yaml.safe_load(f) or {}
                self.direction_evaluator = DirectionEvaluator(direction_cfg)
                _fd = direction_cfg.get("fixed_direction")
                _n_rules = len(direction_cfg.get("direction_rules", []))
                logger.info(
                    f"✅ Direction config loaded: "
                    f"fixed_direction={_fd or 'none'}, {_n_rules} rules"
                )
            else:
                logger.warning(f"⚠️  Direction config not found: {dir_path}")

            # 3. 加载 Entry Filter 配置
            self.entry_filter_checker = EntryFilterChecker(
                load_entry_filters_config(
                    self.strategy_name, self.strategies_root, live_layout=True
                )
            )
            logger.info("✅ Entry filter config loaded")

            # 4. 初始化其他评估器（when_then rules + optional adverse tree veto）
            gate_yaml = self._strategy_package_root() / "archetypes" / "gate.yaml"
            tree_veto = AdverseTreeGateVeto.from_gate_yaml(
                gate_yaml, strategies_root=self.strategies_root
            )
            self.gate_evaluator = GateEvaluator(
                self.archetype, tree_gate_veto=tree_veto
            )
            self.execution_generator = ExecutionParamGenerator(
                self.archetype.execution.raw or {}
            )
            self._decision_alignment = self._load_decision_alignment_config()
            logger.info(
                "✅ Decision alignment: enabled=%s mode=%s window_bars=%d",
                self._decision_alignment.get("enabled", False),
                self._decision_alignment.get("mode", "prefilter_recent_window"),
                self._decision_alignment.get("window_bars", 0),
            )

        except Exception as e:
            logger.error(f"❌ Failed to load configs for {self.strategy_name}: {e}")
            raise

    def _load_decision_alignment_config(self) -> Dict[str, Any]:
        """从 strategy meta.yaml 读取跨 bar 对齐配置。"""
        cfg: Dict[str, Any] = {
            "enabled": False,
            "mode": "prefilter_recent_window",
            "window_bars": 0,
            "layers_allow_recent_window": [],
            "layers_required_same_bar": [],
        }
        meta_path = self._strategy_package_root() / "meta.yaml"
        if not meta_path.exists():
            return cfg

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                raw_meta = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning("⚠️  Failed to read meta.yaml for alignment config: %s", exc)
            return cfg

        meta_strategy = raw_meta.get("strategy", raw_meta)
        da = (
            meta_strategy.get("decision_alignment")
            or raw_meta.get("decision_alignment")
            or {}
        )
        if not isinstance(da, dict):
            return cfg

        window_bars = int(
            da.get("window_bars", da.get("alignment_window_bars", 0)) or 0
        )
        mode = str(da.get("mode", "prefilter_recent_window")).strip().lower()
        enabled = bool(da.get("enabled", False)) and window_bars > 0
        if mode not in {"prefilter_recent_window"}:
            mode = "prefilter_recent_window"

        cfg.update(
            {
                "enabled": enabled,
                "mode": mode,
                "window_bars": max(0, window_bars),
                "layers_allow_recent_window": list(
                    da.get("layers_allow_recent_window") or []
                ),
                "layers_required_same_bar": list(
                    da.get("layers_required_same_bar") or []
                ),
            }
        )
        return cfg

    def _update_prefilter_recent_state(self, symbol: str, passed: bool) -> None:
        """更新按 symbol 隔离的 prefilter 近期通过窗口。"""
        if not self._decision_alignment.get("enabled", False):
            return
        if self._decision_alignment.get("mode") != "prefilter_recent_window":
            return
        if "prefilter" not in self._decision_alignment.get(
            "layers_allow_recent_window", []
        ):
            return

        window = int(self._decision_alignment.get("window_bars", 0))
        if window <= 0:
            return
        key = str(symbol or "")
        if key not in self._prefilter_recent_state:
            self._prefilter_recent_state[key] = deque(maxlen=window)
        self._prefilter_recent_state[key].append(bool(passed))

    def _has_recent_prefilter_pass(self, symbol: str) -> bool:
        """最近 window_bars 内是否有 prefilter 通过。"""
        key = str(symbol or "")
        q = self._prefilter_recent_state.get(key)
        if not q:
            return False
        return any(q)

    def _accumulation_policy(self) -> Dict[str, Any]:
        raw = (self.archetype.execution.raw or {}) if self.archetype else {}
        policy = raw.get("accumulation_policy") or {}
        return dict(policy) if isinstance(policy, dict) else {}

    def _simple_accumulation_policy(self) -> Dict[str, Any]:
        raw = (self.archetype.execution.raw or {}) if self.archetype else {}
        policy = raw.get("simple_accumulation_policy") or {}
        return dict(policy) if isinstance(policy, dict) else {}

    @staticmethod
    def _policy_float(policy: Dict[str, Any], key: str, default: float) -> float:
        try:
            return float(policy.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    def _accumulation_policy_score(
        self, features: Dict[str, Any], policy: Dict[str, Any]
    ) -> Optional[float]:
        feature_name = str(
            policy.get("score_feature") or "abc_macro_regime_score"
        ).strip()
        try:
            score = float(features.get(feature_name))
        except (TypeError, ValueError):
            return None
        return score if score == score else None

    def _accumulation_policy_blocks_deploy(
        self, symbol: str, score: Optional[float], policy: Dict[str, Any]
    ) -> bool:
        if not policy or score is None:
            return False
        key = str(symbol or "")
        deep_max = self._policy_float(policy, "deep_bear_max_score", 2.0)
        bull_min = self._policy_float(
            policy,
            "bull_exposure_min_score",
            self._policy_float(policy, "transition_max_score", 4.0),
        )
        reset_below = self._policy_float(
            policy, "reset_bull_seen_below_score", deep_max
        )
        if score < reset_below:
            self._accumulation_bull_seen_by_symbol[key] = False
        if score >= bull_min:
            self._accumulation_bull_seen_by_symbol[key] = True
        return bool(policy.get("stop_deploy_after_bull_exposure", True)) and bool(
            self._accumulation_bull_seen_by_symbol.get(key, False)
        )

    def _allows_transition_accumulation(
        self, symbol: str, score: Optional[float], policy: Dict[str, Any]
    ) -> bool:
        if not policy or score is None:
            return False
        if not bool(policy.get("allow_transition_deploy", False)):
            return False
        if bool(self._accumulation_bull_seen_by_symbol.get(str(symbol or ""), False)):
            return False
        deep_max = self._policy_float(policy, "deep_bear_max_score", 2.0)
        transition_max = self._policy_float(policy, "transition_max_score", 4.0)
        return deep_max <= score < transition_max

    @staticmethod
    def _merge_features_for_decide(
        features: Dict[str, Any],
        *,
        features_by_timeframe: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Merge primary features with optional multi-timeframe rows (non-destructive)."""
        merged = dict(features or {})
        if not features_by_timeframe:
            return merged
        for row in features_by_timeframe.values():
            if not isinstance(row, dict):
                continue
            for key, val in row.items():
                if key not in merged or merged.get(key) is None:
                    merged[key] = val
        return merged

    def decide(
        self,
        *,
        features: Dict[str, Any],
        symbol: str,
        bars: Optional[List[Dict[str, Any]]] = None,
        features_by_timeframe: Optional[Dict[str, Dict[str, Any]]] = None,
        decision_time: Any = None,
    ) -> List[TradeIntent]:
        """
        核心决策接口 - 通用策略解析引擎

        决策管线:
          0. Prefilter: 从 prefilter.yaml 检查前置环境条件
          1. Direction: 从 direction.yaml 确定方向
          2. Gate: 从 gate.yaml 进行结构性过滤
          3. Entry Filter: 从 entry_filters.yaml 检查入场时机
          4. Evidence: 可选，从 archetype.evidence 计算综合分 → TradeIntent.confidence（不调仓位倍数）
          5. Execution: 从 execution.yaml 生成参数；TradeIntent.size_multiplier 仅来自此处（含 regime_execution）
        """
        if not features and not features_by_timeframe:
            self._last_funnel = {}
            record_fer_entry_eval(
                strategy=self.strategy_name,
                symbol=symbol,
                signal_ts=None,
                outcome="empty_features",
                funnel={},
                features={},
            )
            return []

        features = self._merge_features_for_decide(
            features, features_by_timeframe=features_by_timeframe
        )

        book = getattr(self, "_spot_cycle_book", None)
        if book is not None:
            cycle_block = book.note_and_buy_blocked(
                symbol,
                features.get("weekly_ema_200_position"),
                now=(
                    decision_time
                    if decision_time is not None
                    else features.get("timestamp")
                ),
            )
            if cycle_block:
                funnel = {
                    "prefilter": False,
                    "prefilter_reason": cycle_block,
                    "spot_cycle_closed": True,
                }
                self._last_funnel = funnel
                record_fer_entry_eval(
                    strategy=self.strategy_name,
                    symbol=symbol,
                    signal_ts=(
                        decision_time
                        if decision_time is not None
                        else features.get("timestamp")
                    ),
                    outcome=cycle_block,
                    funnel=funnel,
                    features=features,
                )
                return []

        # 漏斗跟踪 (bool 标记 + 丰富元数据)
        funnel: Dict[str, Any] = {}
        _sig_ts = (
            decision_time if decision_time is not None else features.get("timestamp")
        )
        simple_policy = self._simple_accumulation_policy()
        use_simple_accum = bool(simple_policy.get("enabled", False))
        # spot_accum_simple: buy gate lives in prefilter.yaml when rules are present.
        if use_simple_accum and self.archetype and self.archetype.prefilter.rules:
            from src.time_series_model.live.spot_accum_simple import (
                is_spot_accum_archetype,
            )

            if is_spot_accum_archetype(self.strategy_name):
                use_simple_accum = False
        accumulation_policy = self._accumulation_policy()
        accumulation_score = self._accumulation_policy_score(
            features, accumulation_policy
        )
        if use_simple_accum:
            from src.time_series_model.live.spot_accum_simple import (
                deep_bear_allows_buy,
            )

            ok_buy, wk_pos = deep_bear_allows_buy(features, simple_policy)
            funnel["simple_deep_bear"] = ok_buy
            funnel["weekly_ema_200_position"] = wk_pos
            if not ok_buy:
                self._last_funnel = funnel
                record_fer_entry_eval(
                    strategy=self.strategy_name,
                    symbol=symbol,
                    signal_ts=_sig_ts,
                    outcome="simple_not_deep_bear",
                    funnel=funnel,
                    features=features,
                )
                return []
        elif self._accumulation_policy_blocks_deploy(
            symbol, accumulation_score, accumulation_policy
        ):
            funnel["accumulation_policy"] = "bull_exposure_stop_deploy"
            funnel["accumulation_score"] = accumulation_score
            self._last_funnel = funnel
            record_fer_entry_eval(
                strategy=self.strategy_name,
                symbol=symbol,
                signal_ts=_sig_ts,
                outcome="accumulation_policy_bull_seen",
                funnel=funnel,
                features=features,
            )
            return []

        # ── 0a. Regime check (慢变量数据空间: EMA 带 / chop 上限 / box 状态) ──
        # Regime 与 Prefilter 解耦：Regime 是 A/B/C 共用慢变量层，
        # Prefilter 是策略 archetype 入场形态。
        regime_label: str = "neutral"
        if self.archetype and not self.archetype.regime.is_empty:
            rg_passed, rg_reason = self.archetype.regime.evaluate(features)
            funnel["regime"] = rg_passed
            # 分类到具体 regime 标签 (bull/bear/neutral)，供 execution 层引用
            regime_label = self.archetype.regime.classify_or_default(
                features, "neutral"
            )
            funnel["regime_label"] = regime_label
            if not rg_passed:
                logger.debug(f"❌ Regime denied: {rg_reason}")
                funnel["regime_reason"] = rg_reason
                self._last_funnel = funnel
                record_fer_entry_eval(
                    strategy=self.strategy_name,
                    symbol=symbol,
                    signal_ts=_sig_ts,
                    outcome="regime_deny",
                    funnel=funnel,
                    features=features,
                )
                return []
            if not self.archetype.regime.allows_classified_label(regime_label):
                logger.debug("❌ Regime denied: reject_neutral label=%s", regime_label)
                funnel["regime"] = False
                funnel["regime_reason"] = "regime_reject_neutral"
                self._last_funnel = funnel
                record_fer_entry_eval(
                    strategy=self.strategy_name,
                    symbol=symbol,
                    signal_ts=_sig_ts,
                    outcome="regime_deny",
                    funnel=funnel,
                    features=features,
                )
                return []
            logger.debug("✅ Regime passed")

        # ── 0. Prefilter 前置条件检查 ──
        if self.archetype and self.archetype.prefilter.rules:
            pf_passed, pf_reason = self.archetype.prefilter.evaluate(
                features, latch_id=symbol
            )
            funnel["prefilter"] = pf_passed
            self._update_prefilter_recent_state(symbol, pf_passed)
            if not pf_passed:
                alignment_used = self._has_recent_prefilter_pass(symbol)
                accumulation_override = self._allows_transition_accumulation(
                    symbol, accumulation_score, accumulation_policy
                )
                funnel["prefilter_recent_pass"] = alignment_used
                funnel["alignment_used"] = alignment_used or accumulation_override
                funnel["accumulation_transition_override"] = accumulation_override
                if accumulation_override:
                    funnel["accumulation_score"] = accumulation_score
                logger.debug(f"❌ Prefilter denied: {pf_reason}")
                if alignment_used:
                    logger.debug(
                        "↪️ Alignment override: recent prefilter pass within last %d bars",
                        int(self._decision_alignment.get("window_bars", 0)),
                    )
                    funnel["prefilter_alignment_override"] = True
                else:
                    funnel["prefilter_alignment_override"] = False
                funnel["prefilter_reason"] = pf_reason
                if not (alignment_used or accumulation_override):
                    self._last_funnel = funnel
                    record_fer_entry_eval(
                        strategy=self.strategy_name,
                        symbol=symbol,
                        signal_ts=_sig_ts,
                        outcome="prefilter_deny",
                        funnel=funnel,
                        features=features,
                    )
                    return []
            logger.debug("✅ Prefilter passed")

        # ── 1. 方向判定 ──
        if self.direction_evaluator is None:
            logger.error("❌ Direction evaluator not initialized")
            self._last_funnel = funnel
            record_fer_entry_eval(
                strategy=self.strategy_name,
                symbol=symbol,
                signal_ts=_sig_ts,
                outcome="no_direction_config",
                funnel=funnel,
                features=features,
            )
            return []

        direction, rule_id = self.direction_evaluator.evaluate(features, symbol=symbol)
        funnel["direction"] = direction != 0
        funnel["direction_value"] = direction  # 1=long, -1=short, 0=none
        funnel["direction_rule"] = rule_id
        if direction == 0:
            logger.debug("❌ No valid direction found")
            funnel["direction_reason"] = "no_direction_rule_matched"
            self._last_funnel = funnel
            record_fer_entry_eval(
                strategy=self.strategy_name,
                symbol=symbol,
                signal_ts=_sig_ts,
                outcome="no_direction",
                funnel=funnel,
                features=features,
            )
            return []

        # Regime allowed_sides + optional per-bar side_mask (EMA position / slope)
        if self.archetype and not self.archetype.regime.allows_side_for_bar(
            direction, features
        ):
            funnel["regime_side_block"] = True
            funnel["regime_side_attempted"] = "long" if direction > 0 else "short"
            funnel["regime_allowed_sides"] = list(self.archetype.regime.allowed_sides)
            funnel["direction_reason"] = "regime_disallows_side"
            self._last_funnel = funnel
            record_fer_entry_eval(
                strategy=self.strategy_name,
                symbol=symbol,
                signal_ts=_sig_ts,
                outcome="regime_side_deny",
                funnel=funnel,
                features=features,
            )
            return []

        side_str = "BUY" if direction == 1 else "SELL"
        logger.debug(f"🎯 Direction: {side_str} (rule: {rule_id})")

        # ── 2. Gate 过滤 ──
        gate_weight = 0.0
        if self.gate_evaluator is not None:
            gate_passed, gate_reasons, gate_weight = self.gate_evaluator.evaluate(
                features, self._quantiles
            )
            if not gate_passed:
                logger.debug(f"❌ Gate denied: {gate_reasons}")
                funnel["gate"] = False
                funnel["gate_reasons"] = gate_reasons  # 拦截原因列表
                self._last_funnel = funnel
                record_fer_entry_eval(
                    strategy=self.strategy_name,
                    symbol=symbol,
                    signal_ts=_sig_ts,
                    outcome="gate_deny",
                    funnel=funnel,
                    features=features,
                )
                return []
            funnel["gate"] = True
            funnel["gate_weight"] = round(gate_weight, 4)
            logger.debug(f"✅ Gate passed (weight: {gate_weight:.3f})")

        # ── 3. Entry Filter 检查 ──
        if self.entry_filter_checker is not None:
            ef_passed = self.entry_filter_checker.check(features, direction=direction)
            if not ef_passed:
                logger.debug("❌ Entry filter denied")
                funnel["entry_filter"] = False
                funnel["entry_filter_reason"] = self.entry_filter_checker.explain(
                    features
                )
                self._last_funnel = funnel
                record_fer_entry_eval(
                    strategy=self.strategy_name,
                    symbol=symbol,
                    signal_ts=_sig_ts,
                    outcome="entry_filter_deny",
                    funnel=funnel,
                    features=features,
                )
                return []
            funnel["entry_filter"] = True
            logger.debug("✅ Entry filter passed")

        # ── 4. Evidence 评分 ──
        evidence_score = 0.5
        evidence_active = False
        evidence_breakdown = {}
        if (
            self.archetype
            and self.archetype.evidence
            and self.archetype.evidence.features
        ):
            feature_values = {
                feat.feature: features.get(feat.feature)
                for feat in self.archetype.evidence.features
                if features.get(feat.feature) is not None
            }
            if feature_values:
                evidence_active = True
                evidence_score, evidence_breakdown = (
                    self.archetype.evidence.compute_composite_score(
                        feature_values, self._quantiles
                    )
                )
                logger.debug(
                    f"📊 Composite confidence: score={evidence_score:.3f}, "
                    f"breakdown={evidence_breakdown}"
                )

        # ── 5. 执行参数生成 ──
        exec_params = {}
        if self.execution_generator is not None:
            exec_params = self.execution_generator.generate_params(
                evidence_score,
                features=features,
                direction=direction,
                regime_label=regime_label,
            )
            self._last_tier_params = exec_params
            logger.debug(f"⚙️  Execution params: {exec_params}")

        action = "LONG" if direction == 1 else "SHORT"
        _srb_true_sr: Optional[float] = None
        _srb_opposite_sr: Optional[float] = None

        if str(self.strategy_name).lower() == "srb":
            funnel["srb_regime_bucket"] = features.get("srb_regime_bucket")
            funnel["srb_regime_adx14"] = features.get("srb_regime_adx14")
            funnel["srb_regime_er20"] = features.get("srb_regime_er20")
            funnel["srb_sr_support"] = features.get("srb_sr_support")
            funnel["srb_sr_resistance"] = features.get("srb_sr_resistance")
            funnel["wide_sr_upper_px"] = features.get("wide_sr_upper_px")
            funnel["wide_sr_lower_px"] = features.get("wide_sr_lower_px")

            _raw_ex = (self.archetype.execution.raw or {}) if self.archetype else {}
            _wg = _raw_ex.get("sr_wide_entry_guard") or {}
            if _wg.get("enabled"):
                _mn = float(_wg.get("min_distance_atr", 0) or 0)
                _cl = float(features.get("close") or 0)
                _at = float(features.get("atr") or 0)
                if should_reject_srb_wide_entry(
                    action,
                    _cl,
                    _at,
                    features.get("wide_sr_lower_px"),
                    features.get("wide_sr_upper_px"),
                    _mn,
                ):
                    funnel["reject_srb_wide_sr_too_close"] = True
                    self._last_funnel = funnel
                    record_fer_entry_eval(
                        strategy=self.strategy_name,
                        symbol=symbol,
                        signal_ts=_sig_ts,
                        outcome="srb_wide_sr_guard",
                        funnel=funnel,
                        features=features,
                    )
                    return []

            _tsl_cfg = _raw_ex.get("true_sr_level") or {}
            _fb_atr = float(_tsl_cfg.get("wide_fallback_atr", 0) or 0)
            _prefer = _tsl_cfg.get("prefer")
            _srb_true_sr = pick_srb_true_sr_level(
                action,
                float(features.get("close") or 0),
                float(features.get("atr") or 0),
                narrow_support=features.get("srb_sr_support"),
                narrow_resistance=features.get("srb_sr_resistance"),
                wide_lower_px=features.get("wide_sr_lower_px"),
                wide_upper_px=features.get("wide_sr_upper_px"),
                fallback_atr=_fb_atr,
                prefer=_prefer,
            )
            # SRB 结构化 SL 锚点：LONG 用对面 support，SHORT 用对面 resistance
            # opposite_sr_source: L1（默认）| L3（wide_sr）
            _ssl_cfg = (
                (_raw_ex.get("stop_loss") or {}).get("structural_sl") or {}
            ) or (_raw_ex.get("structural_sl") or {})
            _opp_src = str(_ssl_cfg.get("opposite_sr_source") or "L1")
            _srb_opposite_sr = resolve_srb_opposite_sr_level(
                action,
                narrow_support=features.get("srb_sr_support"),
                narrow_resistance=features.get("srb_sr_resistance"),
                wide_lower_px=features.get("wide_sr_lower_px"),
                wide_upper_px=features.get("wide_sr_upper_px"),
                source=_opp_src,
            )

        # ── 6. 构建 TradeIntent ──
        # 仓位倍数仅来自 execution（含 regime_execution 补丁）；evidence_score 不写进 size_multiplier
        _exec_sm = float(exec_params.get("size_multiplier", 1.0) or 1.0)
        intent = TradeIntent(
            action=action,
            symbol=symbol,
            archetype=self.strategy_name,
            execution_strategy=self.strategy_name,
            confidence=evidence_score,
            size_multiplier=_exec_sm,
            execution_tags=[self.strategy_name, side_str],
            execution_profile={
                "rr_constraints": rr_constraints_from_exec_params(exec_params),
                "strategy_specific": {
                    "direction_rule": rule_id,
                    "gate_weight": gate_weight,
                    **(
                        {"srb_true_sr_level": float(_srb_true_sr)}
                        if _srb_true_sr is not None
                        else {}
                    ),
                    **(
                        {"srb_opposite_sr_level": float(_srb_opposite_sr)}
                        if _srb_opposite_sr is not None
                        else {}
                    ),
                },
                "add_position": (
                    (self.archetype.execution.raw or {}).get("add_position") or {}
                ),
                "execution_constraints": dict(
                    (self.archetype.execution.raw or {}).get("execution_constraints")
                    or {}
                ),
                # Spot per-leg sizing is configured at execution YAML root in
                # both research and live. Keep it on the intent so event_backtest
                # applies the same inventory-depth decay as run_spot_accum_live.
                "deploy_decay": dict(
                    (self.archetype.execution.raw or {}).get("deploy_decay") or {}
                ),
            },
        )

        if evidence_active:
            logger.info(
                f"✅ Signal generated: {action} {symbol} "
                f"(confidence={evidence_score:.3f})"
            )
        else:
            logger.info(
                f"✅ Signal generated: {action} {symbol} (confidence=inactive→neutral)"
            )
        self._last_funnel = funnel
        record_fer_entry_eval(
            strategy=self.strategy_name,
            symbol=symbol,
            signal_ts=_sig_ts,
            outcome="signal",
            funnel=funnel,
            features=features,
        )
        return [intent]

    # ── 兼容属性: 脚本通过 strat._archetype 访问 archetype ──

    @property
    def _archetype(self):
        """兼容属性: 脚本通过 strat._archetype 访问"""
        return self.archetype

    # ── 诊断接口: _evaluate_entry_signal ──

    def _evaluate_entry_signal(
        self,
        features: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        入场信号评估（诊断用）

        管线:
          1. direction.yaml 规则 → 方向
          2. Gate → 结构性否决
          3. Entry Filter → 入场时机
          4. Evidence → Tier 选择
          5. 返回 (should_enter, signal_info)
        """
        if not features:
            return False, {}

        # ── 1. 方向判定 ──
        if self.direction_evaluator is None:
            return False, {"reject_reason": "no_direction_config"}

        direction, rule_id = self.direction_evaluator.evaluate(features)
        if direction == 0:
            return False, {"reject_reason": "no_direction"}

        side_str = "BUY" if direction == 1 else "SELL"

        # ── 2. Gate 检查 ──
        gate_weight = 1.0
        if self.gate_evaluator is not None:
            gate_passed, gate_reasons, gate_weight = self.gate_evaluator.evaluate(
                features, self._quantiles
            )
            if not gate_passed:
                return False, {
                    "reject_reason": "gate_deny",
                    "gate_reasons": gate_reasons,
                }

        # ── 3. Entry Filter 检查 ──
        if self.entry_filter_checker is not None:
            ef_passed = self.entry_filter_checker.check(features, direction=direction)
            if not ef_passed:
                return False, {"reject_reason": "entry_filter_deny"}

        # ── 4. Evidence Score ──
        evidence_score = 0.5
        adjusted_score = 0.5
        evidence_breakdown = {}
        if (
            self.archetype
            and self.archetype.evidence
            and self.archetype.evidence.features
        ):
            feature_values = {
                feat.feature: features.get(feat.feature)
                for feat in self.archetype.evidence.features
                if features.get(feat.feature) is not None
            }
            if feature_values:
                adjusted_score, evidence_breakdown = (
                    self.archetype.evidence.compute_composite_score(
                        feature_values, self._quantiles
                    )
                )
                evidence_score = adjusted_score

        # Tier 选择
        exec_params = {}
        if self.execution_generator is not None:
            exec_params = self.execution_generator.generate_params(
                adjusted_score,
                features=features,
                direction=direction,
                regime_label="neutral",
            )
            self._last_tier_params = exec_params

        # ── 5. 构建 signal_info ──
        signal_info = {
            "side": side_str,
            "direction": direction,
            "reason": (
                f"{self.strategy_name.upper()}_{side_str} "
                f"(gate_w={gate_weight:.2f})"
            ),
            "confidence": adjusted_score,
            "evidence_breakdown": evidence_breakdown,
            "gate_weight": gate_weight,
            "execution_params": exec_params,
            "atr": features.get("atr", 0.0),
        }

        return True, signal_info

    def reset(self) -> None:
        """重置状态"""
        if self.entry_filter_checker and self.entry_filter_checker.ef_state:
            self.entry_filter_checker.ef_state.reset()
        self._last_tier_params = None
        self._prefilter_recent_state.clear()
