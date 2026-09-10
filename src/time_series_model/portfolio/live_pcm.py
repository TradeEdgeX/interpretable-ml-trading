"""LivePCM — Live Portfolio Control Manager (Regime-Aware, v3)

多 archetype 信号仲裁层，硬约束从 constitution.yaml 读取。

控制框架 (v3 职责分工):
  constitution.yaml:  硬约束上限 (slot_count, risk_per_slot, per_strategy_limits)
  pcm_regime.yaml:    仲裁策略 (优先级, Regime 检测, 仓位缩放)

  Layer 1: 硬约束 — slot/risk 从 constitution 读取，不可突破
  Layer 2: Regime 感知 — 动态优先级 + 仓位缩放
     - NORMAL:        LV > FER > ME > BPC  (全仓)
     - HIGH_VOL:      LV > ME > FER > BPC  (缩仓 50%)
     - HIGH_LEVERAGE:  LV > FER > ME > BPC  (缩仓 70%)

优先级依据: 信号条件严格性（越严格越优先）
  LV (liquidation cluster) > FER (均衡偏离反转) > ME (动能扩张) > BPC (趋势延续)

单策略时行为等价于直接挂 GenericLiveStrategy（零额外开销）。
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

import yaml

from src.live_data_stream.constitution_config import (
    classic_slot_policy_from_constitution,
    intent_archetype_priority_tokens,
    load_constitution_dict,
)
from src.time_series_model.core.trade_intent import TradeIntent

logger = logging.getLogger(__name__)


def _calendar_day_utc_str(
    *, features: Dict[str, Any], decision_time: Any = None
) -> str:
    """PCM 日内节流用的日历日 (UTC YYYY-MM-DD)。

    优先 ``decision_time``，其次 ``features["timestamp"]``；缺失时回退为当前 UTC 日（实盘）。
    """
    cand = decision_time if decision_time is not None else features.get("timestamp")
    if cand is None:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        if hasattr(cand, "date") and callable(getattr(cand, "date")):
            d = cand.date()
            if isinstance(d, date):
                return d.strftime("%Y-%m-%d")
    except Exception:
        pass
    try:
        if isinstance(cand, str) and len(cand) >= 10:
            return cand[:10]
    except Exception:
        pass
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _trend_side_token(raw: Any) -> str:
    side = str(raw or "").strip().lower()
    if side in {"long", "buy"}:
        return "long"
    if side in {"short", "sell"}:
        return "short"
    return ""


# ────────────────────────────────────────────────────
# Strategy 接口协议（duck typing）
# ────────────────────────────────────────────────────


@runtime_checkable
class DecisionHandler(Protocol):
    """任何实现 decide(features, symbol) → List[TradeIntent] 的对象"""

    def decide(
        self,
        *,
        features: Dict[str, Any],
        symbol: str,
        bars: Optional[List[Dict[str, Any]]] = None,
    ) -> List[TradeIntent]:  # noqa: E704
        ...


# ────────────────────────────────────────────────────
# Regime Detector
# ────────────────────────────────────────────────────

# 3 个 Regime
REGIME_NORMAL = "NORMAL"
REGIME_HIGH_VOL = "HIGH_VOL"
REGIME_HIGH_LEVERAGE = "HIGH_LEVERAGE"

# 每个 Regime 的默认优先级（可被 YAML 覆盖）
# 决策依据: 信号条件严格性（越严格越优先）
DEFAULT_REGIME_PRIORITIES = {
    REGIME_NORMAL: ["LV", "FER", "ME", "BPC"],
    REGIME_HIGH_VOL: ["LV", "ME", "FER", "BPC"],
    REGIME_HIGH_LEVERAGE: ["LV", "FER", "ME", "BPC"],
}

# 默认检测阈值
DEFAULT_DETECTION = {
    REGIME_HIGH_LEVERAGE: {
        "conditions": [
            {"feature": "oi_zscore", "operator": ">", "threshold": 1.5},
            {"feature": "funding_rate_abs_zscore", "operator": ">", "threshold": 2.0},
        ],
        "logic": "AND",
    },
    REGIME_HIGH_VOL: {
        "conditions": [
            {"feature": "atr_percentile", "operator": ">", "threshold": 0.7},
        ],
        "logic": "AND",
    },
}


class RegimeDetector:
    """极简 Regime 状态机 (< 40 行核心逻辑)

    检测顺序: HIGH_LEVERAGE → HIGH_VOL → NORMAL (严格条件优先)
    防抖: min_bars_in_regime 根 bar 内不允许切换
    仓位缩放: 每个 regime 带 position_scale + per_archetype_scale
    """

    def __init__(
        self,
        regime_priorities: Optional[Dict[str, List[str]]] = None,
        detection: Optional[Dict[str, Dict]] = None,
        min_bars_in_regime: int = 3,
        regime_scales: Optional[Dict[str, float]] = None,
        regime_archetype_scales: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self._regime_priorities = regime_priorities or dict(DEFAULT_REGIME_PRIORITIES)
        self._detection = detection or dict(DEFAULT_DETECTION)
        self._min_bars = min_bars_in_regime
        # Regime 仓位缩放: {regime_name: scale}
        self._regime_scales: Dict[str, float] = regime_scales or {
            REGIME_NORMAL: 1.0,
            REGIME_HIGH_VOL: 0.5,
            REGIME_HIGH_LEVERAGE: 0.3,
        }
        # Per-archetype 覆盖: {regime_name: {archetype: scale}}
        self._regime_archetype_scales: Dict[str, Dict[str, float]] = (
            regime_archetype_scales or {}
        )

        # 状态
        self._current_regime: str = REGIME_NORMAL
        self._bars_in_current: int = 0

        # 统计
        self._regime_history: List[Tuple[str, str]] = []  # (regime, symbol)
        self._switch_count: int = 0

    @property
    def current_regime(self) -> str:
        return self._current_regime

    @property
    def switch_count(self) -> int:
        return self._switch_count

    @property
    def current_priority(self) -> List[str]:
        return list(
            self._regime_priorities.get(
                self._current_regime,
                DEFAULT_REGIME_PRIORITIES[REGIME_NORMAL],
            )
        )

    def detect(self, features: Dict[str, Any]) -> str:
        """根据特征检测当前 regime，带防抖。

        Returns:
            当前 regime 名称
        """
        raw_regime = self._raw_detect(features)

        self._bars_in_current += 1

        if raw_regime != self._current_regime:
            if self._bars_in_current >= self._min_bars:
                old = self._current_regime
                self._current_regime = raw_regime
                self._bars_in_current = 0
                self._switch_count += 1
                logger.info(
                    "PCM Regime: %s → %s (switch #%d)",
                    old,
                    raw_regime,
                    self._switch_count,
                )
            # else: 防抖中，保持原 regime

        self._regime_history.append((self._current_regime, ""))
        return self._current_regime

    def _raw_detect(self, features: Dict[str, Any]) -> str:
        """无防抖的纯检测逻辑。检测顺序: HIGH_LEVERAGE → HIGH_VOL → NORMAL"""
        # 优先检测 HIGH_LEVERAGE（最严格条件）
        for regime_name in [REGIME_HIGH_LEVERAGE, REGIME_HIGH_VOL]:
            det = self._detection.get(regime_name)
            if det and self._check_conditions(features, det):
                return regime_name
        return REGIME_NORMAL

    @staticmethod
    def _check_conditions(features: Dict[str, Any], det: Dict) -> bool:
        """检查一组条件是否满足"""
        conditions = det.get("conditions", [])
        logic = det.get("logic", "AND").upper()

        results = []
        for cond in conditions:
            feat_name = cond["feature"]
            op = cond["operator"]
            thr = float(cond["threshold"])
            val = features.get(feat_name)

            if val is None:
                results.append(False)
                continue

            try:
                val = float(val)
            except (ValueError, TypeError):
                results.append(False)
                continue

            if op == ">":
                results.append(val > thr)
            elif op == ">=":
                results.append(val >= thr)
            elif op == "<":
                results.append(val < thr)
            elif op == "<=":
                results.append(val <= thr)
            else:
                results.append(False)

        if not results:
            return False
        if logic == "AND":
            return all(results)
        return any(results)  # OR

    @property
    def current_position_scale(self) -> float:
        """当前 Regime 的全局仓位缩放因子"""
        return self._regime_scales.get(self._current_regime, 1.0)

    def get_archetype_scale(self, archetype: str) -> float:
        """获取当前 Regime 下特定 archetype 的仓位缩放因子。

        优先级: per_archetype_scale > regime 全局 scale > 1.0
        """
        per_arch = self._regime_archetype_scales.get(self._current_regime, {})
        if archetype.upper() in per_arch:
            return per_arch[archetype.upper()]
        return self.current_position_scale

    def reset(self) -> None:
        """重置状态（用于回测分段）"""
        self._current_regime = REGIME_NORMAL
        self._bars_in_current = 0
        self._switch_count = 0
        self._regime_history.clear()


def load_regime_config(
    config_path: str = "config/pcm_regime.yaml",
) -> Dict[str, Any]:
    """加载 PCM regime 配置文件"""
    p = Path(config_path)
    if not p.exists():
        logger.warning("PCM regime config not found: %s, using defaults", p)
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_regime_detector(cfg: Dict[str, Any]) -> RegimeDetector:
    """从配置字典创建 RegimeDetector（内部辅助）"""
    if not cfg:
        return RegimeDetector()

    priorities = {}
    regime_scales: Dict[str, float] = {}
    regime_archetype_scales: Dict[str, Dict[str, float]] = {}
    for regime_name, regime_cfg in cfg.get("regimes", {}).items():
        if "priority" in regime_cfg:
            priorities[regime_name] = regime_cfg["priority"]
        if "position_scale" in regime_cfg:
            regime_scales[regime_name] = float(regime_cfg["position_scale"])
        per_arch = regime_cfg.get("per_archetype_scale")
        if per_arch and isinstance(per_arch, dict):
            regime_archetype_scales[regime_name] = {
                k.upper(): float(v) for k, v in per_arch.items()
            }

    detection = cfg.get("detection", {})
    min_bars = cfg.get("min_bars_in_regime", 3)

    return RegimeDetector(
        regime_priorities=priorities or None,
        detection=detection or None,
        min_bars_in_regime=min_bars,
        regime_scales=regime_scales or None,
        regime_archetype_scales=regime_archetype_scales or None,
    )


def create_regime_detector_from_config(
    config_path: str = "config/pcm_regime.yaml",
) -> RegimeDetector:
    """从 YAML 配置创建 RegimeDetector"""
    cfg = load_regime_config(config_path)
    return _build_regime_detector(cfg)


# ────────────────────────────────────────────────────
# LivePCM
# ────────────────────────────────────────────────────

# 默认优先级（按信号条件严格性排序）
DEFAULT_ARCHETYPE_PRIORITY = ["LV", "FER", "ME", "BPC"]


def _load_constitution_constraints(
    constitution_yaml: Optional[str],
) -> Dict[str, Any]:
    """Load hard constraints from constitution.yaml.

    Returns dict with keys: slot_count, risk_per_slot, per_strategy_limits,
    add_position_rules.
    Falls back to safe defaults if file not found.

    Uses ``load_constitution_dict`` (extends-aware deep merge) — the same loader
    ``run_live.py`` uses for ``enabled_archetypes`` — so a child constitution that
    only overrides e.g. ``kill_switch`` or ``trend_preempt`` still inherits
    ``slot_count`` / ``trend_pool_guard`` / ``account_risk_limits`` etc. from its
    ``extends:`` parent instead of silently losing them (2026-08-04 fix: a bare
    ``yaml.safe_load`` here ignored ``extends`` entirely, which is how the
    2026-07-29 fade preempt recheck ended up running with slot_count=2 and
    trend_pool_guard disabled instead of the intended prod pool settings).
    """
    defaults = {
        "slot_count": 2,
        "risk_per_slot": 0.01,
        "per_strategy_limits": {},
        "add_position_rules": {},
        "intent_selection_policy": {},
        "direction_policy": {},
        "slot_policy": {},
        "resource_allocation": {},
        "account_risk_limits": {},
    }
    if not constitution_yaml:
        return defaults
    p = Path(constitution_yaml)
    if not p.exists():
        logger.warning("Constitution YAML not found: %s, using defaults", p)
        return defaults
    try:
        obj = load_constitution_dict(str(p)) or {}
    except ValueError as e:
        # extends cycle or other structural error — fail loud via defaults + warning
        # rather than silently trading with an unresolved/partial constitution.
        logger.warning(
            "Failed to resolve constitution extends for %s: %s, using defaults", p, e
        )
        return defaults
    except Exception as e:
        logger.warning("Failed to load constitution YAML: %s, using defaults", e)
        return defaults

    slots = obj.get("slots") or {}
    ra = obj.get("resource_allocation") or {}
    spot = obj.get("spot") or {}
    add_rules = (
        ra.get("add_position_rules")
        or ra.get("add_position")
        or obj.get("add_position")
        or {}
    )
    slot_policy = classic_slot_policy_from_constitution(obj)
    per_strategy_limits = dict(ra.get("per_strategy_limits") or {})
    if isinstance(spot, dict):
        sl = spot.get("strategy_limits") or {}
        if isinstance(sl, dict):
            per_strategy_limits.update(dict(sl))
    return {
        "slot_count": int(slots.get("slot_count", 2)),
        "risk_per_slot": float(slots.get("risk_per_slot", 0.01)),
        "per_strategy_limits": per_strategy_limits,
        "add_position_rules": dict(add_rules),
        "intent_selection_policy": dict(ra.get("intent_selection_policy") or {}),
        "direction_policy": dict(ra.get("direction_policy") or {}),
        "slot_policy": slot_policy,
        "account_risk_limits": dict(ra.get("account_risk_limits") or {}),
        "evidence_min_score": float(ra.get("evidence_min_score", 0.0)),
        "evidence_position_scale": bool(ra.get("evidence_position_scale", True)),
        # Full RA for intent_archetype_priority_tokens (enabled_archetypes order fallback)
        "resource_allocation": dict(ra),
    }


class LivePCM:
    """
    Live Portfolio Control Manager (Regime-Aware, v3)

    职责:
      1. 注册多个策略（BPC, ME, FER, LV），每个策略实现 decide() 接口
      2. Regime 检测 → 动态优先级 + 仓位缩放
      3. 同 symbol 不同 archetype 同时触发 → 当前 regime 优先级选最高
      4. 同优先级比 Evidence Score（高的优先）
      5. 跨 symbol slot 控制（从 constitution 读 max_slots）
      6. Regime 仓位缩放 → 乘到 intent.size_multiplier（来自 execution / regime_execution）
      7. Evidence 入场门槛（score < min_score → 拒绝开仓）；evidence 不参与倍数缩放

    配置来源:
        constitution.yaml:  slot_count, risk_per_slot, per_strategy_limits
        pcm_regime.yaml:    regimes (priority + position_scale), detection
    """

    def __init__(
        self,
        archetype_priority: Optional[List[str]] = None,
        max_slots: Optional[int] = None,
        get_open_slot_count: Optional[callable] = None,
        get_open_trend_positions: Optional[callable] = None,
        regime_detector: Optional[RegimeDetector] = None,
        regime_config_path: Optional[str] = None,
        override_config: Optional[Dict[str, Any]] = None,
        constitution_yaml: Optional[str] = None,
        float_ladder_archetypes: Optional[Iterable[str]] = None,
    ):
        """
        Args:
            archetype_priority: archetype 静态优先级列表。如有 regime_detector 则被忽略。
            max_slots: 显式指定 max_slots（覆盖 constitution）。
                未提供时从 constitution_yaml 读取，均未提供时默认 2。
            get_open_slot_count: 可选回调，返回当前已占用 slot 数
            regime_detector: 可选 RegimeDetector 实例。
            regime_config_path: 可选 pcm_regime.yaml 路径。
            override_config: Layer 3 Override 配置。显式传入优先于 YAML。
            constitution_yaml: 可选 constitution.yaml 路径。
                提供后从中读取 slot_count、risk_per_slot、per_strategy_limits。
                未提供时尝试从 pcm_regime.yaml 的 constitution_ref 自动发现。
        """
        self._strategies: Dict[str, DecisionHandler] = {}
        self._strategy_timeframes: Dict[str, str] = {}  # archetype → timeframe
        self._strategy_symbol_sets: Dict[str, frozenset[str]] = {}
        self._get_open_slot_count = get_open_slot_count
        self._get_open_trend_positions = get_open_trend_positions

        # Slot 追踪: key = "{symbol}:{archetype}", value = True
        self._slot_evidence: Dict[str, float] = {}
        self._latest_features: Dict[str, Any] = {}
        self._dir_state: Dict[str, Dict[str, Any]] = {}
        self._float_ladder_archetypes = frozenset(
            str(a).strip().lower()
            for a in (float_ladder_archetypes or [])
            if str(a or "").strip()
        )

        # Layer 3: Override 配置
        self._override_config: Dict[str, Any] = override_config or {}

        # 加载 regime 配置
        self._regime_cfg: Dict[str, Any] = {}
        if regime_detector is not None:
            self._regime_detector = regime_detector
        elif regime_config_path is not None:
            self._regime_cfg = load_regime_config(regime_config_path)
            # 若未显式配置 regimes/detection，则视为关闭 regime 缩放与切换
            # （避免误用默认阈值和默认缩放）
            _has_regime_logic = bool(
                dict(self._regime_cfg.get("regimes") or {})
                or dict(self._regime_cfg.get("detection") or {})
            )
            if _has_regime_logic:
                self._regime_detector = _build_regime_detector(self._regime_cfg)
            else:
                self._regime_detector = None
                logger.info(
                    "PCM: regime detector disabled (no regimes/detection in %s)",
                    regime_config_path,
                )
            if not self._override_config:
                self._override_config = self._regime_cfg.get("override", {})
        else:
            self._regime_detector = None

        # 从 constitution 加载硬约束
        _const_yaml = constitution_yaml
        if not _const_yaml and self._regime_cfg:
            _const_yaml = self._regime_cfg.get("constitution_ref")
        self._constitution = _load_constitution_constraints(_const_yaml)

        # max_slots: 显式参数 > constitution > 默认 2
        if max_slots is not None:
            self._max_slots = max_slots
        else:
            self._max_slots = self._constitution["slot_count"]
        logger.info(
            "PCM: max_slots=%d (source=%s)",
            self._max_slots,
            "explicit" if max_slots is not None else "constitution",
        )

        # 静态优先级（当无 regime detector 时使用）
        self._archetype_priority = archetype_priority or list(
            DEFAULT_ARCHETYPE_PRIORITY
        )

        # ── EMA 方向过滤器 (price > ema_200 → bull; price ≤ ema_200 → bear) ──
        _ema_cfg = self._regime_cfg.get("ema_direction_filter", {})
        self._ema_filter_enabled: bool = bool(_ema_cfg.get("enabled", False))
        self._ema_close_feature: str = _ema_cfg.get("close_feature", "close")
        self._ema_feature: str = _ema_cfg.get("ema_feature", "ema_200")
        self._ema_bull_allowed: set = set(_ema_cfg.get("bull_allowed", []))
        self._ema_bear_allowed: set = set(_ema_cfg.get("bear_allowed", []))
        self._ema_fallback: str = _ema_cfg.get("fallback", "allow_all")
        if self._ema_filter_enabled:
            logger.info(
                "PCM: EMA方向过滤器已启用 — bull允许=%s | bear允许=%s | fallback=%s",
                sorted(self._ema_bull_allowed),
                sorted(self._ema_bear_allowed),
                self._ema_fallback,
            )

        # ── 每日入场节流 (max_new_entries_per_day) ──
        self._daily_entry_counts: Dict[tuple, int] = (
            {}
        )  # (family, symbol, date_str) -> count
        self._daily_entry_limits: Dict[str, Optional[int]] = {}
        _psl = self._constitution.get("per_strategy_limits") or {}
        for _fam, _cfg in _psl.items():
            if isinstance(_cfg, dict) and "max_new_entries_per_day" in _cfg:
                _lim = int(_cfg["max_new_entries_per_day"])
                self._daily_entry_limits[str(_fam).lower().strip()] = _lim
                logger.info("PCM: %s max_new_entries_per_day=%d", _fam, _lim)
        self._slot_policy = dict(self._constitution.get("slot_policy") or {})
        self._max_trend_slots_per_symbol = int(
            self._slot_policy.get("max_trend_slots_per_symbol", 0) or 0
        )
        self._enforce_single_trend_per_symbol = self._max_trend_slots_per_symbol == 1
        self._trend_families = {
            str(x).strip().lower()
            for x in (self._slot_policy.get("trend_archetypes") or [])
            if str(x).strip()
        }
        _trend_pool_guard = self._slot_policy.get("trend_pool_guard") or {}
        if not isinstance(_trend_pool_guard, dict):
            _trend_pool_guard = {}
        self._trend_pool_guard = dict(_trend_pool_guard)
        self._trend_pool_guard_enabled = bool(
            self._trend_pool_guard.get("enabled", False)
        )
        # Manual regime overlay (2026-07-14): bear→1/3, bull→2/4.
        # Set ``active_regime`` in constitution or env ``MLBOT_TREND_POOL_ACTIVE_REGIME``.
        # No auto flip — ops flips by hand when macro regime changes.
        (
            self._trend_pool_active_regime,
            self._trend_pool_max_unprotected,
            self._trend_pool_max_symbols_after_unlock,
        ) = self._resolve_trend_pool_caps(self._trend_pool_guard)
        self._trend_pool_unlock_on = (
            str(
                self._trend_pool_guard.get("unlock_on", "breakeven_locked")
                or "breakeven_locked"
            )
            .strip()
            .lower()
        )
        self._trend_pool_anchor_symbol = (
            str(self._trend_pool_guard.get("anchor_symbol", "") or "").upper().strip()
        )
        self._trend_pool_require_anchor_first = bool(
            self._trend_pool_guard.get("require_anchor_first", False)
        )
        _corr_guard = self._resolve_corr_guard(
            self._trend_pool_guard, self._trend_pool_active_regime
        )
        self._trend_pool_corr_enabled = bool(_corr_guard.get("enabled", False))
        self._trend_pool_corr_threshold = float(_corr_guard.get("threshold", 0.80))
        self._trend_pool_corr_same_direction_only = bool(
            _corr_guard.get("same_direction_only", True)
        )
        try:
            _cluster = int(_corr_guard.get("max_cluster", 1) or 1)
        except (TypeError, ValueError):
            _cluster = 1
        self._trend_pool_corr_max_cluster = max(1, _cluster)
        self._trend_pool_corr_pairs = self._normalize_symbol_correlations(
            _corr_guard.get("pairs") or {}
        )
        # SRB leadership: close lower-priority fadable trend sleeves then open preemptor.
        # Always flatten+reopen (no same-side handoff). Opt-in via slot_policy.trend_preempt.
        _preempt = self._slot_policy.get("trend_preempt") or {}
        if not isinstance(_preempt, dict):
            _preempt = {}
        self._trend_preempt_enabled = bool(_preempt.get("enabled", False))
        self._trend_preemptors = {
            str(x).strip().lower()
            for x in (_preempt.get("preemptor") or _preempt.get("preemptors") or [])
            if str(x).strip()
        }
        self._trend_preemptable = {
            str(x).strip().lower()
            for x in (
                _preempt.get("preemptable")
                or _preempt.get("preemptable_archetypes")
                or []
            )
            if str(x).strip()
        }
        if self._trend_preempt_enabled:
            logger.info(
                "PCM: trend_preempt enabled preemptor=%s preemptable=%s",
                sorted(self._trend_preemptors),
                sorted(self._trend_preemptable),
            )
        if self._trend_pool_guard_enabled and self._trend_pool_max_unprotected > 0:
            logger.info(
                "PCM: trend_pool_guard enabled "
                "(active_regime=%s, max_unprotected_symbols=%d, unlock_on=%s, "
                "max_symbols_after_unlock=%d, anchor_symbol=%s, require_anchor_first=%s, "
                "corr_guard=%s, corr_threshold=%.2f, corr_max_cluster=%d)",
                self._trend_pool_active_regime or "-",
                self._trend_pool_max_unprotected,
                self._trend_pool_unlock_on,
                self._trend_pool_max_symbols_after_unlock,
                self._trend_pool_anchor_symbol,
                self._trend_pool_require_anchor_first,
                self._trend_pool_corr_enabled,
                self._trend_pool_corr_threshold,
                self._trend_pool_corr_max_cluster,
            )

        # 可选: 监控统计收集器
        self.stats_collector = None  # 通过外部注入 StatsCollector 实例

        # 最近一次 decide() 的候选 intent 与 PCM 层拒因（供事件回测漏斗 / 排障）
        self._last_decide_trace: Dict[str, Any] = {}

    def _parse_family_and_side(
        self,
        archetype: str,
        action: Optional[str] = None,
    ) -> tuple[str, str]:
        a = str(archetype or "").lower().strip()
        parts = [p for p in a.split("-") if p]
        # strip trailing timeframe token: 60t / 240t
        if parts and parts[-1].endswith("t") and parts[-1][:-1].isdigit():
            parts = parts[:-1]
        # handle "<family>-<long|short>(-...)" naming
        if len(parts) >= 2 and parts[1] in {"long", "short"}:
            return parts[0], parts[1]
        act = str(action or "").upper().strip()
        if act == "LONG":
            return a, "long"
        if act == "SHORT":
            return a, "short"
        return a, "unknown"

    def _limit_cfg_for_archetype(self, archetype: str) -> Dict[str, Any]:
        limits = dict(self._constitution.get("per_strategy_limits") or {})
        key = str(archetype or "").lower().strip()
        parts = [p for p in key.split("-") if p]
        cands: list[str] = [key]

        # strip trailing timeframe token
        if parts and parts[-1].endswith("t") and parts[-1][:-1].isdigit():
            cands.append("-".join(parts[:-1]))
            parts = parts[:-1]

        # family-direction key
        if len(parts) >= 2 and parts[1] in {"long", "short"}:
            cands.append("-".join(parts[:2]))

        # family only
        if parts:
            cands.append(parts[0])

        seen: set[str] = set()
        for k in cands:
            kk = str(k).lower().strip()
            if not kk or kk in seen:
                continue
            seen.add(kk)
            cand = limits.get(kk) or {}
            if isinstance(cand, dict) and cand:
                return cand
        return {}

    def _family_token(self, archetype: str) -> str:
        key = str(archetype or "").lower().strip()
        if not key:
            return ""
        return key.split("-", 1)[0]

    def _is_trend_archetype(self, archetype: str) -> bool:
        fam = self._family_token(archetype)
        return bool(fam and fam in self._trend_families)

    def _has_other_trend_slot_on_symbol(self, symbol: str, archetype: str) -> bool:
        sym = str(symbol or "").upper().strip()
        if not sym:
            return False
        target_fam = self._family_token(archetype)
        target_arch = str(archetype or "").lower().strip()
        for key in self._slot_evidence:
            if not key.startswith(f"{sym}:"):
                continue
            try:
                _, existing_arch = key.split(":", 1)
            except Exception:
                continue
            if not self._is_trend_archetype(existing_arch):
                continue
            if self._family_token(existing_arch) != target_fam:
                return True
            if str(existing_arch).lower().strip() != target_arch:
                return True
        return False

    def _other_trend_occupiers(
        self, symbol: str, archetype: str
    ) -> list[tuple[str, str]]:
        """Return [(symbol, existing_archetype), ...] trend slots blocking ``archetype``."""
        sym = str(symbol or "").upper().strip()
        if not sym:
            return []
        target_arch = str(archetype or "").lower().strip()
        out: list[tuple[str, str]] = []
        for key in self._slot_evidence:
            if not key.startswith(f"{sym}:"):
                continue
            try:
                _, existing_arch = key.split(":", 1)
            except Exception:
                continue
            if not self._is_trend_archetype(existing_arch):
                continue
            ex = str(existing_arch).lower().strip()
            if ex == target_arch:
                continue
            out.append((sym, existing_arch))
        return out

    def _maybe_preempt_trend_occupiers(self, intent: TradeIntent) -> bool:
        """If enabled, evict preemptable trend sleeves and free slots for ``intent``.

        Returns True when conflict was cleared (caller should not reject).
        Evictions are recorded on ``_last_evictions`` for EB/live to flatten first.
        """
        if not self._trend_preempt_enabled:
            return False
        arch = str(intent.archetype or "").lower().strip()
        if arch not in self._trend_preemptors:
            return False
        occupiers = self._other_trend_occupiers(intent.symbol, intent.archetype)
        if not occupiers:
            return False
        for _sym, existing in occupiers:
            if str(existing).lower().strip() not in self._trend_preemptable:
                return False
        for sym, existing in occupiers:
            self._last_evictions.append((sym, existing))
            key = f"{sym}:{existing}"
            self._slot_evidence.pop(key, None)
            # Also drop case-variant keys if any
            for k in list(self._slot_evidence):
                if k.upper() == key.upper():
                    self._slot_evidence.pop(k, None)
            logger.info(
                "PCM: trend_preempt %s evicts %s on %s (flatten then open)",
                arch,
                existing,
                sym,
            )
        self._last_decide_trace["trend_preempt_evictions"] = int(
            self._last_decide_trace.get("trend_preempt_evictions", 0) or 0
        ) + len(occupiers)
        return True

    def _is_trend_slot_protected(self, slot: Dict[str, Any]) -> bool:
        unlock_on = self._trend_pool_unlock_on
        if unlock_on == "stop_risk_nonnegative":
            return bool(slot.get("stop_risk_nonnegative")) or bool(
                slot.get("breakeven_locked")
            )
        return bool(slot.get("breakeven_locked"))

    def _normalize_symbol_correlations(self, raw: Any) -> Dict[tuple[str, str], float]:
        out: Dict[tuple[str, str], float] = {}

        def _put(left: Any, right: Any, value: Any) -> None:
            a = str(left or "").upper().strip()
            b = str(right or "").upper().strip()
            if not a or not b or a == b:
                return
            try:
                corr = float(value)
            except Exception:
                return
            out[tuple(sorted((a, b)))] = corr

        if isinstance(raw, dict):
            for left, vals in raw.items():
                if isinstance(vals, dict):
                    for right, value in vals.items():
                        _put(left, right, value)
                elif isinstance(vals, (list, tuple, set)):
                    for right in vals:
                        _put(left, right, 1.0)
        elif isinstance(raw, (list, tuple)):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                _put(
                    item.get("left") or item.get("symbol_a"),
                    item.get("right") or item.get("symbol_b"),
                    item.get("correlation", item.get("corr", 1.0)),
                )
        return out

    def _configured_symbol_correlation(self, left: str, right: str) -> Optional[float]:
        a = str(left or "").upper().strip()
        b = str(right or "").upper().strip()
        if not a or not b or a == b:
            return None
        return self._trend_pool_corr_pairs.get(tuple(sorted((a, b))))

    def _trend_pool_correlation_reject_reason(
        self, intent: TradeIntent, slots: List[Dict[str, Any]]
    ) -> str:
        if (
            not self._trend_pool_corr_enabled
            or not self._trend_pool_corr_pairs
            or bool(intent.add_position)
        ):
            return ""
        sym = str(intent.symbol or "").upper().strip()
        side = _trend_side_token(intent.action)
        n_peers = 0
        worst_sym = ""
        worst_corr = 0.0
        for slot in slots:
            open_sym = str(slot.get("symbol", "")).upper().strip()
            if not open_sym or open_sym == sym:
                continue
            if self._trend_pool_corr_same_direction_only:
                open_side = _trend_side_token(slot.get("side") or slot.get("action"))
                if side and open_side and side != open_side:
                    continue
            corr = self._configured_symbol_correlation(sym, open_sym)
            if corr is None:
                continue
            cabs = abs(float(corr))
            if cabs > self._trend_pool_corr_threshold:
                n_peers += 1
                if cabs >= worst_corr:
                    worst_corr = cabs
                    worst_sym = open_sym
        # max_cluster=1 (prod default) ≡ reject the 2nd same-beta name.
        if n_peers >= int(self._trend_pool_corr_max_cluster):
            return f"symbol_correlation:{worst_sym}:{worst_corr:.2f}"
        return ""

    def _open_trend_slots(self) -> List[Dict[str, Any]]:
        """Trend-pool occupancy = exchange-backed opens, not PCM accepts.

        When ``get_open_trend_positions`` is wired, an empty list means flat —
        do **not** fall back to ``_slot_evidence``. That map is written at
        ``decide()`` (before place) and a failed / unseeded coin must not
        consume ``max_unprotected_symbols``. Fallback is only for callers
        that never bound a position snapshot.
        """
        out: List[Dict[str, Any]] = []
        if self._get_open_trend_positions is not None:
            try:
                rows = self._get_open_trend_positions()
            except Exception:
                rows = None
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    sym = str(row.get("symbol", "")).upper().strip()
                    arch = str(row.get("archetype", "")).lower().strip()
                    if not sym or not arch or not self._is_trend_archetype(arch):
                        continue
                    out.append(
                        {
                            "symbol": sym,
                            "archetype": arch,
                            "side": _trend_side_token(
                                row.get("side") or row.get("action")
                            ),
                            "breakeven_locked": bool(row.get("breakeven_locked")),
                            "stop_risk_nonnegative": bool(
                                row.get("stop_risk_nonnegative")
                            ),
                        }
                    )
                return out
        for key in self._slot_evidence.keys():
            try:
                sym, arch = key.split(":", 1)
            except Exception:
                continue
            if not self._is_trend_archetype(arch):
                continue
            out.append(
                {
                    "symbol": str(sym).upper().strip(),
                    "archetype": str(arch).lower().strip(),
                    "side": "",
                    "breakeven_locked": False,
                    "stop_risk_nonnegative": False,
                }
            )
        return out

    @staticmethod
    def _resolve_trend_pool_caps(
        guard: dict,
    ) -> tuple[str, int, int]:
        """Resolve unprotected / post-unlock caps from optional ``by_regime``.

        Precedence for active label:
        1. env ``MLBOT_TREND_POOL_ACTIVE_REGIME``
        2. ``guard.active_regime``
        3. empty → use top-level scalars only

        When ``by_regime[label]`` exists, overlay
        ``max_unprotected_symbols`` / ``max_symbols_after_unlock`` from that block;
        missing keys fall back to top-level.
        """
        env_rg = str(os.environ.get("MLBOT_TREND_POOL_ACTIVE_REGIME", "") or "").strip()
        cfg_rg = str(guard.get("active_regime", "") or "").strip()
        active = (env_rg or cfg_rg).lower()
        # Aliases
        if active in ("risk_on", "trend_bull"):
            active = "bull"
        elif active in ("risk_off", "trend_bear"):
            active = "bear"
        elif active in ("neutral", "chop", "range_to_bear"):
            active = "range"

        top_unprot = int(guard.get("max_unprotected_symbols", 0) or 0)
        after_raw = guard.get("max_symbols_after_unlock")
        try:
            top_after = 0 if after_raw is None else int(after_raw)
        except Exception:
            top_after = 0

        by = guard.get("by_regime") or {}
        if active and isinstance(by, dict):
            block = by.get(active)
            if isinstance(block, dict) and block:
                if block.get("max_unprotected_symbols") is not None:
                    try:
                        top_unprot = int(block.get("max_unprotected_symbols") or 0)
                    except Exception:
                        pass
                if "max_symbols_after_unlock" in block:
                    try:
                        v = block.get("max_symbols_after_unlock")
                        top_after = 0 if v is None else int(v)
                    except Exception:
                        pass
        return active, top_unprot, top_after

    @staticmethod
    def _resolve_corr_guard(guard: dict, active: str) -> dict:
        """Top-level ``symbol_correlation_guard`` plus optional ``by_regime`` overlay.

        ``by_regime[active].symbol_correlation_guard`` shallow-updates the top-level
        dict (``enabled`` / ``threshold`` / ``max_cluster`` / ``pairs``). Missing
        overlay → unchanged. Used so bull can allow a 2-name same-beta stack while
        bear/range keep the production cluster=1 gate.
        """
        base = guard.get("symbol_correlation_guard") or {}
        if not isinstance(base, dict):
            base = {}
        out = dict(base)
        by = guard.get("by_regime") or {}
        if not active or not isinstance(by, dict):
            return out
        block = by.get(str(active).lower())
        if not isinstance(block, dict):
            return out
        overlay = block.get("symbol_correlation_guard")
        if isinstance(overlay, dict) and overlay:
            out.update(overlay)
        return out

    def _trend_pool_guard_reject_reason(self, intent: TradeIntent) -> str:
        if (
            not self._trend_pool_guard_enabled
            or self._trend_pool_max_unprotected <= 0
            or bool(intent.add_position)
            or not self._is_trend_archetype(intent.archetype)
        ):
            return ""
        slots = self._open_trend_slots()
        sym = str(intent.symbol or "").upper().strip()
        if (
            self._trend_pool_require_anchor_first
            and self._trend_pool_anchor_symbol
            and sym != self._trend_pool_anchor_symbol
        ):
            anchor_slots = [
                s
                for s in slots
                if str(s.get("symbol", "")).upper().strip()
                == self._trend_pool_anchor_symbol
            ]
            if not anchor_slots or not any(
                self._is_trend_slot_protected(s) for s in anchor_slots
            ):
                return "anchor_first"
        if not slots:
            return ""
        open_symbols = {str(s.get("symbol", "")).upper().strip() for s in slots}
        open_symbols.discard("")
        if sym in open_symbols:
            return ""
        unprotected_symbols = {
            str(s.get("symbol", "")).upper().strip()
            for s in slots
            if not self._is_trend_slot_protected(s)
        }
        unprotected_symbols.discard("")
        if len(unprotected_symbols) >= self._trend_pool_max_unprotected:
            return "unprotected_cap"
        corr_reject = self._trend_pool_correlation_reject_reason(intent, slots)
        if corr_reject:
            return corr_reject
        if self._trend_pool_max_symbols_after_unlock > 0 and len(open_symbols) >= int(
            self._trend_pool_max_symbols_after_unlock
        ):
            return "post_unlock_cap"
        return ""

    def _observed_market_side(
        self,
        *,
        features: Dict[str, Any],
        policy: Dict[str, Any],
    ) -> str:
        close_key = str(policy.get("close_feature", "close"))
        ema_key = str(policy.get("ema_feature", "ema_200"))
        close_v = features.get(close_key)
        ema_v = features.get(ema_key)
        if close_v is None or ema_v is None:
            return "unknown"
        try:
            close_f = float(close_v)
            ema_f = float(ema_v)
        except Exception:
            return "unknown"
        return "long" if close_f > ema_f else "short"

    def _effective_market_side(
        self,
        *,
        symbol: str,
        features: Dict[str, Any],
        policy: Dict[str, Any],
    ) -> str:
        """返回防抖后的有效主方向: long/short/unknown"""
        observed = self._observed_market_side(features=features, policy=policy)
        debounce_bars = int(policy.get("debounce_bars", 1) or 1)
        if debounce_bars <= 1:
            return observed
        st = self._dir_state.get(symbol) or {
            "confirmed_side": "unknown",
            "pending_side": None,
            "pending_count": 0,
        }
        confirmed = str(st.get("confirmed_side", "unknown"))
        pending = st.get("pending_side")
        pending_count = int(st.get("pending_count", 0))
        if observed in {"long", "short"}:
            if confirmed == "unknown":
                if pending == observed:
                    pending_count += 1
                else:
                    pending = observed
                    pending_count = 1
                if pending_count >= debounce_bars:
                    confirmed = observed
                    pending = None
                    pending_count = 0
            elif observed != confirmed:
                if pending == observed:
                    pending_count += 1
                else:
                    pending = observed
                    pending_count = 1
                if pending_count >= debounce_bars:
                    confirmed = observed
                    pending = None
                    pending_count = 0
            else:
                pending = None
                pending_count = 0
        self._dir_state[symbol] = {
            "confirmed_side": confirmed,
            "pending_side": pending,
            "pending_count": pending_count,
        }
        return confirmed

    def _is_direction_allowed(
        self,
        intent: TradeIntent,
        *,
        market_side: str,
        policy: Dict[str, Any],
    ) -> bool:
        if not bool(policy.get("enabled", False)):
            return True
        mode = str(policy.get("mode", "ema200_single_direction")).lower().strip()
        if mode != "ema200_single_direction":
            return True

        if market_side == "unknown":
            default_side = str(policy.get("default_side", "both")).lower().strip()
            if default_side == "both":
                return True
            _, s = self._parse_family_and_side(intent.archetype, intent.action)
            return s == default_side

        family, intent_side = self._parse_family_and_side(
            intent.archetype, intent.action
        )
        if intent_side == "unknown":
            return True
        if intent_side == market_side:
            return True

        reverse_allowed = bool(policy.get("fer_reverse_allowed", True))
        reverse_families = [
            str(x).lower().strip()
            for x in (policy.get("reverse_exempt_families") or ["fer"])
        ]
        if reverse_allowed and family in set(reverse_families):
            return True
        return False

    # ── 注册 / 管理 ──

    def register(
        self,
        archetype: str,
        strategy: DecisionHandler,
        *,
        timeframe: Optional[str] = None,
    ) -> None:
        """注册一个 archetype 策略实例

        Args:
            archetype: 策略名称 (如 "bpc", "me", "fer")
            strategy: 实现 decide() 接口的策略实例
            timeframe: 该策略使用的主时间框架 (如 "240T", "60T")
                用于多时间框架模式下路由特征。
        """
        self._strategies[archetype] = strategy
        if timeframe is not None:
            self._strategy_timeframes[archetype] = timeframe
        logger.info(
            "PCM: 注册策略 archetype=%s (%s) timeframe=%s",
            archetype,
            type(strategy).__name__,
            timeframe or "default",
        )

    def set_strategy_symbol_universe(self, mapping: Dict[str, List[str]]) -> None:
        """Per-archetype tradable symbols (already filtered from universe via meta.yaml)."""
        self._strategy_symbol_sets = {
            str(k): frozenset(str(s).upper().strip() for s in vals if str(s).strip())
            for k, vals in (mapping or {}).items()
        }
        logger.info(
            "PCM: strategy symbol filters: %s",
            {k: len(v) for k, v in sorted(self._strategy_symbol_sets.items())},
        )

    def unregister(self, archetype: str) -> None:
        """移除一个 archetype 策略"""
        self._strategies.pop(archetype, None)

    @property
    def registered_archetypes(self) -> List[str]:
        return list(self._strategies.keys())

    @property
    def archetype_priority(self) -> List[str]:
        """\u5f53\u524d\u6709\u6548\u4f18\u5148\u7ea7\u5217\u8868\uff08\u5982\u6709 regime detector \u5219\u8fd4\u56de\u52a8\u6001\u4f18\u5148\u7ea7\uff09"""
        if self._regime_detector is not None:
            return self._regime_detector.current_priority
        return list(self._archetype_priority)

    @property
    def regime_detector(self) -> Optional[RegimeDetector]:
        return self._regime_detector

    @property
    def current_regime(self) -> str:
        if self._regime_detector is not None:
            return self._regime_detector.current_regime
        return REGIME_NORMAL

    @property
    def constitution(self) -> Dict[str, Any]:
        """Constitution 硬约束 (只读)"""
        return dict(self._constitution)

    def resolve_risk_for_strategy(self, archetype: str) -> float:
        """Return effective risk fraction for a strategy.

        Logic: min(risk_per_slot, strategy.max_risk_per_trade)
        If strategy has no max_risk_per_trade, returns risk_per_slot.
        """
        risk_per_slot = float(self._constitution.get("risk_per_slot", 0.01))
        strat = self._limit_cfg_for_archetype(archetype)
        strat_risk = strat.get("max_risk_per_trade")
        if strat_risk is not None:
            return min(risk_per_slot, float(strat_risk))
        return risk_per_slot

    @property
    def current_position_scale(self) -> float:
        """当前 Regime 的全局仓位缩放因子"""
        if self._regime_detector is not None:
            return self._regime_detector.current_position_scale
        return 1.0

    def get_archetype_scale(self, archetype: str) -> float:
        """获取当前 Regime 下特定 archetype 的仓位缩放因子"""
        if self._regime_detector is not None:
            return self._regime_detector.get_archetype_scale(archetype)
        return 1.0

    # ── 核心决策接口 ──

    def _get_priority_rank(self, archetype: str) -> int:
        """获取 archetype 的优先级排名（越小越优先）。
        如有 regime detector，使用动态优先级。
        不在列表中的 archetype 排到最后。"""
        priority = self.archetype_priority
        arch_lower = archetype.lower()
        for i, a in enumerate(priority):
            if a.lower() == arch_lower:
                return i
        return len(priority)  # 未知 archetype 排最后

    def _parse_timeframe_minutes(self, arch_name: str) -> int:
        tf = str(self._strategy_timeframes.get(arch_name, "") or "").upper().strip()
        if tf.endswith("T"):
            try:
                return int(tf[:-1])
            except Exception:
                return 0
        return 0

    def _to_ts_ns(self, val: Any) -> int:
        if val is None:
            return 2**63 - 1
        if isinstance(val, (int, float)):
            # 保持数量级，不做单位猜测；仅用于稳定排序
            return int(val)
        if isinstance(val, datetime):
            return int(val.timestamp() * 1e9)
        try:
            if hasattr(val, "value"):  # pandas.Timestamp
                return int(val.value)
        except Exception:
            pass
        try:
            if hasattr(val, "timestamp"):
                return int(float(val.timestamp()) * 1e9)
        except Exception:
            pass
        return 2**63 - 1

    def _effective_stop_pct_from_intent(
        self,
        intent: TradeIntent,
        feat: Dict[str, Any],
    ) -> float:
        ep = dict(intent.execution_profile or {})
        rr = dict(ep.get("rr_constraints") or {})
        try:
            sl_r = float(rr.get("stop_loss_r", 0.0) or 0.0)
        except Exception:
            sl_r = 0.0
        close = feat.get("close")
        atr = feat.get("atr")
        try:
            close_f = float(close or 0.0)
            atr_f = float(atr or 0.0)
        except Exception:
            return float("inf")
        if sl_r <= 0 or close_f <= 0 or atr_f <= 0:
            return float("inf")
        atr_stop = max(0.0, sl_r * atr_f / close_f)
        eff = atr_stop
        if rr.get("min_stop_pct") is not None:
            try:
                eff = max(eff, float(rr.get("min_stop_pct")))
            except Exception:
                pass
        if rr.get("max_stop_pct") is not None:
            try:
                eff = min(eff, float(rr.get("max_stop_pct")))
            except Exception:
                pass
        return max(0.0, float(eff))

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
        多策略独立 slot 仲裁 → 返回所有通过 per-strategy slot 检查的 TradeIntent

        算法:
          1. Regime 检测 (Layer 2) → 动态缩放
          2. 遍历所有注册策略，收集候选 TradeIntent
          3. 每策略独立检查 per-strategy slot (无跨策略竞争)
          4. slot 满时同 archetype 内 evidence 竞争

        Args:
            features: 主时间框架特征 (默认 4H)
            symbol: 交易对
            bars: 近期 bars (用于执行规则)
            features_by_timeframe: 多时间框架特征 {timeframe: features_dict}
                用于多策略多 timeframe 路由。各策略绑定的 timeframe
                通过 register(timeframe=...) 注册。
            decision_time: 可选，当前决策 bar 的时间（与 event_backtest 时间线一致）。
                用于 ``max_new_entries_per_day`` 的日历键；不传则尽量用 ``features["timestamp"]``，
                仍无时回退 ``datetime.now``（实盘）。

        Returns:
            List[TradeIntent]（每策略最多 1 个，可返回多个）
        """
        _pcm_day = _calendar_day_utc_str(features=features, decision_time=decision_time)
        self._last_decide_trace = {
            "all_intents": 0,
            "accepted": 0,
            "drop_direction_policy": 0,
            "drop_family_conflict": 0,
            "drop_daily_limit": 0,
            "drop_slot": 0,
            "drop_trend_pool_anchor_first": 0,
            "drop_trend_pool_unprotected_cap": 0,
            "drop_trend_pool_post_unlock_cap": 0,
            "drop_trend_pool_symbol_correlation": 0,
        }
        if not self._strategies:
            return []

        # 每次 decide() 重置驱逐列表 — 供调用方 (事件回测/实盘) 关闭被替换仓位
        self._last_evictions: List[Tuple[str, str]] = (
            []
        )  # [(evicted_symbol, archetype), ...]
        self._latest_features = dict(features or {})

        # ── 1. Regime 检测 (Layer 2) ──
        # Regime 使用主时间框架 (4H) 特征检测
        if self._regime_detector is not None:
            self._regime_detector.detect(features)

        # ── 2. 收集所有策略的候选信号 ──
        all_intents: List[TradeIntent] = []
        _intent_meta: Dict[int, Dict[str, Any]] = {}
        _collect_seq = 0
        for arch_name, strategy in self._strategies.items():
            try:
                allowed = self._strategy_symbol_sets.get(arch_name)
                if allowed is not None:
                    sym_u = str(symbol or "").upper().strip()
                    if sym_u not in allowed:
                        strategy._last_funnel = {"skip": "symbol_not_in_strategy_meta"}
                        continue
                # 多时间框架路由: 使用策略绑定的 timeframe 对应的特征
                strat_features = features
                if features_by_timeframe and arch_name in self._strategy_timeframes:
                    tf = self._strategy_timeframes[arch_name]
                    if tf in features_by_timeframe:
                        strat_features = features_by_timeframe[tf]
                    else:
                        # 该策略的 timeframe 当前不可用 → 跳过
                        # (例: 60T bar 上不评估 240T 策略，与实盘行为一致)
                        logger.debug(
                            "PCM: 跳过 %s — timeframe %s 不在当前可用 %s",
                            arch_name,
                            tf,
                            list(features_by_timeframe.keys()),
                        )
                        # 清空 _last_funnel 避免诊断代码读到上一次的结果
                        strategy._last_funnel = {}
                        if str(arch_name).lower() == "fer":
                            try:
                                from src.time_series_model.live.fer_diagnostics import (
                                    record_fer_entry_eval,
                                )

                                record_fer_entry_eval(
                                    strategy=str(arch_name),
                                    symbol=symbol,
                                    signal_ts=None,
                                    outcome="pcm_timeframe_missing",
                                    funnel={"pcm_timeframe": tf},
                                    features=dict(features or {}),
                                )
                            except Exception:
                                pass
                        continue

                # ── EMA 方向过滤 (price > ema_200 → bull; price ≤ ema_200 → bear) ──
                if self._ema_filter_enabled:
                    _close = strat_features.get(self._ema_close_feature)
                    _ema = strat_features.get(self._ema_feature)
                    if _close is not None and _ema is not None:
                        _is_bull = float(_close) > float(_ema)
                        _allowed = (
                            self._ema_bull_allowed
                            if _is_bull
                            else self._ema_bear_allowed
                        )
                        if arch_name not in _allowed:
                            # 标记为 PCM 方向过滤拒绝，供事件回测漏斗诊断使用
                            strategy._last_funnel = {"pcm_direction_filter": False}
                            logger.debug(
                                "PCM: EMA过滤跳过 %s — %s regime (close=%.4f, ema=%.4f)",
                                arch_name,
                                "BULL" if _is_bull else "BEAR",
                                float(_close),
                                float(_ema),
                            )
                            if str(arch_name).lower() == "fer":
                                try:
                                    from src.time_series_model.live.fer_diagnostics import (
                                        record_fer_entry_eval,
                                    )

                                    record_fer_entry_eval(
                                        strategy=str(arch_name),
                                        symbol=symbol,
                                        signal_ts=strat_features.get("timestamp"),
                                        outcome="pcm_ema_filter_deny",
                                        funnel={
                                            "pcm_direction_filter": False,
                                            "pcm_ema_regime": (
                                                "bull" if _is_bull else "bear"
                                            ),
                                        },
                                        features=dict(strat_features),
                                    )
                                except Exception:
                                    pass
                            continue
                    elif self._ema_fallback != "allow_all":
                        _allowed_fb = (
                            self._ema_bull_allowed
                            if self._ema_fallback == "bull"
                            else self._ema_bear_allowed
                        )
                        if arch_name not in _allowed_fb:
                            strategy._last_funnel = {"pcm_direction_filter": False}
                            if str(arch_name).lower() == "fer":
                                try:
                                    from src.time_series_model.live.fer_diagnostics import (
                                        record_fer_entry_eval,
                                    )

                                    record_fer_entry_eval(
                                        strategy=str(arch_name),
                                        symbol=symbol,
                                        signal_ts=strat_features.get("timestamp"),
                                        outcome="pcm_ema_fallback_deny",
                                        funnel={"pcm_direction_filter": False},
                                        features=dict(strat_features),
                                    )
                                except Exception:
                                    pass
                            continue

                intents = strategy.decide(
                    features=strat_features, symbol=symbol, bars=bars
                )
                all_intents.extend(intents)
                for it in intents:
                    _intent_meta[id(it)] = {
                        "strategy_name": arch_name,
                        "timeframe_minutes": self._parse_timeframe_minutes(arch_name),
                        "archetype": str(getattr(it, "archetype", "")).lower().strip(),
                        "effective_stop_pct": self._effective_stop_pct_from_intent(
                            it, strat_features
                        ),
                        "signal_ts_ns": self._to_ts_ns(strat_features.get("timestamp")),
                        "collect_seq": _collect_seq,
                    }
                    _collect_seq += 1

                # 收集漏斗统计
                if self.stats_collector is not None:
                    funnel = getattr(strategy, "_last_funnel", {})
                    self.stats_collector.record_strategy_eval(
                        symbol=symbol,
                        strategy=arch_name,
                        funnel=funnel,
                    )
            except Exception:
                logger.exception(
                    "PCM: 策略 %s 对 %s 调用 decide() 异常", arch_name, symbol
                )

        if not all_intents:
            return []

        # deterministic v1:
        # timeframe(大优先) -> archetype (constitution) -> effective_stop_pct(小优先) -> FIFO
        _arch_pri = intent_archetype_priority_tokens(self._constitution)
        _arch_rank = {a: i for i, a in enumerate(_arch_pri)}

        def _intent_sort_key(intent: TradeIntent) -> tuple:
            md = _intent_meta.get(id(intent), {})
            return (
                -int(md.get("timeframe_minutes", 0) or 0),
                int(_arch_rank.get(str(md.get("archetype", "")), 99)),
                float(md.get("effective_stop_pct", float("inf"))),
                int(md.get("signal_ts_ns", 2**63 - 1)),
                int(md.get("collect_seq", 10**9)),
            )

        all_intents = sorted(all_intents, key=_intent_sort_key)

        # 同 symbol + 同 archetype 的“新开仓”候选，只保留 deterministic 排序后的第一条。
        _dedup: Dict[tuple[str, str], TradeIntent] = {}
        _dedup_out: List[TradeIntent] = []
        for intent in all_intents:
            if bool(intent.add_position):
                _dedup_out.append(intent)
                continue
            _k = (str(intent.symbol), str(intent.archetype).lower())
            if _k not in _dedup:
                _dedup[_k] = intent
        all_intents = list(_dedup.values()) + _dedup_out

        self._last_decide_trace["all_intents"] = int(len(all_intents))

        # ── 3. 每策略独立 slot 检查 (无跨策略竞争) ──
        _dir_pol = dict(self._constitution.get("direction_policy") or {})
        _market_side = self._effective_market_side(
            symbol=str(symbol),
            features=features,
            policy=_dir_pol,
        )
        accepted: List[TradeIntent] = []
        for intent in all_intents:
            if not self._is_direction_allowed(
                intent,
                market_side=_market_side,
                policy=_dir_pol,
            ):
                self._last_decide_trace["drop_direction_policy"] = (
                    int(self._last_decide_trace.get("drop_direction_policy", 0) or 0)
                    + 1
                )
                logger.info(
                    "PCM: 方向过滤拒绝 %s %s action=%s (market_side=%s)",
                    intent.symbol,
                    intent.archetype,
                    intent.action,
                    _market_side,
                )
                continue
            # 同 symbol + 同策略家族（BPC/ME/FER/...）只允许单方向持仓（避免 one-way 模式净仓歧义）
            fam, side = self._parse_family_and_side(intent.archetype, intent.action)
            if side in {"long", "short"}:
                _conflict = False
                for k in self._slot_evidence:
                    if not k.startswith(f"{intent.symbol}:"):
                        continue
                    try:
                        _, arch2 = k.split(":", 1)
                    except Exception:
                        continue
                    fam2, side2 = self._parse_family_and_side(arch2)
                    if fam2 == fam and side2 in {"long", "short"} and side2 != side:
                        _conflict = True
                        break
                if _conflict:
                    self._last_decide_trace["drop_family_conflict"] = (
                        int(self._last_decide_trace.get("drop_family_conflict", 0) or 0)
                        + 1
                    )
                    logger.info(
                        "PCM: 家族方向冲突拒绝 %s %s (family=%s side=%s)",
                        intent.symbol,
                        intent.archetype,
                        fam,
                        side,
                    )
                    continue

            _slot_key = f"{intent.symbol}:{intent.archetype}"
            if _slot_key in self._slot_evidence and not bool(intent.add_position):
                _arch_lc = str(intent.archetype or "").strip().lower()
                if _arch_lc in self._float_ladder_archetypes:
                    self._last_decide_trace["drop_float_ladder_reentry"] = (
                        int(
                            self._last_decide_trace.get("drop_float_ladder_reentry", 0)
                            or 0
                        )
                        + 1
                    )
                    logger.info(
                        "PCM: %s %s 已有持仓 — float_r_ladder 跳过再入场，交由阶梯加仓",
                        intent.symbol,
                        intent.archetype,
                    )
                    continue
                intent = replace(intent, add_position=True)
                logger.info(
                    "PCM: %s %s 已有持仓，转为 add_position 意图",
                    intent.symbol,
                    intent.archetype,
                )
            if (
                self._enforce_single_trend_per_symbol
                and not bool(intent.add_position)
                and self._is_trend_archetype(intent.archetype)
                and self._has_other_trend_slot_on_symbol(
                    intent.symbol, intent.archetype
                )
            ):
                if self._maybe_preempt_trend_occupiers(intent):
                    # Conflict cleared via eviction; allow preemptor intent through.
                    pass
                else:
                    self._last_decide_trace["drop_trend_symbol_slot_conflict"] = (
                        int(
                            self._last_decide_trace.get(
                                "drop_trend_symbol_slot_conflict", 0
                            )
                            or 0
                        )
                        + 1
                    )
                    logger.info(
                        "PCM: trend symbol slot conflict reject %s %s",
                        intent.symbol,
                        intent.archetype,
                    )
                    continue
            _pool_reject = self._trend_pool_guard_reject_reason(intent)
            if _pool_reject:
                if _pool_reject == "unprotected_cap":
                    _trace_key = "drop_trend_pool_unprotected_cap"
                elif _pool_reject == "anchor_first":
                    _trace_key = "drop_trend_pool_anchor_first"
                elif _pool_reject.startswith("symbol_correlation:"):
                    _trace_key = "drop_trend_pool_symbol_correlation"
                else:
                    _trace_key = "drop_trend_pool_post_unlock_cap"
                self._last_decide_trace[_trace_key] = (
                    int(self._last_decide_trace.get(_trace_key, 0) or 0) + 1
                )
                logger.info(
                    "PCM: trend pool guard reject %s %s (%s)",
                    intent.symbol,
                    intent.archetype,
                    _pool_reject,
                )
                continue

            # ── 每日入场节流: 新开仓检查日内上限 ──
            if not bool(intent.add_position):
                _fam_throttle, _ = self._parse_family_and_side(
                    intent.archetype, intent.action
                )
                _throttle_limit = self._daily_entry_limits.get(_fam_throttle)
                if _throttle_limit is not None:
                    _dk = (_fam_throttle, str(symbol or "").upper().strip(), _pcm_day)
                    if self._daily_entry_counts.get(_dk, 0) >= _throttle_limit:
                        self._last_decide_trace["drop_daily_limit"] = (
                            int(self._last_decide_trace.get("drop_daily_limit", 0) or 0)
                            + 1
                        )
                        logger.info(
                            "PCM: %s %s 日入场上限已满 (%d/%d)，拒绝",
                            symbol,
                            intent.archetype,
                            self._daily_entry_counts[_dk],
                            _throttle_limit,
                        )
                        continue

            if not bool(intent.add_position) and not self._slot_available(
                symbol, intent.archetype
            ):
                # 该策略 slot 满 → 直接拒绝
                self._last_decide_trace["drop_slot"] = (
                    int(self._last_decide_trace.get("drop_slot", 0) or 0) + 1
                )
                logger.info(
                    "PCM: %s %s slot 已满 (%d/%d)，拒绝",
                    symbol,
                    intent.archetype,
                    self._count_archetype_slots(intent.archetype),
                    self._max_slots_for_strategy(intent.archetype),
                )
                continue
            ev = intent.confidence if intent.confidence is not None else 0.5
            if not bool(intent.add_position):
                self._record_slot(symbol, intent.archetype, ev)
                _fam_rec, _ = self._parse_family_and_side(
                    intent.archetype, intent.action
                )
                if _fam_rec in self._daily_entry_limits:
                    _dk = (_fam_rec, str(symbol or "").upper().strip(), _pcm_day)
                    self._daily_entry_counts[_dk] = (
                        self._daily_entry_counts.get(_dk, 0) + 1
                    )
            if self.stats_collector is not None:
                self.stats_collector.record_pcm_selected(symbol, intent.archetype)
            accepted.append(self._apply_regime_scale(intent))
            logger.info(
                "PCM: %s 选中 %s (scale=%.2f)",
                symbol,
                intent.archetype,
                self.get_archetype_scale(intent.archetype),
            )
        self._last_decide_trace["accepted"] = int(len(accepted))
        return accepted

    # ── Layer 3: Override 极端信号覆盖 ──

    def _check_override(
        self,
        intents: List[TradeIntent],
        features: Dict[str, Any],
    ) -> Optional[TradeIntent]:
        """Layer 3: 极端信号覆盖

        文档精神:
          - LV 属于非线性事件，应有最高抢占权 → LV 覆盖所有
          - 强动能机会 → ME 覆盖 BPC
          - 失败/反转信号 → FER 覆盖 ME

        检查顺序按抢占权高低: LV → FER → ME

        Returns:
            覆盖后的优胜 TradeIntent，或 None（无覆盖触发）
        """
        if not self._override_config:
            return None

        # 构建 archetype → intent 映射（大写键，保留最高 evidence）
        intent_map: Dict[str, TradeIntent] = {}
        for intent in intents:
            key = intent.archetype.upper()
            existing = intent_map.get(key)
            if existing is None or (intent.confidence or 0) > (
                existing.confidence or 0
            ):
                intent_map[key] = intent

        # 按抢占权顺序检查: LV > FER > ME
        for arch_name in ["LV", "FER", "ME"]:
            rule = self._override_config.get(arch_name)
            if rule is None:
                continue

            if arch_name not in intent_map:
                continue  # 该 archetype 未触发信号

            candidate = intent_map[arch_name]
            evidence = candidate.confidence if candidate.confidence is not None else 0.5

            # 检查最低 evidence 阈值
            min_ev = float(rule.get("min_evidence", 0.0))
            if evidence < min_ev:
                continue

            # 检查额外特征条件
            conditions = rule.get("conditions", [])
            if conditions:
                det = {"conditions": conditions, "logic": rule.get("logic", "AND")}
                if not RegimeDetector._check_conditions(features, det):
                    continue

            # 检查覆盖目标
            overrides_target = rule.get("overrides", [])
            if isinstance(overrides_target, str) and overrides_target.upper() == "ALL":
                # 覆盖所有：只要该 archetype 触发就赢
                logger.info(
                    "PCM Override: %s 覆盖所有 (evidence=%.2f)",
                    arch_name,
                    evidence,
                )
                return candidate
            else:
                # 覆盖特定 archetype
                if isinstance(overrides_target, str):
                    targets = [overrides_target.upper()]
                else:
                    targets = [t.upper() for t in overrides_target]

                # 只有当被覆盖目标也在候选中时才触发
                if any(t in intent_map for t in targets):
                    logger.info(
                        "PCM Override: %s 覆盖 %s (evidence=%.2f)",
                        arch_name,
                        ", ".join(targets),
                        evidence,
                    )
                    return candidate

        return None

    # ── Regime 仓位缩放 ──

    def _apply_regime_scale(self, intent: TradeIntent) -> TradeIntent:
        """Apply regime-aware position scaling to TradeIntent.

        缩放因子乘在 size_multiplier 上，不修改其他字段。
        最终 size_multiplier = original × regime_scale
        """
        existing_mult = (
            intent.size_multiplier if intent.size_multiplier is not None else 1.0
        )

        # Regime 缩放
        regime_scale = self.get_archetype_scale(intent.archetype)
        new_mult = existing_mult * regime_scale

        if new_mult >= 1.0 and regime_scale >= 1.0:
            return intent

        logger.debug(
            "PCM scale: %s %s regime=%.2f total=%.2f → %.2f",
            intent.symbol,
            intent.archetype,
            regime_scale,
            existing_mult,
            new_mult,
        )
        # TradeIntent 是 frozen dataclass，需要重建
        return TradeIntent(
            action=intent.action,
            symbol=intent.symbol,
            archetype=intent.archetype,
            execution_strategy=intent.execution_strategy,
            confidence=intent.confidence,
            quantity=intent.quantity,
            size_multiplier=new_mult,
            position_id=intent.position_id,
            add_position=intent.add_position,
            parent_position_id=intent.parent_position_id,
            current_r=intent.current_r,
            locked_profit=intent.locked_profit,
            execution_tags=intent.execution_tags,
            execution_evidence=intent.execution_evidence,
            execution_profile=intent.execution_profile,
            pcm_budget=intent.pcm_budget,
        )

    # ── 内部方法 ──

    def _current_slot_count(self) -> int:
        """当前已占用 slot 数"""
        if self._get_open_slot_count is not None:
            return self._get_open_slot_count()
        return 0

    def _max_slots_for_strategy(self, archetype: str) -> int:
        """获取策略的 max_slots (从 per_strategy_limits 读取，缺省回退全局)"""
        strat = self._limit_cfg_for_archetype(archetype)
        return max(0, int(strat.get("max_slots", self._max_slots)))

    def _risk_for_strategy(self, archetype: str) -> float:
        strat = self._limit_cfg_for_archetype(archetype)
        risk_per_slot = float(self._constitution.get("risk_per_slot", 0.01))
        if "max_risk_per_trade" not in strat:
            return risk_per_slot
        try:
            return min(risk_per_slot, float(strat.get("max_risk_per_trade")))
        except Exception:
            return risk_per_slot

    def _count_archetype_slots(self, archetype: str) -> int:
        """统计某 archetype 当前占用的 slot 数"""
        suffix = f":{archetype}"
        return sum(1 for k in self._slot_evidence if k.endswith(suffix))

    def _slot_available(self, symbol: str, archetype: str) -> bool:
        """检查策略是否有可用 slot (per-strategy 独立 + 全局上限)"""
        if self._get_open_slot_count is None:
            return True  # 未配置回调，不做限制
        # Per-strategy slot 上限
        max_strat = self._max_slots_for_strategy(archetype)
        if self._count_archetype_slots(archetype) >= max_strat:
            return False
        # 全局 slot 上限
        return self._current_slot_count() < self._max_slots

    def _record_slot(self, symbol: str, archetype: str, evidence: float) -> None:
        """记录已入场 slot"""
        key = f"{symbol}:{archetype}"
        self._slot_evidence[key] = evidence

    def notify_position_closed(self, symbol: str, archetype: str = "") -> None:
        """外部通知仓位已平仓，清理 slot 追踪

        由 PositionManager/OrderFlowListener 在仓位关闭时调用。
        真平仓不退回 ``max_new_entries_per_day``（当天已用掉一次开仓）。
        """
        if archetype:
            key = f"{symbol}:{archetype}"
            self._slot_evidence.pop(key, None)
        else:
            # archetype 未知，清理该 symbol 的所有 slot
            to_remove = [k for k in self._slot_evidence if k.startswith(f"{symbol}:")]
            for k in to_remove:
                del self._slot_evidence[k]

    def notify_open_failed(
        self,
        symbol: str,
        archetype: str = "",
        *,
        action: str = "",
        features: Optional[Dict[str, Any]] = None,
        decision_time: Any = None,
    ) -> None:
        """开仓未成交（软拒 / 下单失败）：清 slot，并退回当日新开计数。

        PCM 在 decide() 选中时就 +1。若随后 place 失败仍占额度，同日下一根
        2h 会 ``drop_daily_limit``（2026-08-20 SRB 漏斗过、16:00 仍无单）。
        """
        self.notify_position_closed(symbol, archetype)
        arch = str(archetype or "").strip()
        if not arch:
            return
        fam, _ = self._parse_family_and_side(arch, action)
        if fam not in self._daily_entry_limits:
            return
        day = _calendar_day_utc_str(
            features=features or {}, decision_time=decision_time
        )
        dk = (fam, str(symbol or "").upper().strip(), day)
        cur = int(self._daily_entry_counts.get(dk, 0) or 0)
        if cur > 0:
            self._daily_entry_counts[dk] = cur - 1
            logger.info(
                "PCM: %s %s 开仓失败，退回日入场计数 %d→%d (%s)",
                symbol,
                arch,
                cur,
                cur - 1,
                day,
            )

    def hydrate_slot_evidence_from_constitution_slots(self, runtime_st: Any) -> None:
        """After process restart, rebuild in-memory ``_slot_evidence`` from persisted slots.

        Call **after** ``load_runtime_state`` and ``_sync_slots_with_exchange`` so the
        snapshot only contains exchange-backed slots. Without this, ``_slot_evidence`` is
        empty while constitution still tracks active slots → the next signal is treated
        as a new entry (``add_position=False``) instead of an add.
        """
        if runtime_st is None:
            return
        slots = getattr(runtime_st, "slots", None)
        active = getattr(slots, "active", None) if slots is not None else None
        if not isinstance(active, dict) or not active:
            return
        n = 0
        for rec in active.values():
            if rec is None:
                continue
            sym = str(getattr(rec, "symbol", None) or "").upper().strip()
            arch = str(getattr(rec, "archetype", None) or "").strip().lower()
            if not sym or not arch:
                continue
            self._record_slot(sym, arch, 0.5)
            n += 1
        if n:
            logger.info(
                "PCM: hydrated %d slot(s) from constitution runtime (restart recovery)",
                n,
            )

    # ── 配置加载 ──

    def load_all_configs(self) -> None:
        """调用所有注册策略的 load_configs()"""
        for arch_name, strategy in self._strategies.items():
            if hasattr(strategy, "load_configs"):
                strategy.load_configs()
                logger.info("PCM: 已加载配置给 %s", arch_name)

    # ── PCM 统计 ──

    def get_stats(self) -> Dict[str, Any]:
        """获取 PCM 运行统计信息（用于 KPI 评估）"""
        stats: Dict[str, Any] = {
            "registered_archetypes": self.registered_archetypes,
            "current_priority": self.archetype_priority,
            "max_slots": self._max_slots,
            "max_slots_source": (
                "constitution"
                if not hasattr(self, "_explicit_max_slots")
                else "explicit"
            ),
            "override_enabled": bool(self._override_config),
            "override_rules": (
                list(self._override_config.keys()) if self._override_config else []
            ),
            "constitution": {
                "slot_count": self._constitution.get("slot_count"),
                "risk_per_slot": self._constitution.get("risk_per_slot"),
                "per_strategy_limits": self._constitution.get("per_strategy_limits"),
                "slot_policy": self._constitution.get("slot_policy"),
                "account_risk_limits": self._constitution.get("account_risk_limits"),
            },
        }
        if self._regime_detector is not None:
            stats["current_regime"] = self._regime_detector.current_regime
            stats["regime_switch_count"] = self._regime_detector.switch_count
            stats["current_position_scale"] = (
                self._regime_detector.current_position_scale
            )
        return stats


def create_live_pcm(
    archetype_priority: Optional[List[str]] = None,
    max_slots: Optional[int] = None,
    get_open_slot_count: Optional[callable] = None,
    get_open_trend_positions: Optional[callable] = None,
    regime_config_path: Optional[str] = None,
    override_config: Optional[Dict[str, Any]] = None,
    constitution_yaml: Optional[str] = None,
) -> LivePCM:
    """
    创建 LivePCM 实例 (v3)

    Args:
        archetype_priority: 静态优先级列表。
        max_slots: 显式指定 max_slots。未提供时从 constitution 读取。
        get_open_slot_count: 可选回调
        regime_config_path: 可选 pcm_regime.yaml 路径
        override_config: Layer 3 Override 配置
        constitution_yaml: 可选 constitution.yaml 路径

    Returns:
        初始化好的 LivePCM（尚未注册策略）
    """
    return LivePCM(
        archetype_priority=archetype_priority,
        max_slots=max_slots,
        get_open_slot_count=get_open_slot_count,
        get_open_trend_positions=get_open_trend_positions,
        regime_config_path=regime_config_path,
        override_config=override_config,
        constitution_yaml=constitution_yaml,
    )
