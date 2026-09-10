"""
Archetype Loader - 多层配置加载器

从 config/strategies/{strategy}/archetypes/ 加载：
- regime.yaml: Regime 数据空间约束（EMA 带通 / chop 上限 / box 空间 + allowed_sides 多空掩码）
- prefilter.yaml: Prefilter 策略入场形态 (archetype 成立前提)
- direction.yaml: Direction 多空检测（在 regime allowed_sides 范围内）
- gate.yaml: Gate 规则 (硬 veto，仅执行风险)
- evidence.yaml: Evidence 规则 (软调整)
- entry_filters.yaml: Entry 订单流确认
- execution.yaml: Execution 约束 (RR/持仓)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import yaml

from src.config.strategy_layout import resolve_strategy_package_under_root

# =============================================================================
# Gate Config
# =============================================================================


@dataclass
class GateRule:
    """单条 Gate 规则"""

    id: str
    tag: str
    phase: str  # system_safety / hard_gate / guardrail
    priority: int
    reason: str
    when: Dict[str, Any]
    then: Dict[str, Any]
    frozen: bool = False  # 禁止优化阈值
    locked: bool = False  # 特征锁定（慢变量不可删除）
    # promote 时优化失败也不写 disabled（可与 locked 配合，且仍允许非 frozen 下调阈值）
    promote_never_disable: bool = False
    disabled: bool = False  # 临时禁用（KPI 不满足时保留但不执行）

    @property
    def is_hard(self) -> bool:
        return self.phase in ("system_safety", "hard_gate", "guardrail")


@dataclass
class GateConfig:
    """Gate 配置 - 从 gate.yaml 加载"""

    hard_gates: List[GateRule] = field(default_factory=list)
    system_safety: List[GateRule] = field(default_factory=list)
    guardrails: List[GateRule] = field(default_factory=list)
    governance: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_rules(self) -> List[GateRule]:
        """按 phase -> priority 排序的所有规则（含 guardrails）"""
        phase_order = {"system_safety": 0, "hard_gate": 1, "guardrail": 2}
        all_rules = self.system_safety + self.hard_gates + self.guardrails
        return sorted(
            all_rules, key=lambda r: (phase_order.get(r.phase, 99), r.priority)
        )

    @property
    def hard_rules(self) -> List[GateRule]:
        """所有硬规则 (system_safety + hard_gate + guardrail)"""
        return self.system_safety + self.hard_gates + self.guardrails

    @classmethod
    def from_yaml(
        cls,
        path: Path,
        *,
        prefilter_path: Optional[Path] = None,  # 保留参数签名向后兼容, 但不再使用
    ) -> "GateConfig":
        """从 gate.yaml 加载。prefilter 仅在训练时过滤数据, 不再注入为 guardrails。"""
        if not path.exists():
            return cls()

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        def _parse_rules(rules_list: List[Dict], default_phase: str) -> List[GateRule]:
            result = []
            for r in rules_list or []:
                result.append(
                    GateRule(
                        id=str(r.get("id", "")),
                        tag=str(r.get("tag", r.get("id", ""))),
                        phase=str(r.get("phase", default_phase)),
                        priority=int(r.get("priority", 99)),
                        reason=str(r.get("reason", "")),
                        when=dict(r.get("when") or {}),
                        then=dict(r.get("then") or {}),
                        frozen=bool(r.get("frozen", False)),
                        locked=bool(r.get("locked", False)),
                        promote_never_disable=bool(
                            r.get("promote_never_disable", False)
                        ),
                        disabled=bool(r.get("disabled", False)),
                    )
                )
            return result

        # gate.yaml 中的 guardrails（向后兼容）
        yaml_guardrails = _parse_rules(raw.get("guardrails"), "guardrail")

        return cls(
            hard_gates=_parse_rules(raw.get("hard_gates"), "hard_gate"),
            system_safety=_parse_rules(raw.get("system_safety"), "system_safety"),
            guardrails=yaml_guardrails,
            governance=dict(
                raw.get("governance") or raw.get("schema", {}).get("governance") or {}
            ),
        )


# =============================================================================
# Evidence Config
# =============================================================================

_DIRECTION_MAP = {
    "positive": "higher_is_better",
    "negative": "lower_is_better",
    "higher_is_better": "higher_is_better",
    "lower_is_better": "lower_is_better",
}


def _map_direction(raw: str) -> str:
    """Map YAML direction values to internal direction constants."""
    return _DIRECTION_MAP.get(str(raw).lower().strip(), "higher_is_better")


@dataclass
class EvidenceFeature:
    """单个 Evidence 特征"""

    id: str
    feature: str
    rank: int
    split_count: int
    quantile_bins: List[float]
    quantile_labels: List[str]
    # ❗ Bug 1 修复: 特征方向
    # "higher_is_better": 值越大越好 (如 strength, momentum)
    # "lower_is_better": 值越小越好 (如 volatility, risk, drawdown)
    direction: str = "higher_is_better"
    # 以下字段保留兼容旧 YAML，运行时不消费
    usage_hint: str = ""
    affects: List[str] = field(default_factory=list)
    threshold_examples: List[float] = field(default_factory=list)
    distribution_hint: str = ""

    def compute_label(self, value: float, quantiles: Dict[str, float]) -> str:
        """
        根据 quantile_mapping 计算语义标签

        Args:
            value: 特征原始值
            quantiles: {feature: {0.2: v1, 0.4: v2, ...}} 分位数查找表

        Returns:
            语义标签: suppress/downweight/neutral/favor/amplify
        """
        # 处理 quantiles 为 None 的情况 - 用 value 直接作为分位数
        if quantiles is None:
            # 假设 value 已经是 [0, 1] 范围的分位数
            percentile = value
            for i, bin_val in enumerate(self.quantile_bins):
                if percentile <= bin_val:
                    return (
                        self.quantile_labels[i]
                        if i < len(self.quantile_labels)
                        else "neutral"
                    )
            return self.quantile_labels[-1] if self.quantile_labels else "neutral"

        feat_q = quantiles.get(self.feature, {})
        if not feat_q:
            return "neutral"

        # 获取分位数阈值
        thresholds = []
        for q in self.quantile_bins:
            q_key = f"{q:.2f}".rstrip("0").rstrip(".")
            if q_key in feat_q:
                thresholds.append(float(feat_q[q_key]))
            elif str(q) in feat_q:
                thresholds.append(float(feat_q[str(q)]))
            else:
                return "neutral"  # 缺少分位数数据

        # 根据阈值确定标签
        for i, thresh in enumerate(thresholds):
            if value <= thresh:
                return (
                    self.quantile_labels[i]
                    if i < len(self.quantile_labels)
                    else "neutral"
                )

        # 超过所有阈值，返回最后一个标签
        return self.quantile_labels[-1] if self.quantile_labels else "neutral"

    def compute_score(self, value: float, quantiles: Dict[str, float]) -> float:
        """
        计算 Evidence 评分 (0-1 范围)

        标签映射:
        - suppress: 0.0
        - downweight: 0.25
        - neutral: 0.5
        - favor: 0.75
        - amplify: 1.0
        """
        label = self.compute_label(value, quantiles)
        score_map = {
            "suppress": 0.0,
            "downweight": 0.25,
            "neutral": 0.5,
            "favor": 0.75,
            "amplify": 1.0,
        }
        return score_map.get(label, 0.5)


@dataclass
class EvidenceConfig:
    """Evidence 配置 - 从 evidence.yaml 加载"""

    features: List[EvidenceFeature] = field(default_factory=list)
    min_score: float = (
        0.0  # 策略级 evidence 入场门槛 (由 optimize_evidence_plateau 自动计算)
    )
    # 保留兼容旧 YAML，运行时不消费
    label_semantics: Dict[str, str] = field(default_factory=dict)

    def compute_composite_score(
        self,
        feature_values: Dict[str, float],
        quantiles: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算 Evidence 综合评分

        Args:
            feature_values: {feature_name: value} 特征值字典
            quantiles: 分位数查找表

        Returns:
            (composite_score, {feature_id: score}) 综合分和各特征得分
        """
        scores = {}
        total_weight = 0.0
        weighted_sum = 0.0

        for feat in self.features:
            if feat.feature not in feature_values:
                continue

            value = feature_values[feat.feature]
            score = feat.compute_score(value, quantiles)
            scores[feat.id] = score

            # 用 rank 作为权重 (rank 越低越重要)
            weight = 1.0 / max(1, feat.rank)
            weighted_sum += score * weight
            total_weight += weight

        composite = weighted_sum / total_weight if total_weight > 0 else 0.5
        return composite, scores

    @classmethod
    def from_yaml(cls, path: Path) -> "EvidenceConfig":
        """从 YAML 文件加载"""
        if not path.exists():
            return cls()

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        features = []
        for e in raw.get("evidence") or []:
            qm = e.get("quantile_mapping") or {}
            features.append(
                EvidenceFeature(
                    id=str(e.get("id", "")),
                    feature=str(e.get("feature", "")),
                    rank=int(e.get("rank", 99)),
                    split_count=int(e.get("split_count", 0)),
                    usage_hint=str(e.get("usage_hint", "")),
                    affects=list(e.get("affects") or []),
                    quantile_bins=list(qm.get("bins") or [0.2, 0.4, 0.6, 0.8]),
                    quantile_labels=list(
                        qm.get("labels")
                        or ["suppress", "downweight", "neutral", "favor", "amplify"]
                    ),
                    threshold_examples=list(e.get("threshold_examples") or []),
                    distribution_hint=str(e.get("distribution_hint", "")),
                    direction=_map_direction(e.get("direction", "positive")),
                )
            )

        return cls(
            features=features,
            min_score=float(raw.get("min_score", 0.0)),
            label_semantics=dict(
                (raw.get("schema") or {}).get("label_semantics") or {}
            ),
        )


# =============================================================================
# Execution Config
# =============================================================================


@dataclass
class ExecutionConfig:
    """Execution 配置 - 从 execution.yaml 加载"""

    allow_add_on: bool = False
    min_order_interval_minutes: int = 60
    stop_loss_r: float = 1.0
    take_profit_r: float = 2.5
    max_holding_bars: Optional[int] = None
    min_holding_bars: Optional[int] = None
    direction_source: str = "structure"
    direction_method: str = "trend_sign"
    direction_lookback_bars: int = 5
    direction_min_consistency: float = 0.6
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "ExecutionConfig":
        """从 YAML 文件加载"""
        if not path.exists():
            return cls()

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        ec = raw.get("execution_constraints") or {}
        fixed_rr = ec.get("fixed_rr") or {}
        dp = raw.get("direction_policy") or {}
        sd = dp.get("structure_direction") or {}

        return cls(
            allow_add_on=bool(ec.get("allow_add_on", False)),
            min_order_interval_minutes=int(ec.get("min_order_interval_minutes", 60)),
            stop_loss_r=float(fixed_rr.get("stop_loss_r", 1.0)),
            take_profit_r=float(fixed_rr.get("take_profit_r", 2.5)),
            max_holding_bars=fixed_rr.get("max_holding_bars"),
            min_holding_bars=fixed_rr.get("min_holding_bars"),
            direction_source=str(dp.get("direction_source", "structure")),
            direction_method=str(sd.get("method", "trend_sign")),
            direction_lookback_bars=int(sd.get("lookback_bars", 5)),
            direction_min_consistency=float(sd.get("min_consistency", 0.6)),
            raw=raw,
        )


# =============================================================================
# Prefilter Config
# =============================================================================

import operator as _op

_PF_OPS = {
    ">=": _op.ge,
    ">": _op.gt,
    "<=": _op.le,
    "<": _op.lt,
    "==": _op.eq,
    "!=": _op.ne,
}


@dataclass
class PrefilterConfig:
    """
    Prefilter 配置 - 运行时前置条件过滤。

    语义: archetype 成立的前提环境条件。
    不满足 prefilter 的 bar 不应产生信号 (训练时过滤数据, 运行时跳过决策)。
    独立于 Gate, 在 decide() 管线最前端执行。
    """

    rules: List[Dict[str, Any]] = field(default_factory=list)
    _latched: Set[Tuple[str, int]] = field(
        default_factory=set, init=False, repr=False, compare=False
    )

    @staticmethod
    def _is_nan(val: Any) -> bool:
        try:
            fv = float(val)
        except (TypeError, ValueError):
            return True
        return fv != fv

    @classmethod
    def from_yaml(cls, path: Path) -> "PrefilterConfig":
        if not path.exists():
            return cls()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(rules=raw.get("rules", []))

    def evaluate(
        self,
        features: Dict[str, Any],
        *,
        latch_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        评估 prefilter 规则。

        ``latch: true`` on a rule (opt-in): after the rule first passes for
        ``latch_id`` (usually the symbol), later bars keep passing even if
        the feature fails. Live spot may set this as an operator override
        (research latch variants were worse). Persist via
        ``restore_latched`` / ``persist_latched``.

        Returns:
            (passed, reject_reason)
            passed=True 表示满足前置条件, 可继续;
            passed=False + reject_reason 说明哪条规则不满足。
        """
        if not self.rules:
            return True, None

        for i, rule in enumerate(self.rules):
            skip = {
                str(x).strip().upper()
                for x in (rule.get("skip_symbols") or [])
                if str(x).strip()
            }
            if latch_id and str(latch_id).strip().upper() in skip:
                continue
            latch_key: Optional[Tuple[str, int]] = None
            if bool(rule.get("latch")) and latch_id:
                latch_key = (str(latch_id), i)
                if latch_key in self._latched:
                    continue

            # ── any_of OR 组: 任一子规则满足即通过 ──
            if "any_of" in rule:
                sub_rules = rule["any_of"]
                available_sub_rules = [
                    sub
                    for sub in sub_rules
                    if isinstance(sub, dict)
                    and sub.get("feature") in features
                    and features.get(sub.get("feature")) is not None
                    and not PrefilterConfig._is_nan(features.get(sub.get("feature")))
                ]
                if not available_sub_rules:
                    return (
                        False,
                        "prefilter_any_of_fail: all sub-rules unavailable (missing or nan)",
                    )
                any_pass = False
                for sub in available_sub_rules:
                    if self._check_single(sub, features):
                        any_pass = True
                        break
                if not any_pass:
                    descs = []
                    for s in available_sub_rules:
                        if not isinstance(s, dict):
                            continue
                        fn = s.get("feature", "?")
                        fv = features.get(fn)
                        av = (
                            "missing"
                            if fv is None
                            else (f"{float(fv):.6g}" if fv == fv else "nan")
                        )
                        descs.append(
                            f"{fn}{s.get('operator', '?')}{s.get('value', '?')}(got={av})"
                        )
                    return False, f"prefilter_any_of_fail: {' OR '.join(descs)}"
                if latch_key is not None:
                    self._latched.add(latch_key)
                continue

            # ── 普通 AND 规则 ──
            if not self._check_single(rule, features):
                feat = rule.get("feature", "?")
                op_str = rule.get("operator", "?")
                val = rule.get("value", "?")
                fv = features.get(feat)
                if fv is None:
                    actual = "missing"
                else:
                    try:
                        fv_f = float(fv)
                        actual = "nan" if fv_f != fv_f else f"{fv_f:.6g}"
                    except (TypeError, ValueError):
                        actual = str(fv)
                return False, f"prefilter_fail: {feat} {op_str} {val} (actual={actual})"
            if latch_key is not None:
                self._latched.add(latch_key)

        return True, None

    def restore_latched(self, path: Path | str) -> int:
        """Load ``[(latch_id, rule_index), ...]`` from JSON. Missing file = 0."""
        p = Path(path)
        if not p.is_file():
            return 0
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return 0
        n = 0
        if not isinstance(raw, list):
            return 0
        for row in raw:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            try:
                self._latched.add((str(row[0]), int(row[1])))
            except (TypeError, ValueError):
                continue
            n += 1
        return n

    def persist_latched(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        rows = [[k[0], int(k[1])] for k in sorted(self._latched)]
        p.write_text(json.dumps(rows) + "\n", encoding="utf-8")

    def clear_latch_id(self, latch_id: str) -> int:
        """Drop every latched rule for ``latch_id``. Returns removed keys."""
        lid = str(latch_id or "").strip().upper()
        if not lid:
            return 0
        before = len(self._latched)
        self._latched = {
            key for key in self._latched if str(key[0]).strip().upper() != lid
        }
        return before - len(self._latched)

    def seed_latch_ids(self, symbols: Iterable[str]) -> int:
        """Arm every ``latch: true`` rule for the given ids. Returns new keys."""
        n = 0
        for i, rule in enumerate(self.rules):
            if not bool(rule.get("latch")):
                continue
            for raw in symbols:
                sym = str(raw or "").strip().upper()
                if not sym:
                    continue
                key = (sym, i)
                if key in self._latched:
                    continue
                self._latched.add(key)
                n += 1
        return n

    @staticmethod
    def _check_single(rule: Dict[str, Any], features: Dict[str, Any]) -> bool:
        """检查单条 prefilter 规则"""
        feat = rule.get("feature")
        op_str = rule.get("operator")
        val = rule.get("value")
        if not feat or not op_str:
            return True  # 格式不完整, 跳过

        op_func = _PF_OPS.get(op_str)
        if op_func is None:
            raise ValueError(
                f"Prefilter: unknown operator '{op_str}' for feature '{feat}'"
            )

        fv = features.get(feat)
        if fv is None:
            import logging as _logging

            warned = getattr(PrefilterConfig, "_warned_missing_features", None)
            if warned is None:
                warned = set()
                PrefilterConfig._warned_missing_features = warned
            if feat not in warned:
                warned.add(str(feat))
                _logging.getLogger(__name__).warning(
                    "Prefilter feature '%s' is missing from features dict "
                    "(available: %d keys). Rule treated as FAIL (no trade).",
                    feat,
                    len(features),
                )
            return False  # 特征缺失 → prefilter 不通过 (保守: 不交易)
        try:
            fv_f = float(fv)
        except (TypeError, ValueError):
            return False
        if fv_f != fv_f:  # NaN — side-gated / unavailable leg
            return False
        return bool(op_func(fv_f, float(val)))


# =============================================================================
# Regime Config
# =============================================================================

_DEFAULT_ALLOWED_REGIMES: Tuple[str, ...] = ("bull", "bear", "neutral")
_DEFAULT_ALLOWED_SIDES: Tuple[str, ...] = ("long", "short")


@dataclass
class RegimeConfig(PrefilterConfig):
    """
    Regime 配置 - 数据空间约束 + 多空掩码。

    职责（区分 Prefilter）:
        - Prefilter 描述 archetype 入场形态（结构条件，策略语义）
        - Regime 描述这段策略允许的数据空间（EMA 带、chop 上限、box 状态）
          以及方向掩码（allowed_sides），是 A/B/C 系统共用的慢变量层。

    规则评估复用 PrefilterConfig 的逻辑（threshold rules），但语义上独立：
        - 不满足 regime → reject_reason = "regime_..."
        - 不满足 prefilter → reject_reason = "prefilter_..."
    """

    allowed_regimes: List[str] = field(
        default_factory=lambda: list(_DEFAULT_ALLOWED_REGIMES)
    )
    allowed_sides: List[str] = field(
        default_factory=lambda: list(_DEFAULT_ALLOWED_SIDES)
    )
    # Optional per-bar side mask (evaluated after direction is known).
    # Keys: enabled, long_when, short_when — same when-clause shape as gate rules.
    side_mask: Dict[str, Any] = field(default_factory=dict)
    # Labeled schema: if True, classify()=="neutral" → entry deny (default False = prod).
    reject_neutral: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RegimeConfig":
        """从 regime.yaml 映射加载（缺失键 → 全开放默认值）

        兼容两种 schema:
          - 旧: ``allowed_regimes: [bull, bear, neutral]`` (mask 模式)
          - 新: ``allowed_regimes: {bull: {rules: [...]}, bear: {...}}`` (分类模式)
        """
        _ar = raw.get("allowed_regimes")
        if isinstance(_ar, dict):
            # New labeled schema — classify() will use per-label rules
            allowed_regimes = dict(_ar)
        elif isinstance(_ar, list):
            allowed_regimes = list(_ar)
        else:
            allowed_regimes = list(_DEFAULT_ALLOWED_REGIMES)

        return cls(
            rules=list(raw.get("rules") or []),
            allowed_regimes=allowed_regimes,
            allowed_sides=list(
                raw.get("allowed_sides") or list(_DEFAULT_ALLOWED_SIDES)
            ),
            side_mask=dict(raw.get("side_mask") or {}),
            reject_neutral=bool(raw.get("reject_neutral", False)),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "RegimeConfig":
        """从 regime.yaml 加载（缺失文件 → 全开放默认值）"""
        if not path.exists():
            return cls()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.from_mapping(raw)

    def evaluate(
        self,
        features: Dict[str, Any],
        *,
        latch_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """评估 regime 阈值规则；reject_reason 以 'regime_' 前缀替代 'prefilter_'"""
        passed, reason = super().evaluate(features, latch_id=latch_id)
        if not passed and reason and reason.startswith("prefilter_"):
            reason = "regime_" + reason[len("prefilter_") :]
        return passed, reason

    def classify(self, features: Dict[str, Any]) -> str:
        """将当前 bar 特征分类到 regime 标签。

        优先匹配 labeled regimes (new schema)，回退到 flat rules (old schema)。
        返回 'bull' / 'bear' / 'neutral'，默认返回 'neutral'。
        """
        # ── New schema: labeled regimes with per-label rules ──
        _ar = self.allowed_regimes
        if isinstance(_ar, dict) and _ar:
            for label, cfg in _ar.items():
                if not isinstance(cfg, dict):
                    continue
                rules = cfg.get("rules") or []
                if not rules:
                    continue
                match_mode = str(cfg.get("match", "all")).strip().lower()
                if match_mode == "any":
                    for rule in rules:
                        if self._check_single(rule, features):
                            return str(label)
                else:  # "all" — default
                    if all(self._check_single(r, features) for r in rules):
                        return str(label)
            return "neutral"

        # ── Old schema: flat list — can't distinguish bull/bear, return generic ──
        return "neutral"

    def classify_or_default(
        self, features: Dict[str, Any], default: str = "neutral"
    ) -> str:
        """classify() with fallback for empty/missing features."""
        if not features or not self.allowed_regimes:
            return default
        return self.classify(features)

    def allows_classified_label(self, label: str) -> bool:
        """Entry allow for ``classify()`` label. Neutral denied only if ``reject_neutral``."""
        lab = str(label or "neutral").strip().lower()
        if self.reject_neutral and lab == "neutral":
            return False
        return True

    def allows_side(self, direction: int) -> bool:
        """direction: +1=long / -1=short / 0=neutral。0 视为无方向，不拦截。"""
        if direction > 0:
            return "long" in self.allowed_sides
        if direction < 0:
            return "short" in self.allowed_sides
        return True

    def allows_side_for_bar(self, direction: int, features: Dict[str, Any]) -> bool:
        """Global allowed_sides plus optional EMA/slope side_mask clauses."""
        if not self.allows_side(direction):
            return False
        mask = self.side_mask or {}
        if not bool(mask.get("enabled", False)):
            return True
        if direction > 0:
            when = mask.get("long_when") or {}
        elif direction < 0:
            when = mask.get("short_when") or {}
        else:
            return True
        if not when:
            return True
        return _evaluate_when_clause(when, features, quantiles=None)

    @property
    def is_empty(self) -> bool:
        """所有字段为默认值（无规则、无方向限制）

        兼容 labeled regime schema：dict 型 allowed_regimes 有 per-label rules 时不算空。
        """
        _ar = self.allowed_regimes
        if isinstance(_ar, dict):
            for _cfg in _ar.values():
                if isinstance(_cfg, dict) and _cfg.get("rules"):
                    return False
            # dict with no per-label rules → treat as empty
        if self.reject_neutral:
            # Flag alone must keep regime layer active (allows_classified_label).
            return False
        return (
            not self.rules
            and (
                (
                    isinstance(_ar, (list, tuple))
                    and tuple(_ar) == _DEFAULT_ALLOWED_REGIMES
                )
                or (
                    isinstance(_ar, dict)
                    and not any(
                        isinstance(c, dict) and c.get("rules") for c in _ar.values()
                    )
                )
            )
            and tuple(self.allowed_sides) == _DEFAULT_ALLOWED_SIDES
        )


# =============================================================================
# Strategy Archetype
# =============================================================================


@dataclass
class StrategyArchetype:
    """策略 Archetype - 组合 Regime / Prefilter / Gate / Evidence / Execution 配置。

    Layer order at runtime（live + vector backtest 一致）:
        Regime → Prefilter → Direction → Gate → Entry filters → Evidence → Execution
    """

    name: str
    gate: GateConfig
    evidence: EvidenceConfig
    execution: ExecutionConfig
    prefilter: PrefilterConfig = field(default_factory=PrefilterConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)

    # ==========================================================================
    # 向后兼容属性 (兼容旧的 ExecutionArchetype 接口)
    # ==========================================================================

    @property
    def gate_rules(self) -> Dict[str, Any]:
        """兼容旧接口：返回 when_then_rules 格式"""
        rules = []
        for r in self.gate.all_rules:
            rules.append(
                {
                    "id": r.id,
                    "phase": r.phase,
                    "priority": r.priority,
                    "reason": r.reason,
                    "when": r.when,
                    "then": r.then,
                }
            )
        return {
            "when_then_rules": rules,
            "default_action": "allow",
        }

    @property
    def direction_policy(self) -> Dict[str, Any]:
        """兼容旧接口：返回 direction_policy"""
        return self.execution.raw.get(
            "direction_policy",
            {
                "direction_source": self.execution.direction_source,
                "structure_direction": {
                    "method": self.execution.direction_method,
                    "lookback_bars": self.execution.direction_lookback_bars,
                    "min_consistency": self.execution.direction_min_consistency,
                },
            },
        )

    @property
    def execution_constraints(self) -> Dict[str, Any]:
        """兼容旧接口：返回 execution_constraints"""
        return self.execution.raw.get(
            "execution_constraints",
            {
                "allow_add_on": self.execution.allow_add_on,
                "min_order_interval_minutes": self.execution.min_order_interval_minutes,
                "fixed_rr": {
                    "stop_loss_r": self.execution.stop_loss_r,
                    "take_profit_r": self.execution.take_profit_r,
                    "max_holding_bars": self.execution.max_holding_bars,
                },
            },
        )

    @property
    def when_then_rules(self) -> List[Dict[str, Any]]:
        """兼容旧接口：返回 when_then_rules 列表"""
        return self.gate_rules.get("when_then_rules", [])

    @property
    def default_action(self) -> str:
        """兼容旧接口：返回默认动作"""
        return self.gate_rules.get("default_action", "allow")

    @property
    def evidence_rules(self) -> List[Dict[str, Any]]:
        """兼容旧接口：返回 evidence_rules 格式"""
        # 返回空列表，因为新架构使用 compute_evidence_score
        return []

    def apply_gate(
        self,
        features: Dict[str, Any],
        quantiles: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, List[str], float]:
        """
        应用 Gate 规则

        Args:
            features: 特征值字典
            quantiles: 预留参数；live/research gate 不再支持运行时 quantile_* 条件

        Returns:
            (passed, deny_reasons, cumulative_weight)
            - passed: 是否通过 Gate
            - deny_reasons: 触发的 deny 规则 tag 列表
            - cumulative_weight: 始终返回 1.0（保持接口兼容）
        """
        passed, deny_reasons = evaluate_gate_rules(self.gate, features, quantiles)
        return passed, deny_reasons, (1.0 if passed else 0.0)

    def compute_evidence_score(
        self,
        features: Dict[str, Any],
        quantiles: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算 Evidence 综合评分

        Args:
            features: 特征值字典
            quantiles: 分位数查找表

        Returns:
            (composite_score, {feature_id: score})
        """
        return self.evidence.compute_composite_score(features, quantiles or {})


# =============================================================================
# 条件评估
# =============================================================================


def evaluate_gate_rules(
    gate: "GateConfig",
    features: Dict[str, Any],
    quantiles: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    """Evaluate a ``GateConfig`` against a feature dict (engine-agnostic).

    Shared by ``StrategyArchetype.apply_gate`` (TPC/BPC/ME/SRB) and any other
    engine that loads ``archetypes/gate.yaml`` directly (e.g. chop_grid), so gate
    semantics (``when``/``then: {action: deny}``, phase ordering, ``disabled``)
    stay identical across archetypes.

    Returns ``(passed, deny_reasons)``; first matching non-disabled rule with
    ``action: deny`` short-circuits.
    """
    deny_reasons: List[str] = []
    for rule in gate.all_rules:
        if rule.disabled:
            continue
        if _evaluate_when_clause(rule.when, features, quantiles):
            action = rule.then.get("action", "deny")
            if action == "deny":
                deny_reasons.append(rule.tag)
                return False, deny_reasons
    return True, deny_reasons


def _evaluate_when_clause(
    when: Dict[str, Any],
    features: Dict[str, Any],
    quantiles: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    评估 when 子句

    支持的格式：
    - {feature: {value_lt: 0.5}}
    - {all_of: [...]}
    - {any_of: [...]}
    """
    if not when:
        return False

    # all_of
    if "all_of" in when:
        conditions = when["all_of"]
        min_matches = when.get("min_matches", len(conditions))
        matches = sum(
            1 for c in conditions if _evaluate_when_clause(c, features, quantiles)
        )
        return matches >= min_matches

    # any_of
    if "any_of" in when:
        conditions = when["any_of"]
        min_matches = when.get("min_matches", 1)
        matches = sum(
            1 for c in conditions if _evaluate_when_clause(c, features, quantiles)
        )
        return matches >= min_matches

    # 单个条件: {feature: {op: value}}
    for key, cond in when.items():
        if key in ("all_of", "any_of", "min_matches"):
            continue

        if not isinstance(cond, dict):
            continue

        value = features.get(key)
        if value is None:
            # on_missing 处理: 默认 error，特征缺失必须立刻暴露
            on_missing = cond.get("on_missing", "error")
            if on_missing == "true":
                return True
            elif on_missing == "false":
                return False
            else:  # "error" 或其他
                raise ValueError(
                    f"Gate/prefilter feature '{key}' is missing from features dict. "
                    f"Available keys ({len(features)}): {sorted(features.keys())[:20]}..."
                )

        try:
            value = float(value)
        except (TypeError, ValueError):
            return False

        # NaN 值无法参与有意义的比较 → 视为条件不满足
        import math

        if math.isnan(value):
            return False

        # 直接阈值比较 (value_le/value_ge 是 value_lte/value_gte 的别名)
        if "value_lt" in cond:
            if not (value < float(cond["value_lt"])):
                return False
        if "value_lte" in cond:
            if not (value <= float(cond["value_lte"])):
                return False
        if "value_le" in cond:
            if not (value <= float(cond["value_le"])):
                return False
        if "value_gt" in cond:
            if not (value > float(cond["value_gt"])):
                return False
        if "value_gte" in cond:
            if not (value >= float(cond["value_gte"])):
                return False
        if "value_ge" in cond:
            if not (value >= float(cond["value_ge"])):
                return False

    return True


# =============================================================================
# 加载函数
# =============================================================================


def load_strategy_archetype(
    strategy: str,
    strategies_root: str | Path = "config/strategies",
    *,
    gate_path: str | Path | None = None,
    live_layout: bool = False,
) -> StrategyArchetype:
    """
    加载单个策略的 Archetype 配置

    加载策略的完整 Archetype 配置 (含 Prefilter)。

    Args:
        strategy: 策略名 (如 "bpc")
        strategies_root: 策略配置根目录
        gate_path: 自定义 gate YAML 路径 (如 gate_draft.yaml)，
                   默认 None 表示读取 archetypes/gate.yaml
        live_layout: ``True`` 时按实盘树解析（不向 ``bad-candidates/`` 回退）。

    Returns:
        StrategyArchetype 实例
    """
    root = Path(strategies_root)
    pkg = resolve_strategy_package_under_root(
        root,
        strategy,
        allow_bad_candidates=not live_layout,
    )
    arch_dir = pkg / "archetypes"

    if not arch_dir.exists():
        raise FileNotFoundError(f"Archetype directory not found: {arch_dir}")

    # gate_path 优先级: 显式指定 > archetypes/gate.yaml
    effective_gate_path = Path(gate_path) if gate_path else arch_dir / "gate.yaml"

    return StrategyArchetype(
        name=strategy,
        gate=GateConfig.from_yaml(effective_gate_path),
        evidence=EvidenceConfig.from_yaml(arch_dir / "evidence.yaml"),
        execution=ExecutionConfig.from_yaml(arch_dir / "execution.yaml"),
        prefilter=PrefilterConfig.from_yaml(arch_dir / "prefilter.yaml"),
        regime=RegimeConfig.from_yaml(arch_dir / "regime.yaml"),
    )


def load_all_strategy_archetypes(
    strategies_root: str | Path = "config/strategies",
) -> Dict[str, StrategyArchetype]:
    """
    加载所有策略的 Archetype 配置

    Args:
        strategies_root: 策略配置根目录

    Returns:
        {strategy_name: StrategyArchetype}
    """
    root = Path(strategies_root)
    archetypes = {}

    for strategy_dir in root.iterdir():
        if not strategy_dir.is_dir():
            continue

        arch_dir = strategy_dir / "archetypes"
        if not arch_dir.exists():
            continue

        strategy_name = strategy_dir.name
        try:
            archetypes[strategy_name] = load_strategy_archetype(
                strategy_name,
                strategies_root,
            )
        except Exception as e:
            # 加载失败时跳过，打印警告
            import warnings

            warnings.warn(f"Failed to load archetype for {strategy_name}: {e}")

    return archetypes
