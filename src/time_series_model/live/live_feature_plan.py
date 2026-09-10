from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import yaml

from src.config.regime_layer import (
    extract_features_from_multileg_regime,
    multileg_regime_section,
)


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    obj = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return obj if isinstance(obj, dict) else {}


def _load_yaml_list(path: str | Path) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    obj = yaml.safe_load(p.read_text(encoding="utf-8")) or []
    if isinstance(obj, list):
        return [str(x) for x in obj]
    return []


def _node_to_output_columns(
    *,
    nodes: Iterable[str],
    feature_deps: Dict[str, Any],
) -> Set[str]:
    out: Set[str] = set()
    feats = feature_deps.get("features") or {}
    for node in nodes:
        info = feats.get(str(node))
        if isinstance(info, dict):
            cols = info.get("output_columns") or []
            for c in cols:
                out.add(str(c))
            continue
        # fallback: strip _f
        if str(node).endswith("_f"):
            out.add(str(node)[:-2])
        else:
            out.add(str(node))
    return out


# ---------------------------------------------------------------------------
# Derived (ef_*) feature → source column dependencies
# ---------------------------------------------------------------------------
# DerivedEntryFeatureState 动态计算的 ef_* 特征依赖这些 DAG 特征列，
# 但 archetype YAML 中只引用 ef_* 名称，自动检测无法发现底层依赖。
_DERIVED_FEATURE_SOURCE_DEPS: Dict[str, List[str]] = {
    "ef_liquidity_silence": ["vol_percentile_approx"],
    "ef_vol_regime_shift": ["bb_width_normalized_pct"],
    # ef_consolidation_bars 依赖 bpc_was_in_pullback（已由 bpc_soft_phase_f 产出）
}

# Columns that may appear in research/evidence artifacts but are not produced by
# the live FeatureComputer DAG. Keeping them in live_feature_set only creates
# permanent health-check noise.
_NON_COMPUTED_LIVE_COLUMNS = frozenset({"pred"})

# Non-numeric DAG outputs that may be useful in research frames but are not part
# of the numeric live feature dict consumed by decision rules.
_NON_NUMERIC_LIVE_OUTPUT_COLUMNS = frozenset({"box_regime_label"})


# ---------------------------------------------------------------------------
# Archetypes auto-detect (no NN dependency)
# ---------------------------------------------------------------------------


# Reserved keys in gate 'when' clauses — NOT feature names.
_WHEN_RESERVED = frozenset(
    {"and", "or", "not", "all_of", "any_of", "min_matches", "min_matches_any"}
)


def _extract_features_from_when(when: Any) -> Set[str]:
    """Extract feature column names from a gate-style 'when' clause."""
    out: Set[str] = set()
    if isinstance(when, dict):
        for k, v in when.items():
            if k in _WHEN_RESERVED:
                # Logical operators — recurse into children
                if isinstance(v, list):
                    for item in v:
                        out |= _extract_features_from_when(item)
                elif isinstance(v, dict):
                    out |= _extract_features_from_when(v)
            else:
                # k is a feature name (e.g. bpc_dir_consistency_long)
                out.add(str(k))
    return out


def _extract_features_from_gate(cfg: Dict[str, Any]) -> Set[str]:
    """Extract all feature columns referenced in gate.yaml."""
    features: Set[str] = set()
    for section in ("hard_gates", "guardrails", "system_safety"):
        rules = cfg.get(section)
        if not isinstance(rules, list):
            continue
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            when = rule.get("when")
            if when:
                features |= _extract_features_from_when(when)
    return features


def _extract_features_from_evidence(cfg: Dict[str, Any]) -> Set[str]:
    """Extract all feature columns referenced in evidence.yaml."""
    features: Set[str] = set()
    items = cfg.get("evidence")
    if not isinstance(items, list):
        return features
    for item in items:
        if not isinstance(item, dict):
            continue
        feat = item.get("feature")
        if feat:
            features.add(str(feat))
    return features


def _extract_features_from_entry_filters(cfg: Dict[str, Any]) -> Set[str]:
    """Extract feature columns from *enabled* entry filters.

    Optionally merge ``prefetch_columns`` — columns needed by disabled experimental
    filters so IncrementalFeatureComputer still resolves their feature nodes without
    toggling ``enabled: true``.
    """
    features: Set[str] = set()
    filters = cfg.get("filters")
    if isinstance(filters, list):
        for f in filters:
            if not isinstance(f, dict):
                continue
            if not f.get("enabled", False):
                continue
            for cond in f.get("conditions") or []:
                if isinstance(cond, dict) and cond.get("feature"):
                    features.add(str(cond["feature"]))
    for col in cfg.get("prefetch_columns") or []:
        if col:
            features.add(str(col))
    return features


def _extract_features_from_prefilter(cfg: Dict[str, Any]) -> Set[str]:
    """Extract feature columns referenced in prefilter.yaml rules.

    Handles both flat rules ({feature, operator, value}) and OR groups
    ({any_of: [...]}). Same schema is reused by regime.yaml.
    """
    features: Set[str] = set()
    rules = cfg.get("rules")
    if not isinstance(rules, list):
        return features

    def _walk(rule: Any) -> None:
        if isinstance(rule, dict):
            feat = rule.get("feature")
            if feat:
                features.add(str(feat))
            for sub in rule.get("any_of") or []:
                _walk(sub)
            for sub in rule.get("all_of") or []:
                _walk(sub)
        elif isinstance(rule, list):
            for sub in rule:
                _walk(sub)

    for rule in rules:
        _walk(rule)
    return features


def _extract_features_from_regime_side_mask(cfg: Dict[str, Any]) -> Set[str]:
    """Feature columns referenced in regime.yaml side_mask when-clauses."""
    mask = cfg.get("side_mask")
    if not isinstance(mask, dict) or not mask.get("enabled", False):
        return set()
    out: Set[str] = set()
    for key in ("long_when", "short_when"):
        when = mask.get(key)
        if when:
            out |= _extract_features_from_when(when)
    return out


def _extract_features_from_direction(cfg: Dict[str, Any]) -> Set[str]:
    """Extract feature columns referenced in direction.yaml direction_rules."""
    from src.time_series_model.live.direction_rule_ops import (
        parse_dual_rule,
        parse_signal_match_position_band_rule,
        parse_single_position_band_rule,
    )

    features: Set[str] = set()
    rules = cfg.get("direction_rules")
    if not isinstance(rules, list):
        return features
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        cmp = parse_signal_match_position_band_rule(rule)
        if cmp is not None:
            features.add(cmp["band_feature"])
            for sr in cmp.get("signal_rules") or []:
                if not isinstance(sr, dict):
                    continue
                dr = parse_dual_rule(sr)
                if dr is not None:
                    features.add(str(dr[0]))
                    features.add(str(dr[1]))
                    continue
                b2 = parse_single_position_band_rule(sr)
                if b2 is not None:
                    features.add(str(b2[0]))
                    continue
                f2 = sr.get("feature")
                if f2:
                    features.add(str(f2))
            rsa = cmp.get("require_sign_agreement")
            if isinstance(rsa, dict):
                rf = rsa.get("feature")
                if isinstance(rf, str) and rf.strip():
                    features.add(rf.strip())
            continue
        if rule.get("method") == "dual_position_agree_deadband":
            raw = rule.get("features")
            if isinstance(raw, list):
                for f in raw:
                    if f:
                        features.add(str(f))
            continue
        feat = rule.get("feature")
        if feat:
            features.add(str(feat))
    return features


def extract_features_from_archetypes(
    archetypes_dir: str | Path,
    feature_deps_path: str | Path = "config/feature_dependencies.yaml",
) -> Tuple[Set[str], List[str]]:
    """
    从 archetypes 目录自动提取实盘所需的全部特征。

    扫描 prefilter.yaml / gate.yaml / evidence.yaml / entry_filters.yaml /
    features.yaml（仅 legacy feature_groups），收集引用的特征列并映射为
    feature nodes；再按 feature_dependencies 递归展开依赖节点，确保中间列
    （如 wide_sr_*）进入 live_feature_set 并被 bus 发布。不读 research 全量
    requested_features 池（见 live-feature-plan-dag-not-research-pool）。

    Returns:
        (live_feature_set, live_feature_nodes)
    """
    d = Path(archetypes_dir)
    feature_columns: Set[str] = set()

    # 0. Regime (慢变量数据空间，与 prefilter 同 rules schema，复用 extractor)
    regime_path = d / "regime.yaml"
    if regime_path.exists():
        regime_raw = _load_yaml(regime_path)
        feature_columns |= _extract_features_from_prefilter(regime_raw)
        # labeled regime schema: allowed_regimes.{bull,bear,neutral}.rules
        _ar = regime_raw.get("allowed_regimes") or {}
        if isinstance(_ar, dict):
            for _label_cfg in _ar.values():
                if isinstance(_label_cfg, dict):
                    feature_columns |= _extract_features_from_prefilter(_label_cfg)
        feature_columns |= _extract_features_from_regime_side_mask(regime_raw)
        multileg = multileg_regime_section(regime_raw)
        if multileg:
            feature_columns |= extract_features_from_multileg_regime(
                {"regime": multileg}
            )

    # 1. Prefilter
    prefilter_path = d / "prefilter.yaml"
    if prefilter_path.exists():
        feature_columns |= _extract_features_from_prefilter(_load_yaml(prefilter_path))

    # 2. Gate
    gate_path = d / "gate.yaml"
    if gate_path.exists():
        feature_columns |= _extract_features_from_gate(_load_yaml(gate_path))

    # 3. Evidence
    evidence_path = d / "evidence.yaml"
    if evidence_path.exists():
        feature_columns |= _extract_features_from_evidence(_load_yaml(evidence_path))

    # 4. Entry Filters (enabled only)
    ef_path = d / "entry_filters.yaml"
    if ef_path.exists():
        feature_columns |= _extract_features_from_entry_filters(_load_yaml(ef_path))

    # 5. Direction (方向特征 — 决定交易方向的核心特征)
    dir_path = d / "direction.yaml"
    if dir_path.exists():
        feature_columns |= _extract_features_from_direction(_load_yaml(dir_path))

    # 6. Execution (structural_exit → exit-signal feature nodes)
    exec_path = d / "execution.yaml"
    if exec_path.exists():
        feature_columns |= _extract_features_from_execution(_load_yaml(exec_path))

    # 7. Strategy features.yaml (research contract; event backtest merges baseline nodes)
    strategy_feature_nodes = _feature_nodes_from_strategy_features_yaml(d.parent)
    for node in strategy_feature_nodes:
        feature_columns |= _node_to_output_columns(
            nodes=[node], feature_deps=_load_yaml(feature_deps_path)
        )

    if not feature_columns and not strategy_feature_nodes:
        return set(), []

    feature_columns -= _NON_COMPUTED_LIVE_COLUMNS
    if not feature_columns:
        return set(), []

    # Map columns → feature nodes (pick ONE per column — smallest output set)
    deps = _load_yaml(feature_deps_path)
    feats = deps.get("features") or {}

    # Build col → list of (node_name, output_count)
    col_to_nodes: Dict[str, List[Tuple[str, int]]] = {}
    for node_name, node_cfg in feats.items():
        if not isinstance(node_cfg, dict):
            continue
        out_cols = node_cfg.get("output_columns") or []
        for c in out_cols:
            col_to_nodes.setdefault(str(c), []).append((node_name, len(out_cols)))

    # For each needed column, pick the node with fewest outputs (most specific)
    selected_nodes: Set[str] = set()
    for col in feature_columns:
        candidates = col_to_nodes.get(col)
        if not candidates:
            continue
        # Sort by output_count ascending → pick most specific
        best = min(candidates, key=lambda x: x[1])
        selected_nodes.add(best[0])

    # Build output column set from selected nodes
    live_feature_set = _node_to_output_columns(nodes=selected_nodes, feature_deps=deps)
    # Also include raw archetype columns (some computed inline, e.g. ef_*)
    live_feature_set |= feature_columns

    # Expand ef_* derived features → their source column dependencies
    for col in list(feature_columns):
        src_deps = _DERIVED_FEATURE_SOURCE_DEPS.get(col)
        if src_deps:
            for src_col in src_deps:
                live_feature_set.add(src_col)
                # Also resolve src_col → feature node
                candidates = col_to_nodes.get(src_col)
                if candidates:
                    best = min(candidates, key=lambda x: x[1])
                    selected_nodes.add(best[0])
                    # Add output columns of that node
                    live_feature_set |= _node_to_output_columns(
                        nodes=[best[0]], feature_deps=deps
                    )

    # Always include OHLCV + execution-critical columns
    # atr is needed by pick_atr() for stop loss distance calculation
    # bb_width_normalized_pct is an intermediate for bpc_bb_compression but useful for diagnostics
    live_feature_set |= {"open", "high", "low", "close", "volume", "atr"}
    live_feature_set -= _NON_NUMERIC_LIVE_OUTPUT_COLUMNS

    # Ensure atr_f node is in selected_nodes — atr is always needed for
    # execution (stop-loss sizing) but load_features_from_requested only
    # returns output columns of *requested* features.  Without atr_f in
    # the request list the computed atr column is silently dropped.
    atr_candidates = col_to_nodes.get("atr")
    if atr_candidates and not any(n for n, _ in atr_candidates if n in selected_nodes):
        best = min(atr_candidates, key=lambda x: x[1])
        selected_nodes.add(best[0])

    # Strategy features.yaml + execution structural_exit nodes (not only archetype columns)
    for node in strategy_feature_nodes:
        selected_nodes.add(node)
    exec_path = d / "execution.yaml"
    if exec_path.exists():
        for node in _execution_structural_exit_nodes(_load_yaml(exec_path)):
            selected_nodes.add(node)

    # Expand DAG dependencies into the live plan (compute + publish).
    # Loader strips intermediate outputs unless the dep node is requested;
    # without this, e.g. srb_l3_breakout_window_f runs but wide_sr_* never
    # enter live_feature_set → bus omits them → age_decay stays 0.
    try:
        from src.cli.auto_detect_compute_requirements import (
            resolve_feature_dependencies,
        )

        selected_nodes = resolve_feature_dependencies(selected_nodes, deps)
    except Exception:
        pass

    live_feature_set |= _node_to_output_columns(nodes=selected_nodes, feature_deps=deps)
    live_feature_set |= feature_columns
    live_feature_set -= _NON_NUMERIC_LIVE_OUTPUT_COLUMNS

    # Deduplicated ordered list
    ordered = sorted(selected_nodes)
    return live_feature_set, ordered


# structural_exit enum in execution.yaml → feature node (event/live must compute signal)
_STRUCTURAL_EXIT_FEATURE_NODES: Dict[str, str] = {
    "weekly_macro_cycle": "weekly_macro_cycle_exit_f",
}


def _feature_nodes_from_strategy_features_yaml(strategy_dir: Path) -> List[str]:
    """Feature nodes from legacy ``features.yaml`` ``feature_groups`` blocks.

    NOTE: we intentionally do NOT read ``feature_pipeline.requested_features``.
    That list is the research/training feature *pool* (80–90 nodes incl.
    path-dependent tick features like vpin/wpt/scene) and is far larger than
    what the archetype rules actually consume live. Pulling it wholesale into
    the live plan bloated the primary FeatureComputer (TPC ~42→139 nodes) and
    risked breaching the 2h-close bounded-tail compute budget. The wide_sr fix
    is handled purely by DAG dependency expansion in
    ``extract_features_from_archetypes`` (see ``resolve_feature_dependencies``),
    which only adds the deps of nodes the archetype already selected.
    """
    path = Path(strategy_dir) / "features.yaml"
    if not path.exists():
        return []
    cfg = _load_yaml(path)
    nodes: List[str] = []
    groups = cfg.get("feature_groups") or {}
    if isinstance(groups, dict):
        for group in groups.values():
            if isinstance(group, list):
                nodes.extend(str(x) for x in group if x)
    return nodes


def _execution_structural_exit_nodes(exec_raw: Dict[str, Any]) -> List[str]:
    sl = (
        exec_raw.get("stop_loss") if isinstance(exec_raw.get("stop_loss"), dict) else {}
    )
    key = str(sl.get("structural_exit") or "").strip().lower()
    node = _STRUCTURAL_EXIT_FEATURE_NODES.get(key)
    return [node] if node else []


# Rolling / spot execution.yaml uses boolean flags and indicator aliases that are
# not ``feature:`` rules. Map them so the bus still publishes the columns the
# live engines read (golden-cross + momentum + regime_exit).
_EXECUTION_INDICATOR_ALIASES: Dict[str, str] = {
    "weekly_ema_200_position": "weekly_ema_200_position",
    "ema1200": "ema_1200_position",
    "ema_1200": "ema_1200_position",
    "vwap1200": "macro_tp_vwap_1200_position",
    "vwap_1200": "macro_tp_vwap_1200_position",
}


def _extract_features_from_execution(exec_raw: Dict[str, Any]) -> Set[str]:
    """Columns required by execution.yaml (structural_exit + rolling/spot flags)."""
    deps = _load_yaml("config/feature_dependencies.yaml")
    cols: Set[str] = set()
    for node in _execution_structural_exit_nodes(exec_raw):
        cols |= _node_to_output_columns(nodes=[node], feature_deps=deps)

    entry = exec_raw.get("entry") if isinstance(exec_raw.get("entry"), dict) else {}
    if entry.get("ema1200_cross_above_vwap1200"):
        cols |= {"ema_1200_position", "macro_tp_vwap_1200_position"}
    if entry.get("require_momentum"):
        cols |= {"roc_5", "roc_20"}
    if any(k.startswith("weekly_ema_200_position_") for k in entry):
        cols.add("weekly_ema_200_position")
    bb = entry.get("bb_lower") if isinstance(entry.get("bb_lower"), dict) else {}
    if bb.get("enabled"):
        cols.add("bb_position")

    roll = (
        exec_raw.get("roll_trigger")
        if isinstance(exec_raw.get("roll_trigger"), dict)
        else {}
    )
    if roll.get("near_vwap1200_pct") is not None:
        cols.add("macro_tp_vwap_1200_position")

    reentry = (
        exec_raw.get("reentry") if isinstance(exec_raw.get("reentry"), dict) else {}
    )
    if reentry.get("wave2_require_roc5_positive"):
        cols.add("roc_5")

    signal = exec_raw.get("signal") if isinstance(exec_raw.get("signal"), dict) else {}
    for ind in signal.get("entry_indicators") or []:
        mapped = _EXECUTION_INDICATOR_ALIASES.get(str(ind).strip())
        if mapped:
            cols.add(mapped)

    sl = (
        exec_raw.get("stop_loss") if isinstance(exec_raw.get("stop_loss"), dict) else {}
    )
    regime_exit = (
        sl.get("regime_exit") if isinstance(sl.get("regime_exit"), dict) else {}
    )
    if regime_exit.get("enabled"):
        if any(k.startswith("ema_1200_position_") for k in regime_exit):
            cols.add("ema_1200_position")
        if any(k.startswith("vwap_1200_position_") for k in regime_exit):
            cols.add("macro_tp_vwap_1200_position")
        if any(k.startswith("weekly_ema_200_position_") for k in regime_exit):
            cols.add("weekly_ema_200_position")

    return cols


def load_live_feature_plan(
    *,
    plan_path: str | Path = "config/live/live_feature_plan.yaml",
    feature_deps_path: str | Path = "config/feature_dependencies.yaml",
) -> Set[str]:
    cfg = _load_yaml(plan_path)
    base_plan_path = str(cfg.get("base_feature_plan") or "")
    mode = str(cfg.get("base_feature_plan_mode") or "tiers")
    overlay = cfg.get("overlay") or {}

    base_features: Set[str] = set()
    base_nodes: List[str] = []
    if base_plan_path:
        base = _load_yaml(base_plan_path)
        fp = base.get("feature_plan") or {}
        if mode == "minimal_required_cols":
            base_features = {
                str(x)
                for x in (fp.get("feature_contract") or {}).get(
                    "minimal_required_cols", []
                )
            }
        else:
            tiers = fp.get("tiers_enabled") or []
            tier_files = fp.get("tier_feature_files") or {}
            nodes: List[str] = []
            for t in tiers:
                f = tier_files.get(t)
                if f:
                    nodes.extend(_load_yaml_list(f))
            base_nodes = list(nodes)
            # optional blocks (base plan only)
            blocks = fp.get("optional_blocks_library") or {}
            enabled = fp.get("optional_blocks_enabled") or []
            for b in enabled:
                nodes.extend(list(blocks.get(b) or []))
            deps = _load_yaml(feature_deps_path)
            base_features = _node_to_output_columns(nodes=nodes, feature_deps=deps)

    # Overlay adjustments
    add_features = {str(x) for x in (overlay.get("add_features") or [])}
    add_nodes = [str(x) for x in (overlay.get("add_feature_nodes") or [])]
    add_blocks = [str(x) for x in (overlay.get("add_optional_blocks") or [])]
    drop_features = {str(x) for x in (overlay.get("drop_features") or [])}

    if add_nodes or add_blocks:
        deps = _load_yaml(feature_deps_path)
        nodes = list(add_nodes)
        if add_blocks:
            base = _load_yaml(base_plan_path)
            fp = base.get("feature_plan") or {}
            blocks = fp.get("optional_blocks_library") or {}
            for b in add_blocks:
                nodes.extend(list(blocks.get(b) or []))
        base_nodes.extend(nodes)
        base_features |= _node_to_output_columns(nodes=nodes, feature_deps=deps)

    base_features |= add_features
    base_features -= drop_features

    # Always include raw OHLCV (filled by IncrementalFeatureComputer from bar data)
    base_features |= {"open", "high", "low", "close", "volume"}

    # Merge gate-required columns (same auto-detect as load_live_feature_nodes)
    # so that _want(gate_column) is True and computed gate columns are kept.
    try:
        from src.cli.auto_detect_compute_requirements import (
            extract_required_features_from_execution_archetypes,
            map_features_to_tier_nodes,
            resolve_feature_dependencies,
        )

        deps_path = Path(feature_deps_path)
        project_root = (
            deps_path.resolve().parents[1]
            if deps_path.is_absolute()
            else Path(__file__).resolve().parents[3]
        )
        gate_features = extract_required_features_from_execution_archetypes(
            project_root / "config/strategies/tpc/archetypes/gate.yaml"
        )
        if gate_features:
            base_features |= gate_features  # gate rule keys (e.g. bpc_dir_consistency_long) in case filled by IncrementalFeatureComputer
            deps = _load_yaml(feature_deps_path)
            if deps:
                gate_nodes = map_features_to_tier_nodes(gate_features, deps)
                if gate_nodes:
                    all_gate_nodes = resolve_feature_dependencies(gate_nodes, deps)
                    gate_cols = _node_to_output_columns(
                        nodes=all_gate_nodes, feature_deps=deps
                    )
                    base_features |= gate_cols
    except Exception:
        pass

    return base_features


def load_live_feature_nodes(
    *,
    plan_path: str | Path = "config/live/live_feature_plan.yaml",
    feature_deps_path: str | Path = "config/feature_dependencies.yaml",
) -> List[str]:
    cfg = _load_yaml(plan_path)
    base_plan_path = str(cfg.get("base_feature_plan") or "")
    mode = str(cfg.get("base_feature_plan_mode") or "tiers")
    overlay = cfg.get("overlay") or {}
    nodes: List[str] = []
    if base_plan_path and mode == "tiers":
        base = _load_yaml(base_plan_path)
        fp = base.get("feature_plan") or {}
        tiers = fp.get("tiers_enabled") or []
        tier_files = fp.get("tier_feature_files") or {}
        for t in tiers:
            f = tier_files.get(t)
            if f:
                nodes.extend(_load_yaml_list(f))
        blocks = fp.get("optional_blocks_library") or {}
        enabled = fp.get("optional_blocks_enabled") or []
        for b in enabled:
            nodes.extend(list(blocks.get(b) or []))
    add_nodes = [str(x) for x in (overlay.get("add_feature_nodes") or [])]
    add_blocks = [str(x) for x in (overlay.get("add_optional_blocks") or [])]
    if add_nodes or add_blocks:
        base = _load_yaml(base_plan_path)
        fp = base.get("feature_plan") or {}
        blocks = fp.get("optional_blocks_library") or {}
        nodes.extend(add_nodes)
        for b in add_blocks:
            nodes.extend(list(blocks.get(b) or []))
    # AUTO-DETECT: 自动检测gate规则需要的特征并添加到nodes
    try:
        from src.cli.auto_detect_compute_requirements import (
            extract_required_features_from_execution_archetypes,
            map_features_to_tier_nodes,
            resolve_feature_dependencies,
        )
        from pathlib import Path

        # 获取项目根目录（从feature_deps_path推断）
        deps_path = Path(feature_deps_path)
        if not deps_path.is_absolute():
            # 假设相对于项目根目录
            project_root = (
                Path(__file__).resolve().parents[3]
            )  # src/time_series_model/live -> project root
        else:
            project_root = deps_path.parents[
                1
            ]  # config/feature_dependencies.yaml -> project root

        # 提取gate规则需要的特征列名
        gate_features = extract_required_features_from_execution_archetypes(
            project_root / "config/strategies/tpc/archetypes/gate.yaml"
        )

        if gate_features:
            # 读取feature_dependencies
            deps = _load_yaml(feature_deps_path)
            if deps:
                # 映射到feature nodes
                gate_nodes = map_features_to_tier_nodes(gate_features, deps)

                if gate_nodes:
                    # 递归解析依赖关系
                    all_gate_nodes = resolve_feature_dependencies(gate_nodes, deps)

                    # 添加gate nodes到nodes列表
                    existing_nodes_set = set(nodes)
                    new_gate_nodes = [
                        n for n in all_gate_nodes if n not in existing_nodes_set
                    ]
                    if new_gate_nodes:
                        nodes.extend(new_gate_nodes)
    except Exception:
        # 如果自动检测失败，不影响正常流程
        pass

    # de-dup while preserving order
    seen = set()
    out = []
    for n in nodes:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out
