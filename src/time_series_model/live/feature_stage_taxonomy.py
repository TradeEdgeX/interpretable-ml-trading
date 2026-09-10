"""Feature column taxonomy for console UI: account layer × strategy × pipeline stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from time_series_model.live.live_feature_plan import (
    _extract_features_from_direction,
    _extract_features_from_entry_filters,
    _extract_features_from_evidence,
    _extract_features_from_execution,
    _extract_features_from_gate,
    _extract_features_from_prefilter,
    _load_yaml,
)

from src.config.regime_layer import (
    extract_features_from_multileg_regime as _extract_features_from_multileg_regime,
    multileg_regime_section,
)

# Canonical alias map lives in src.features.semantic_chop; re-exported here for
# console services (strategy_stage_regions) that read runtime aliases by column.
from src.features.semantic_chop import (
    _MULTILEG_RUNTIME_ALIASES as _MULTILEG_RUNTIME_ALIASES,
)

CONSOLE_STRATEGIES: Tuple[Dict[str, str], ...] = (
    {"id": "tpc", "account_layer": "trend", "title": "TPC"},
    {"id": "bpc", "account_layer": "trend", "title": "BPC"},
    {"id": "me", "account_layer": "trend", "title": "ME"},
    {"id": "srb", "account_layer": "trend", "title": "SRB"},
    {"id": "largebar_fade", "account_layer": "sa", "title": "largebar_fade"},
    # Legacy package / funnel ids (renamed → largebar_fade 2026-08-05).
    {"id": "chop_largebar_fade", "account_layer": "sa", "title": "largebar_fade"},
    {"id": "fade", "account_layer": "sa", "title": "largebar_fade"},
    # Aux (constitution aux_strategies) — CMS trade-map; not LivePCM / OMS.
    {"id": "sr_confirm", "account_layer": "trend", "title": "SRC"},
    {"id": "sr_fade", "account_layer": "trend", "title": "SRF"},
    {"id": "spot_accum_simple", "account_layer": "spot", "title": "spot_accum_simple"},
    {"id": "chop_grid", "account_layer": "multi_leg", "title": "Chop Grid"},
    # 8h sleeve of same-account dual (2h∥8h); CMS merges with chop_grid for display.
    {"id": "chop_grid_c3b", "account_layer": "multi_leg", "title": "Chop Grid 8h"},
    {"id": "trend_scalp", "account_layer": "multi_leg", "title": "Trend Scalp"},
)

STAGE_ORDER: Tuple[str, ...] = (
    "regime",
    "prefilter",
    "direction",
    "gate",
    "entry",
    "evidence",
    "execution",
)

STAGE_LABELS: Dict[str, str] = {
    "prefilter": "Prefilter",
    "direction": "Direction",
    "gate": "Gate",
    "entry": "Entry",
    "evidence": "Evidence",
    "regime": "Regime",
    "execution": "Execution",
}

ACCOUNT_LAYER_LABELS: Dict[str, str] = {
    "trend": "B·趋势",
    "sa": "SA·Fade",
    "spot": "A·现货",
    "multi_leg": "C·多腿",
}

# Multileg engines read these aliases at runtime (not always in YAML rules).
# Canonical alias map lives in src.features.semantic_chop.multileg_feature_aliases.


def _extract_features_from_rule_block(rule: Any) -> Set[str]:
    out: Set[str] = set()
    if isinstance(rule, dict):
        feat = rule.get("feature")
        if feat:
            out.add(str(feat))
        for key in ("all_of", "any_of", "rules"):
            block = rule.get(key)
            if isinstance(block, list):
                for sub in block:
                    out |= _extract_features_from_rule_block(sub)
    elif isinstance(rule, list):
        for sub in rule:
            out |= _extract_features_from_rule_block(sub)
    return out


def _extract_features_from_prefilter_full(cfg: Dict[str, Any]) -> Set[str]:
    out = _extract_features_from_prefilter(cfg)
    rules = cfg.get("rules")
    if isinstance(rules, list):
        for rule in rules:
            out |= _extract_features_from_rule_block(rule)
    return out


def _extract_features_from_execution_full(cfg: Dict[str, Any]) -> Set[str]:
    out = _extract_features_from_execution(cfg)
    rf = cfg.get("regime_feature")
    if isinstance(rf, str) and rf.strip():
        out.add(rf.strip())
    return out


def extract_strategy_stage_columns(archetypes_dir: Path) -> Dict[str, List[str]]:
    """Map pipeline stage -> sorted feature columns for one strategy archetypes dir."""
    d = Path(archetypes_dir)
    stages: Dict[str, Set[str]] = {s: set() for s in STAGE_ORDER}

    pre_path = d / "prefilter.yaml"
    if pre_path.is_file():
        pre = _load_yaml(pre_path)
        stages["prefilter"] |= _extract_features_from_prefilter_full(pre)
        # 多 leg 引擎仍在 prefilter.yaml 内嵌 regime: block (different schema)
        stages["regime"] |= _extract_features_from_multileg_regime(pre)

    # B-system trend / spot 单 leg：独立 archetypes/regime.yaml (rules schema 与 prefilter 一致)
    regime_path = d / "regime.yaml"
    if regime_path.is_file():
        regime_raw = _load_yaml(regime_path)
        stages["regime"] |= _extract_features_from_prefilter_full(regime_raw)
        multileg = multileg_regime_section(regime_raw)
        if multileg:
            stages["regime"] |= _extract_features_from_multileg_regime(
                {"regime": multileg}
            )

    gate_path = d / "gate.yaml"
    if gate_path.is_file():
        stages["gate"] |= _extract_features_from_gate(_load_yaml(gate_path))

    dir_path = d / "direction.yaml"
    if dir_path.is_file():
        stages["direction"] |= _extract_features_from_direction(_load_yaml(dir_path))

    ef_path = d / "entry_filters.yaml"
    if ef_path.is_file():
        stages["entry"] |= _extract_features_from_entry_filters(_load_yaml(ef_path))

    ev_path = d / "evidence.yaml"
    if ev_path.is_file():
        stages["evidence"] |= _extract_features_from_evidence(_load_yaml(ev_path))

    ex_path = d / "execution.yaml"
    if ex_path.is_file():
        stages["execution"] |= _extract_features_from_execution_full(
            _load_yaml(ex_path)
        )

    return {k: sorted(v) for k, v in stages.items() if v}


def build_console_feature_taxonomy(
    strategies_root: str | Path,
    *,
    strategies: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Build taxonomy from locked archetype YAML under config/strategies/<slug>/archetypes/.

    Returns:
        strategies: per-strategy stage column lists
        index: column -> list of usage records (strategy, account_layer, stage, labels)
    """
    root = Path(strategies_root)
    strategies_out: List[Dict[str, Any]] = []
    index: Dict[str, List[Dict[str, str]]] = {}
    registry = list(strategies) if strategies else list(CONSOLE_STRATEGIES)

    for meta in registry:
        sid = meta["id"]
        arch = root / sid / "archetypes"
        if not arch.is_dir():
            continue
        stage_cols = extract_strategy_stage_columns(arch)
        if not any(stage_cols.values()):
            continue
        strategies_out.append(
            {
                "id": sid,
                "account_layer": meta["account_layer"],
                "account_layer_title": ACCOUNT_LAYER_LABELS.get(
                    meta["account_layer"], meta["account_layer"]
                ),
                "title": meta["title"],
                "stages": stage_cols,
            }
        )
        for stage, cols in stage_cols.items():
            stage_title = STAGE_LABELS.get(stage, stage)
            for col in cols:
                rec = {
                    "column": col,
                    "strategy": sid,
                    "strategy_title": meta["title"],
                    "account_layer": meta["account_layer"],
                    "account_layer_title": ACCOUNT_LAYER_LABELS.get(
                        meta["account_layer"], meta["account_layer"]
                    ),
                    "stage": stage,
                    "stage_title": stage_title,
                }
                index.setdefault(col, []).append(rec)

    seen = {s["id"] for s in strategies_out}
    for meta in registry:
        sid = str(meta["id"])
        if sid in seen:
            continue
        strategies_out.append(
            {
                "id": sid,
                "account_layer": meta["account_layer"],
                "account_layer_title": ACCOUNT_LAYER_LABELS.get(
                    meta["account_layer"], meta["account_layer"]
                ),
                "title": meta.get("title") or sid,
                "stages": {},
            }
        )

    return {
        "strategies": strategies_out,
        "index": index,
        "stage_order": list(STAGE_ORDER),
        "stage_labels": dict(STAGE_LABELS),
        "account_layer_labels": dict(ACCOUNT_LAYER_LABELS),
    }
