"""Scaffold English front-matter onto legacy DECISION.md files.

Re-running 300+ backtests is not how experiments get standardized: data, clocks,
and harnesses have moved. This module only writes *format*:

- ``verdict`` stays empty (never inferred into a declared verdict)
- ``strategy`` / ``harness`` / ``segments`` / ``kill_switch`` come from
  grid yaml, the directory name, and the harness registry

A human or agent still fills ``verdict`` + ``tags`` + ``kpi`` at close.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from src.research.experiment_index import (
    find_decision_file,
    iter_experiment_dirs,
    load_known_segments,
    parse_front_matter,
)
from src.research.harness_registry import normalize_family, required_harness, spec_for

# Longest-first so ``fast_scalp_alts`` wins over ``fast_scalp``.
_STRATEGY_HINTS: Tuple[str, ...] = ("ma_cross",)


@dataclass
class ScaffoldPlan:
    experiment_id: str
    decision_path: Optional[Path]
    action: str  # write | skip | missing
    reason: str
    fields: Dict[str, Any] = field(default_factory=dict)


def _yaml_strategy(path: Path) -> Optional[str]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict) or not data.get("strategy"):
        return None
    raw = str(data["strategy"]).strip()
    # Combo grids list several families; do not stamp a comma-tag as one strategy.
    if "," in raw or " " in raw:
        return None
    return normalize_family(raw)


def _infer_from_name(name: str) -> Optional[str]:
    hay = name.lower()
    for hint in _STRATEGY_HINTS:
        if hint in hay:
            return normalize_family(hint)
    return None


def _infer_strategy(exp_dir: Path) -> Optional[str]:
    """Prefer yaml ``strategy:`` over a directory-name hint."""
    for path in sorted(exp_dir.glob("rd_loop_*.yaml")) + sorted(
        exp_dir.glob("*_grid.yaml")
    ):
        found = _yaml_strategy(path)
        if found:
            return found
    return _infer_from_name(exp_dir.name)


def _segments_from_grids(exp_dir: Path) -> List[str]:
    seen: List[str] = []
    for path in sorted(exp_dir.glob("*_grid.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(data, dict):
            continue
        raw = data.get("segments")
        if not raw:
            raw = (data.get("segment_matrix") or {}).get("segments")
        if isinstance(raw, list):
            for item in raw:
                sid = _segment_id(item)
                if sid and sid not in seen:
                    seen.append(sid)
        else:
            sid = _segment_id(raw)
            if sid and sid not in seen:
                seen.append(sid)
    return seen


def _segment_id(item: Any) -> Optional[str]:
    if isinstance(item, dict) and item.get("id"):
        return str(item["id"]).strip() or None
    if isinstance(item, str) and item.strip() and not item.strip().startswith("{"):
        return item.strip()
    return None


def _kill_switch_hint(exp_dir: Path, decision_text: str) -> bool:
    """Default OFF. Only stamp True on an explicit enabled flag — never 'ks on' prose."""
    del decision_text
    for path in list(exp_dir.glob("*_grid.yaml")) + list(exp_dir.glob("rd_loop_*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(data, dict):
            continue
        ks = data.get("kill_switch")
        if isinstance(ks, dict) and ks.get("enabled") is True:
            return True
        if ks is True:
            return True
    return False


def propose_scaffold(
    exp_dir: Path, *, known_segments: Iterable[str] = ()
) -> ScaffoldPlan:
    decision = find_decision_file(exp_dir)
    if not decision:
        return ScaffoldPlan(exp_dir.name, None, "missing", "no DECISION.md")

    text = decision.read_text(encoding="utf-8", errors="replace")
    existing, _ = parse_front_matter(text)
    if existing:
        if existing.get("verdict"):
            return ScaffoldPlan(
                exp_dir.name, decision, "skip", "already declared", fields=existing
            )
        auto = existing.get("standardization") == "auto-scaffold"
        if not auto and (existing.get("harness") or existing.get("strategy")):
            return ScaffoldPlan(
                exp_dir.name, decision, "skip", "front-matter already present", fields=existing
            )

    strategy = _infer_strategy(exp_dir)
    harness = required_harness(strategy) if strategy and spec_for(strategy) else None

    fields = {
        "topic": exp_dir.name,
        "strategy": strategy,
        "harness": harness,
        "segments": [
            s
            for s in _segments_from_grids(exp_dir)
            if not known_segments or s in set(known_segments)
        ],
        "kill_switch": _kill_switch_hint(exp_dir, text),
        "verdict": None,
        "kpi": {},
        "supersedes": [],
        "tags": [strategy] if strategy and "," not in strategy else [],
        "standardization": "auto-scaffold",
    }
    return ScaffoldPlan(exp_dir.name, decision, "write", "scaffold empty verdict", fields=fields)


def _dump_front_matter(fields: Dict[str, Any]) -> str:
    payload = {
        "topic": fields.get("topic"),
        "strategy": fields.get("strategy"),
        "harness": fields.get("harness"),
        "segments": fields.get("segments") or [],
        "kill_switch": bool(fields.get("kill_switch")),
        "verdict": fields.get("verdict"),
        "kpi": fields.get("kpi") or {},
        "supersedes": fields.get("supersedes") or [],
        "tags": fields.get("tags") or [],
        "standardization": fields.get("standardization") or "auto-scaffold",
    }
    dumped = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    return f"---\n{dumped}---\n"


def apply_scaffold(plan: ScaffoldPlan, *, write: bool) -> bool:
    if plan.action != "write" or not plan.decision_path:
        return False
    original = plan.decision_path.read_text(encoding="utf-8", errors="replace")
    existing, body = parse_front_matter(original)
    if existing and existing.get("verdict"):
        return False
    if existing and existing.get("standardization") != "auto-scaffold":
        if existing.get("harness") or existing.get("strategy"):
            return False
    rest = body if existing else original
    updated = _dump_front_matter(plan.fields) + "\n" + rest.lstrip()
    if write:
        plan.decision_path.write_text(updated, encoding="utf-8")
    return True


def plan_all(
    experiments_root: Path, *, known_segments: Iterable[str] = ()
) -> List[ScaffoldPlan]:
    known = list(known_segments)
    if not known:
        repo = experiments_root.parent.parent
        if (repo / "config").is_dir():
            known = list(load_known_segments(repo))
    return [
        propose_scaffold(p, known_segments=known)
        for p in iter_experiment_dirs(experiments_root)
    ]
