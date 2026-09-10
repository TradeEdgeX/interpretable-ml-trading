"""Fixed experiment scorecard. Lab always renders these slots.

Phase 1 numbers do not promote. Phase 3 (named ``phase3`` here) is the
headline court: CAGR / Calmar / WR / MaxDD / Sharpe.

Do not invent AUC/ICIR when the artifact does not have them. This repo's
Phase 1 default is Spearman IC + label lift, not sklearn AUC.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.research.experiment_gate import find_kpi_artifacts, find_segment_kpi_rows
from src.research.experiment_index import iter_experiment_dirs, read_experiment_meta

PHASE1_KEYS = ("ic", "icir", "auc", "lift_pp")
PHASE3_KEYS = ("cagr", "calmar", "win_rate", "maxdd", "sharpe")

CENSUS_PATH = "config/experiments/SCORECARD_CENSUS.json"


def empty_scorecard() -> Dict[str, Any]:
    return {
        "phase1": {k: None for k in PHASE1_KEYS},
        "phase3": {k: None for k in PHASE3_KEYS},
        "phase3_segments": [],
        "sources": [],
        "can_close": False,
        "note": (
            "phase1 = hypothesis only; phase3 = court KPI. "
            "Empty cell = no artifact, not a fail."
        ),
    }


def _take_floats(raw: Dict[str, Any], aliases: Dict[str, tuple]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {k: None for k in aliases}
    if not isinstance(raw, dict):
        return out
    for dest, keys in aliases.items():
        for key in keys:
            if key in raw and raw[key] is not None:
                try:
                    out[dest] = float(raw[key])
                    break
                except (TypeError, ValueError):
                    continue
    return out


def _harvest_phase1(exp_dir: Path) -> tuple[Dict[str, Optional[float]], List[str]]:
    aliases = {
        "ic": ("ic", "spearman_ic", "ic_mean"),
        "icir": ("icir", "ic_ir"),
        "auc": ("auc",),
        "lift_pp": ("lift_pp", "label_lift_pp", "delta_pp"),
    }
    sources: List[str] = []
    got = {k: None for k in PHASE1_KEYS}  # type: Dict[str, Optional[float]]
    candidates: List[Path] = []
    for pattern in ("quick_scan/*.json", "*ic*.json", "*scan*.json"):
        candidates.extend(exp_dir.glob(pattern))
    for path in candidates[:12]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "summary" in data and isinstance(data["summary"], dict):
            data = {**data, **data["summary"]}
        if not isinstance(data, dict):
            continue
        part = _take_floats(data, aliases)
        if any(v is not None for v in part.values()):
            for k, v in part.items():
                if v is not None and got[k] is None:
                    got[k] = v
            sources.append(path.name)
        if all(v is not None for v in got.values()):
            break
    return got, sources


def harvest_scorecard(
    exp_dir: Path,
    *,
    repo_root: Path,
    declared_kpi: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fill slots from files already on disk. Never re-runs a backtest."""
    card = empty_scorecard()
    p1, p1_src = _harvest_phase1(exp_dir)
    card["phase1"] = p1
    files, p3 = find_kpi_artifacts(exp_dir, repo_root=repo_root)
    for k in PHASE3_KEYS:
        if k in p3:
            card["phase3"][k] = p3[k]
    card["phase3_segments"] = find_segment_kpi_rows(exp_dir, repo_root=repo_root)
    if declared_kpi:
        for k in PHASE3_KEYS:
            if card["phase3"].get(k) is None and declared_kpi.get(k) is not None:
                try:
                    card["phase3"][k] = float(declared_kpi[k])
                except (TypeError, ValueError):
                    continue
        if any(card["phase3"].get(k) is not None for k in PHASE3_KEYS):
            card["sources"].append("front-matter kpi")
    card["sources"] = p1_src + [str(p.name) for p in files[:8]] + list(card["sources"])
    card["can_close"] = all(card["phase3"].get(k) is not None for k in PHASE3_KEYS)
    return card


def _bucket(card: Dict[str, Any]) -> str:
    p1 = any(card["phase1"].get(k) is not None for k in PHASE1_KEYS)
    p3_n = sum(card["phase3"].get(k) is not None for k in PHASE3_KEYS)
    if card.get("can_close"):
        return "phase3_complete" if not p1 else "both_complete"
    if p3_n and p1:
        return "both_partial"
    if p3_n:
        return "phase3_partial"
    if p1:
        return "phase1_only"
    return "empty"


def build_scorecard_census(
    *, repo_root: Path, experiments_root: Optional[Path] = None
) -> Dict[str, Any]:
    """Harvest every indexed folder. Does not re-run anything."""
    root = experiments_root or (repo_root / "config" / "experiments")
    rows: List[Dict[str, Any]] = []
    buckets: Dict[str, int] = {}
    by_class: Dict[str, Dict[str, int]] = {}
    filled_slots: Dict[str, int] = {k: 0 for k in PHASE1_KEYS + PHASE3_KEYS}

    for exp_dir in iter_experiment_dirs(root):
        meta = read_experiment_meta(exp_dir)
        card = harvest_scorecard(
            exp_dir, repo_root=repo_root, declared_kpi=dict(meta.kpi)
        )
        bucket = _bucket(card)
        buckets[bucket] = buckets.get(bucket, 0) + 1
        cls = meta.record_class
        by_class.setdefault(cls, {})
        by_class[cls][bucket] = by_class[cls].get(bucket, 0) + 1
        for k in PHASE1_KEYS:
            if card["phase1"].get(k) is not None:
                filled_slots[k] += 1
        for k in PHASE3_KEYS:
            if card["phase3"].get(k) is not None:
                filled_slots[k] += 1
        rows.append(
            {
                "id": exp_dir.name,
                "record_class": cls,
                "strategy": meta.strategy,
                "harness": meta.harness,
                "verdict": meta.verdict,
                "verdict_source": meta.verdict_source,
                "bucket": bucket,
                "can_close": card["can_close"],
                "phase1": card["phase1"],
                "phase3": card["phase3"],
                "sources": card["sources"],
            }
        )

    empty_legacy = [
        r["id"]
        for r in rows
        if r["record_class"] == "legacy" and r["bucket"] == "empty"
    ]
    numbered_not_trusted = [
        r["id"]
        for r in rows
        if r["record_class"] != "trusted" and r["bucket"] != "empty"
    ]
    return {
        "count": len(rows),
        "buckets": dict(sorted(buckets.items())),
        "by_record_class": {k: dict(sorted(v.items())) for k, v in sorted(by_class.items())},
        "filled_slots": filled_slots,
        "empty_legacy": empty_legacy,
        "numbered_not_trusted": numbered_not_trusted,
        "rows": rows,
    }
