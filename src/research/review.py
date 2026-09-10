"""Paper review of an archive experiment before any replay.

Replay never writes ``verdict`` on the old folder. A new court directory
is the only place new numbers may land.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.research.experiment_index import (
    iter_experiment_dirs,
    load_known_segments,
    read_experiment_meta,
)
from src.research.harness_registry import harness_mismatch_issue, spec_for
from src.research.scorecard import harvest_scorecard

REVIEW_QUEUE_PATH = "config/experiments/REPLAY_QUEUE.json"

_CHOP_VACATE = (
    "chop_grid product retired; open-bar timeline VACATE. "
    "Do not replay to recover old CAGR."
)


@dataclass
class ReviewRow:
    experiment_id: str
    record_class: str
    family: Optional[str]
    declared_harness: Optional[str]
    required_harness: Optional[str]
    kill_switch: Optional[bool]
    flags: List[str] = field(default_factory=list)
    replayable: bool = False
    action: str = "paper_only"
    run_hint: Optional[str] = None
    skip_reason: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def review_experiment(exp_dir: Path, *, repo_root: Path) -> ReviewRow:
    meta = read_experiment_meta(
        exp_dir, known_segments=load_known_segments(repo_root)
    )
    spec = spec_for(meta.strategy or "") if meta.strategy else None
    flags: List[str] = []
    if meta.record_class == "trusted":
        flags.append("trusted_closed")
    if meta.verdict_source == "inferred":
        flags.append("inferred_verdict_not_court")
    if meta.verdict_source == "missing":
        flags.append("no_declared_verdict")
    if not meta.strategy:
        flags.append("no_family")
    if not meta.harness:
        flags.append("no_harness")
    mismatch = harness_mismatch_issue(meta.strategy, meta.harness)
    if mismatch:
        flags.append("harness_mismatch")
    if meta.kill_switch is True:
        flags.append("kill_switch_on")
    elif meta.kill_switch is None:
        flags.append("kill_switch_unknown")

    family = spec.family if spec else meta.strategy
    required = spec.harness if spec else None
    if family == "chop_grid":
        flags.append("chop_grid_retired")
    if family == "rolling_trend" and meta.harness == "event_backtest":
        flags.append("rolling_via_event_backtest")

    card = harvest_scorecard(exp_dir, repo_root=repo_root, declared_kpi=dict(meta.kpi))
    p3_n = sum(v is not None for v in card["phase3"].values())
    p1_n = sum(v is not None for v in card["phase1"].values())
    if p3_n == 0 and p1_n == 0:
        flags.append("empty_scorecard")
    elif not card["can_close"]:
        flags.append("scorecard_partial")

    action = "paper_only"
    replayable = False
    skip: Optional[str] = None
    hint: Optional[str] = spec.entry if spec else None

    if meta.record_class == "trusted":
        skip = "already declared; restated, do not rescan"
        action = "skip"
    elif family == "chop_grid":
        skip = _CHOP_VACATE
        action = "do_not_replay"
    elif family == "rolling_trend" and meta.harness == "event_backtest":
        skip = "old numbers used the wrong harness; replay only with trend_rolling_simulate"
        action = "replay_correct_harness"
        replayable = True
        hint = spec.entry if spec else hint
    elif spec is None:
        skip = "unknown family — paper review only"
        action = "paper_only"
    else:
        replayable = True
        action = "replay_new_court"
        hint = spec.entry

    return ReviewRow(
        experiment_id=exp_dir.name,
        record_class=meta.record_class,
        family=family,
        declared_harness=meta.harness,
        required_harness=required,
        kill_switch=meta.kill_switch,
        flags=flags,
        replayable=replayable,
        action=action,
        run_hint=hint,
        skip_reason=skip,
    )


def build_review_queue(*, repo_root: Path) -> Dict[str, Any]:
    root = repo_root / "config" / "experiments"
    rows = [
        review_experiment(p, repo_root=repo_root) for p in iter_experiment_dirs(root)
    ]
    by_action: Dict[str, int] = {}
    for row in rows:
        by_action[row.action] = by_action.get(row.action, 0) + 1
    suggested = [
        r.experiment_id
        for r in rows
        if r.action in {"replay_new_court", "replay_correct_harness"}
        and "scorecard_partial" in r.flags
    ]
    suggested += [
        r.experiment_id
        for r in rows
        if r.action == "replay_new_court"
        and r.experiment_id not in suggested
        and "empty_scorecard" in r.flags
        and r.family
    ]
    return {
        "count": len(rows),
        "by_action": dict(sorted(by_action.items())),
        "suggested_first": suggested[:40],
        "note": (
            "Replay one id at a time into a new court directory. "
            "Do not overwrite the archive verdict. Do not batch --run."
        ),
        "rows": [r.as_dict() for r in rows],
    }
