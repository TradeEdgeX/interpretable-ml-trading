"""Open a new court directory that replays one archive experiment.

Never writes verdict on the parent folder. Never starts a backtest unless
``--run`` is passed for a single replayable id.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.research.court_run import copy_parent_court_grid, list_court_grids
from src.research.review import review_experiment


def replay_topic(parent_id: str, *, today: Optional[str] = None) -> str:
    date = today or datetime.now(timezone.utc).strftime("%Y%m%d")
    stem = parent_id
    if len(stem) > 8 and stem[8] == "_":
        stem = stem[9:]
    return f"{date}_replay_{stem}"


def scaffold_replay(
    parent_id: str,
    *,
    repo_root: Path,
    today: Optional[str] = None,
) -> Path:
    parent = repo_root / "config" / "experiments" / parent_id
    if not parent.is_dir():
        raise FileNotFoundError(f"unknown experiment: {parent_id}")
    review = review_experiment(parent, repo_root=repo_root)
    if review.action == "do_not_replay":
        raise RuntimeError(review.skip_reason or "do not replay")
    if review.action == "skip":
        raise RuntimeError(review.skip_reason or "skip trusted close")

    topic = replay_topic(parent_id, today=today)
    dest = repo_root / "config" / "experiments" / topic
    dest.mkdir(parents=True, exist_ok=True)
    family = review.family or "unknown"
    harness = review.required_harness or review.declared_harness or "event_backtest"
    copied = copy_parent_court_grid(parent, dest, new_id=topic)
    if copied is not None:
        run_block = f"mlbot research run {topic}"
    elif list_court_grids(parent):
        run_block = (
            f"Parent has multiple grids; copy one *_grid.yaml then "
            f"`mlbot research run {topic} --grid <file>`"
        )
    else:
        run_block = (
            f"No court *_grid.yaml. Write one, then `mlbot research run {topic}`. "
            f"Do not add run_grid.py."
        )
    decision = f"""---
topic: {topic}
strategy: {family}
harness: {harness}
segments: [bear_2022, bull_2023_2024, recent_range_to_bear]
kill_switch: false
verdict:
kpi: {{}}
supersedes: []
replays: [{parent_id}]
tags: [replay]
---

# DECISION — replay of `{parent_id}`

Parent archive is unchanged. New numbers belong here only.
Do not type `verdict`. Use `mlbot research close {topic} --declare` after the run.

## Parent review

```json
{json.dumps(review.as_dict(), indent=2, ensure_ascii=False)}
```

## Run (kill switch OFF)

```
{run_block}
```
"""
    (dest / "DECISION.md").write_text(decision, encoding="utf-8")
    (dest / "README.md").write_text(
        f"# Replay {parent_id}\n\nNew court for a reliability replay. "
        f"Parent: `config/experiments/{parent_id}`.\n",
        encoding="utf-8",
    )
    (dest / "PARENT_REVIEW.json").write_text(
        json.dumps(review.as_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return dest
