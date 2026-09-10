"""Single-id court runner: ``*_grid.yaml`` → event_backtest.

``run_grid.py`` in an experiment folder is not the court. Refuse it.
Do not exec the family harness hint. Do not invent variants.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from src.research.experiment_gate import (
    CANONICAL_CRYPTO,
    build_gate_index,
    evaluate_experiment,
    merge_gate_into_index,
)
from src.research.experiment_index import (
    build_index,
    find_decision_file,
    load_known_segments,
    read_experiment_meta,
)
from src.research.harness_registry import normalize_family, spec_for
from src.research.scorecard import CENSUS_PATH, build_scorecard_census

COURT_EXIT = 2
EVENT_BACKTEST_FAMILIES = frozenset({"ma_cross"})
NO_RUN_GRID = (
    "run_grid.py is not the court runner. Write a unique *_grid.yaml "
    "(nested results/<family>/experiments/<id>/<variant>/<segment>/). "
    "Do not add experiment-dir Python."
)


class CourtRunError(RuntimeError):
    """Refused before the harness started."""


def list_court_grids(exp_dir: Path) -> List[Path]:
    if not exp_dir.is_dir():
        return []
    return sorted(
        p
        for p in exp_dir.glob("*_grid.yaml")
        if p.is_file() and not p.name.startswith(".")
    )


def resolve_court_grid(
    exp_dir: Path, *, grid: Optional[Path] = None
) -> Path:
    if grid is not None:
        path = grid if grid.is_absolute() else (exp_dir / grid)
        if not path.is_file():
            raise CourtRunError(f"grid not found: {path}")
        return path
    grids = list_court_grids(exp_dir)
    if not grids:
        extra = ""
        if (exp_dir / "run_grid.py").is_file():
            extra = " " + NO_RUN_GRID
        raise CourtRunError(
            "no *_grid.yaml; write a court grid before run." + extra
        )
    if len(grids) > 1:
        names = ", ".join(p.name for p in grids)
        raise CourtRunError(f"multiple *_grid.yaml ({names}); pass --grid")
    return grids[0]


def convention_results_root(family: str, experiment_id: str) -> str:
    return f"results/{normalize_family(family)}/experiments/{experiment_id}"


def _segment_ids(data: Dict[str, Any]) -> List[str]:
    raw = (data.get("segment_matrix") or {}).get("segments") or data.get("segments")
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, dict) and item.get("id"):
            out.append(str(item["id"]).strip())
    return out


def _variant_output_dirs(data: Dict[str, Any]) -> List[str]:
    variants = (data.get("segment_matrix") or {}).get("variants") or []
    out: List[str] = []
    if isinstance(variants, list):
        for var in variants:
            if isinstance(var, dict) and var.get("output_dir"):
                out.append(str(var["output_dir"]).replace("\\", "/").rstrip("/"))
    top = data.get("output_dir")
    if top:
        out.append(str(top).replace("\\", "/").rstrip("/"))
    return out


def load_court_grid(grid_path: Path) -> Dict[str, Any]:
    try:
        data = yaml.safe_load(grid_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise CourtRunError(f"cannot read grid {grid_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CourtRunError(f"grid {grid_path.name} is not a mapping")
    return data


def validate_court_grid(
    data: Dict[str, Any],
    *,
    family: str,
    experiment_id: str,
) -> None:
    fam = normalize_family(family)
    grid_fam = normalize_family(str(data.get("strategy") or ""))
    if not grid_fam:
        raise CourtRunError("grid missing strategy:")
    if grid_fam != fam:
        raise CourtRunError(
            f"grid strategy {grid_fam!r} does not match family {fam!r}"
        )
    segs = set(_segment_ids(data))
    missing = [s for s in CANONICAL_CRYPTO if s not in segs]
    if missing:
        raise CourtRunError(
            "grid segment_matrix.segments missing: " + ",".join(missing)
        )
    prefix = convention_results_root(fam, experiment_id)
    dirs = _variant_output_dirs(data)
    if not dirs:
        raise CourtRunError("grid has no variant output_dir")
    for raw in dirs:
        if not raw.startswith(prefix):
            raise CourtRunError(
                f"output_dir {raw!r} must start with {prefix!r} "
                f"(nested variant/segment/; not run_grid.py flat dirs)"
            )
    declared = str(data.get("experiment_id") or "").strip()
    if declared and declared != experiment_id:
        raise CourtRunError(
            f"grid experiment_id {declared!r} != folder {experiment_id!r}"
        )


def argv_for_grid(grid_path: Path) -> List[str]:
    return [
        sys.executable,
        "-m",
        "scripts.event_backtest",
        "--variant-grid",
        str(grid_path),
        "--quiet-signal-logs",
    ]


def pythonpath_env(repo_root: Path, env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    out = dict(env if env is not None else os.environ)
    extra = f"{repo_root / 'src'}{os.pathsep}{repo_root / 'scripts'}"
    current = out.get("PYTHONPATH", "")
    out["PYTHONPATH"] = f"{extra}{os.pathsep}{current}" if current else extra
    return out


def check_runnable(
    exp_dir: Path,
    *,
    repo_root: Path,
    force: bool = False,
) -> str:
    meta = read_experiment_meta(
        exp_dir, known_segments=load_known_segments(repo_root)
    )
    family = normalize_family(meta.strategy or "")
    spec = spec_for(family) if family else None
    if spec is None or family not in EVENT_BACKTEST_FAMILIES:
        need = spec.harness if spec else "unknown"
        raise CourtRunError(
            f"v1 run only registered event_backtest families "
            f"(ma_cross); {family or 'unset'} → {need}. Do not exec a hint."
        )
    if spec.harness != "event_backtest":
        raise CourtRunError(
            f"family {family!r} requires {spec.harness!r}, not event_backtest"
        )
    if meta.harness and meta.harness != "event_backtest":
        raise CourtRunError(
            f"DECISION harness {meta.harness!r} is not event_backtest"
        )
    if meta.verdict and not force:
        raise CourtRunError(
            f"{exp_dir.name} already has verdict={meta.verdict}; "
            f"pass --force to remeasure"
        )
    if meta.kill_switch is True:
        raise CourtRunError("kill_switch on — cannot rank edge")
    return family


def rewrite_grid_for_replay(
    data: Dict[str, Any], *, old_id: str, new_id: str
) -> Dict[str, Any]:
    """Rewrite experiment_id and output_dir only. Do not change variants."""
    out = yaml.safe_load(yaml.safe_dump(data)) or {}
    if not isinstance(out, dict):
        return data
    out["experiment_id"] = new_id

    def _fix_dir(value: Any) -> Any:
        if not isinstance(value, str) or old_id not in value:
            return value
        return value.replace(old_id, new_id)

    if "output_dir" in out:
        out["output_dir"] = _fix_dir(out["output_dir"])
    matrix = out.get("segment_matrix")
    if isinstance(matrix, dict):
        variants = matrix.get("variants")
        if isinstance(variants, list):
            for var in variants:
                if isinstance(var, dict) and "output_dir" in var:
                    var["output_dir"] = _fix_dir(var["output_dir"])
    return out


def copy_parent_court_grid(
    parent: Path, dest: Path, *, new_id: str
) -> Optional[Path]:
    grids = list_court_grids(parent)
    if len(grids) != 1:
        return None
    src = grids[0]
    data = load_court_grid(src)
    old_id = str(data.get("experiment_id") or parent.name).strip() or parent.name
    rewritten = rewrite_grid_for_replay(data, old_id=old_id, new_id=new_id)
    dest_grid = dest / src.name
    dest_grid.write_text(
        yaml.safe_dump(rewritten, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return dest_grid


def append_results_cite(decision: Path, cite: str) -> bool:
    text = decision.read_text(encoding="utf-8")
    if cite in text:
        return False
    decision.write_text(text.rstrip() + f"\n\n产物：`{cite}`\n", encoding="utf-8")
    return True


def refresh_indexes(*, repo_root: Path) -> None:
    experiments = repo_root / "config" / "experiments"
    census = build_scorecard_census(repo_root=repo_root)
    census_path = repo_root / CENSUS_PATH
    census_path.parent.mkdir(parents=True, exist_ok=True)

    census["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    census_path.write_text(
        json.dumps(census, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    gate = build_gate_index(repo_root=repo_root, experiments_root=experiments)
    (experiments / "GATE_INDEX.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    index = build_index(repo_root=repo_root, experiments_root=experiments)
    merge_gate_into_index(index, gate)
    (experiments / "EXPERIMENT_INDEX.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _harvest_after_run(
    exp_dir: Path, *, repo_root: Path, family: str, refresh: bool
) -> None:
    cite = convention_results_root(family, exp_dir.name)
    decision = find_decision_file(exp_dir)
    if decision:
        append_results_cite(decision, cite)
    gate = evaluate_experiment(
        exp_dir,
        repo_root=repo_root,
        known_segments=load_known_segments(repo_root),
    )
    print(
        f"{gate.experiment_id}  gate={gate.gate_status}  "
        f"suggested={gate.suggested_verdict or '-'}"
    )
    if refresh:
        refresh_indexes(repo_root=repo_root)


def run_experiment(
    experiment_id: str,
    *,
    repo_root: Path,
    force: bool = False,
    grid: Optional[Path] = None,
    execute: bool = True,
    refresh_index: bool = False,
    runner: Optional[Callable[..., int]] = None,
) -> int:
    exp_dir = repo_root / "config" / "experiments" / experiment_id
    if not exp_dir.is_dir():
        raise CourtRunError(f"unknown experiment: {experiment_id}")
    family = check_runnable(exp_dir, repo_root=repo_root, force=force)
    grid_path = resolve_court_grid(exp_dir, grid=grid)
    data = load_court_grid(grid_path)
    validate_court_grid(data, family=family, experiment_id=experiment_id)
    argv = argv_for_grid(grid_path)
    print(" ".join(argv))
    if not execute:
        print("dry-run: harness not started")
        return 0
    call = runner or subprocess.call
    rc = call(argv, cwd=str(repo_root), env=pythonpath_env(repo_root))
    if rc == 0:
        _harvest_after_run(
            exp_dir, repo_root=repo_root, family=family, refresh=refresh_index
        )
    return int(rc)
