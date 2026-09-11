"""Read-only experiment cards over ``config/experiments/``."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.lab.paths import experiments_root as experiments_root_path
from src.lab.paths import repo_root as default_repo_root
from src.research.experiment_index import load_known_segments, read_experiment_meta
from src.research.scorecard import (
    CENSUS_PATH,
    PHASE1_KEYS,
    PHASE3_KEYS,
    harvest_scorecard,
)

_DIR_RE = re.compile(r"^(\d{8})(?:_(\d{4}))?_(.+)$")
_STRATEGY_HINTS = ("ma_cross",)
_RESULTS_PATH_RE = re.compile(
    r"`(results/[^`\s]+)`|(?:^|\s)(results/[A-Za-z0-9_./-]+)",
    re.MULTILINE,
)


def _safe_read_text(path: Path, *, limit: int = 200_000) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def _parse_dir_name(name: str) -> Dict[str, Optional[str]]:
    m = _DIR_RE.match(name)
    if not m:
        return {"date": None, "time": None, "topic": name}
    date_s, time_s, topic = m.group(1), m.group(2), m.group(3)
    return {
        "date": f"{date_s[:4]}-{date_s[4:6]}-{date_s[6:8]}",
        "time": time_s,
        "topic": topic,
    }


def _infer_strategy(topic: str, rd_loop_yaml: Optional[Path]) -> Optional[str]:
    topic_l = topic.lower()
    for hint in _STRATEGY_HINTS:
        if (
            hint in topic_l
            or topic_l.startswith(hint + "_")
            or topic_l.endswith("_" + hint)
        ):
            return hint
    if rd_loop_yaml and rd_loop_yaml.is_file():
        try:
            data = yaml.safe_load(rd_loop_yaml.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict) and data.get("strategy"):
                return str(data["strategy"])
        except (OSError, yaml.YAMLError):
            pass
    parts = topic.split("_")
    if parts:
        return parts[0]
    return None


def _find_decision_file(exp_dir: Path) -> Optional[Path]:
    for name in ("DECISION.md",):
        p = exp_dir / name
        if p.is_file():
            return p
    for p in sorted(exp_dir.glob("DECISION*.md")):
        if p.is_file():
            return p
    return None


def _extract_hypothesis(readme: str) -> str:
    if not readme:
        return ""
    lines = readme.splitlines()
    in_hypothesis = False
    collected: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("## 假设") or stripped.lower().startswith(
            "## hypothesis"
        ):
            in_hypothesis = True
            continue
        if in_hypothesis:
            if stripped.startswith("## "):
                break
            if stripped and not stripped.startswith("|"):
                collected.append(stripped)
                if len(collected) >= 3:
                    break
    if collected:
        return " ".join(collected)[:400]
    for line in lines:
        s = line.strip()
        if s.startswith("**") and "目的" in s:
            return re.sub(r"\*\*", "", s)[:400]
        if (
            s
            and not s.startswith("#")
            and not s.startswith("|")
            and not s.startswith("```")
        ):
            if len(s) > 20:
                return s[:400]
    return ""


def _extract_results_links(text: str) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for m in _RESULTS_PATH_RE.finditer(text):
        path = (m.group(1) or m.group(2) or "").strip().rstrip(".,;)")
        if path and path not in seen:
            seen.add(path)
            out.append(path)
    return out


def _load_gate_by_id(repo_root: Path) -> Dict[str, Dict[str, Any]]:
    path = repo_root / "config" / "experiments" / "GATE_INDEX.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for row in data.get("rows") or []:
        if isinstance(row, dict) and row.get("experiment_id"):
            out[str(row["experiment_id"])] = row
    return out


def _court_card(meta: Any, gate: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    trusted = meta.record_class == "trusted"
    kpi = dict(meta.kpi)
    if gate and gate.get("kpi") and not kpi:
        kpi = dict(gate["kpi"])
    return {
        "record_class": meta.record_class,
        "display_verdict": meta.verdict if trusted else None,
        "inferred_verdict": meta.verdict if meta.record_class == "legacy" else None,
        "verdict_source": meta.verdict_source,
        "harness": meta.harness,
        "segments": list(meta.segments),
        "kill_switch": meta.kill_switch,
        "kpi": kpi,
        "gate_status": (gate or {}).get("gate_status"),
        "gate_reasons": (gate or {}).get("reasons") or [],
        "scorecard": None,
    }


def _decision_title(decision_text: str) -> str:
    for line in decision_text.splitlines():
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip()
    return ""


def _scan_experiment_dir(
    exp_dir: Path,
    *,
    repo_root: Path,
    gate_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    if not exp_dir.is_dir():
        return None
    name = exp_dir.name
    if name.startswith("."):
        return None

    parsed = _parse_dir_name(name)
    readme_path = exp_dir / "README.md"
    decision_path = _find_decision_file(exp_dir)
    rd_loops = sorted(exp_dir.glob("rd_loop_*.yaml"))
    grids = sorted(exp_dir.glob("*_grid.yaml"))
    readme_text = _safe_read_text(readme_path) if readme_path.is_file() else ""
    decision_text = _safe_read_text(decision_path) if decision_path else ""

    rd_loop_rel = str(rd_loops[0].relative_to(repo_root)) if rd_loops else None
    strategy = _infer_strategy(
        parsed.get("topic") or name, rd_loops[0] if rd_loops else None
    )

    rel_dir = str(exp_dir.relative_to(repo_root))
    results_links = _extract_results_links(readme_text + "\n" + decision_text)

    for yf in rd_loops[:1]:
        try:
            ydata = yaml.safe_load(yf.read_text(encoding="utf-8")) or {}
            if isinstance(ydata, dict):
                out_dir = ydata.get("output_dir")
                if out_dir:
                    od = str(out_dir).strip()
                    if od and od not in results_links:
                        results_links.insert(0, od)
        except (OSError, yaml.YAMLError):
            pass

    category = "special" if name.startswith("_") else "experiment"
    meta = read_experiment_meta(exp_dir, known_segments=load_known_segments(repo_root))
    gate = (gate_by_id or {}).get(name)
    court = _court_card(meta, gate)

    return {
        "id": name,
        "category": category,
        "record_class": court["record_class"],
        "display_verdict": court["display_verdict"],
        "court": court,
        "date": parsed.get("date"),
        "time": parsed.get("time"),
        "topic": parsed.get("topic"),
        "strategy": meta.strategy or strategy,
        "hypothesis": _extract_hypothesis(readme_text),
        "has_readme": readme_path.is_file(),
        "has_decision": decision_path is not None,
        "verdict": meta.verdict,
        "verdict_source": meta.verdict_source,
        "harness": meta.harness,
        "segments": list(meta.segments),
        "kill_switch": meta.kill_switch,
        "kpi": dict(meta.kpi),
        "meta_issues": list(meta.issues),
        "decision_title": _decision_title(decision_text) if decision_text else "",
        "dir": rel_dir,
        "readme_path": (
            str(readme_path.relative_to(repo_root)) if readme_path.is_file() else None
        ),
        "decision_path": (
            str(decision_path.relative_to(repo_root)) if decision_path else None
        ),
        "rd_loop_yaml": rd_loop_rel,
        "rd_loop_yamls": [str(p.relative_to(repo_root)) for p in rd_loops],
        "grid_yamls": [str(p.relative_to(repo_root)) for p in grids],
        "results_links": results_links,
    }


def _newest_first_key(name: str) -> Tuple[int, str]:
    if _DIR_RE.match(name):
        return (1, name)
    return (0, name)


def _iter_experiment_dirs(experiments_root: Path) -> List[Path]:
    if not experiments_root.is_dir():
        return []
    dirs = [
        p
        for p in experiments_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    ]
    return sorted(dirs, key=lambda p: _newest_first_key(p.name), reverse=True)


def _dir_names_stamp(experiments_root: Path) -> str:
    if not experiments_root.is_dir():
        return ""
    parts: List[str] = []
    for p in experiments_root.iterdir():
        if not p.is_dir() or p.name.startswith("."):
            continue
        newest = p.stat().st_mtime_ns
        for hit in p.glob("DECISION*.md"):
            newest = max(newest, hit.stat().st_mtime_ns)
        readme = p / "README.md"
        if readme.is_file():
            newest = max(newest, readme.stat().st_mtime_ns)
        parts.append(f"{p.name}:{newest}")
    return "\n".join(sorted(parts))


@lru_cache(maxsize=8)
def _cached_scan(
    experiments_root_str: str, repo_root_str: str, stamp: str
) -> Tuple[Dict[str, Any], ...]:
    experiments_root = Path(experiments_root_str)
    repo_root = Path(repo_root_str)
    gate_by_id = _load_gate_by_id(repo_root)
    rows: List[Dict[str, Any]] = []
    for exp_dir in _iter_experiment_dirs(experiments_root):
        if exp_dir.name.startswith("_"):
            continue
        row = _scan_experiment_dir(exp_dir, repo_root=repo_root, gate_by_id=gate_by_id)
        if row:
            rows.append(row)
    return tuple(rows)


def clear_experiments_cache() -> None:
    _cached_scan.cache_clear()


def _census_fill(repo_root: Path) -> Dict[str, Dict[str, Any]]:
    path = repo_root / CENSUS_PATH
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for raw in data.get("rows") or []:
        if not isinstance(raw, dict) or not raw.get("id"):
            continue
        p1 = sum(1 for k in PHASE1_KEYS if (raw.get("phase1") or {}).get(k) is not None)
        p3 = sum(1 for k in PHASE3_KEYS if (raw.get("phase3") or {}).get(k) is not None)
        if raw.get("can_close"):
            label = "齐"
        elif p3:
            label = f"P3 {p3}/5"
        elif p1:
            label = f"P1 {p1}/4"
        else:
            label = "空"
        out[str(raw["id"])] = {
            "bucket": raw.get("bucket") or "empty",
            "phase1_filled": p1,
            "phase3_filled": p3,
            "label": label,
        }
    return out


def _query_matches(q: str, hay: str) -> bool:
    if q in hay:
        return True
    if "_" in q:
        stem = q.split("_", 1)[1]
        if len(stem) >= 6 and stem in hay:
            return True
    tokens = [
        t for t in re.split(r"[_\s/-]+", q) if t and not t.isdigit() and len(t) >= 3
    ]
    return bool(tokens) and all(t in hay for t in tokens)


def list_experiments(
    *,
    strategy: Optional[str] = None,
    q: Optional[str] = None,
    since: Optional[str] = None,
    category: Optional[str] = None,
    experiments_root: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    root = experiments_root or experiments_root_path()
    repo = repo_root or default_repo_root()
    rows = list(
        _cached_scan(
            str(root.resolve()),
            str(repo.resolve()),
            _dir_names_stamp(root),
        )
    )

    strat_f = (strategy or "").strip().lower()
    q_f = (q or "").strip().lower()
    since_f = (since or "").strip()
    cat_f = (category or "").strip().lower()

    fills = _census_fill(repo)
    out: List[Dict[str, Any]] = []
    for row in rows:
        if cat_f and row.get("category", "") != cat_f:
            continue
        if strat_f and (row.get("strategy") or "").lower() != strat_f:
            continue
        if since_f and (row.get("date") or "") < since_f:
            continue
        if q_f:
            hay = " ".join(
                str(row.get(k) or "")
                for k in (
                    "id",
                    "topic",
                    "strategy",
                    "hypothesis",
                    "decision_title",
                    "dir",
                    "verdict",
                    "display_verdict",
                )
            ).lower()
            if not _query_matches(q_f, hay):
                continue
        fill = fills.get(str(row.get("id") or ""))
        out.append({**row, "scorecard_fill": fill} if fill else dict(row))
    return out


def list_strategies(
    experiments_root: Optional[Path] = None, repo_root: Optional[Path] = None
) -> List[str]:
    rows = list_experiments(experiments_root=experiments_root, repo_root=repo_root)
    return sorted({str(r["strategy"]) for r in rows if r.get("strategy")})


def get_experiment(
    experiment_id: str,
    *,
    experiments_root: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    root = experiments_root or experiments_root_path()
    repo = repo_root or default_repo_root()
    exp_dir = (root / experiment_id).resolve()
    try:
        exp_dir.relative_to(root.resolve())
    except ValueError:
        return None
    if not exp_dir.is_dir():
        return None

    base = _scan_experiment_dir(
        exp_dir,
        repo_root=repo,
        gate_by_id=_load_gate_by_id(repo),
    )
    if not base:
        return None

    court = dict(base.get("court") or {})
    court["scorecard"] = harvest_scorecard(
        exp_dir, repo_root=repo, declared_kpi=court.get("kpi") or {}
    )
    base = {**base, "court": court}

    readme_path = exp_dir / "README.md"
    decision_path = _find_decision_file(exp_dir)
    readme_text = _safe_read_text(readme_path) if readme_path.is_file() else ""
    decision_text = _safe_read_text(decision_path) if decision_path else ""

    yaml_snippets: Dict[str, str] = {}
    for rel in base.get("rd_loop_yamls") or []:
        p = repo / rel
        if p.is_file():
            yaml_snippets[rel] = _safe_read_text(p, limit=50_000)
    for rel in base.get("grid_yamls") or []:
        p = repo / rel
        if p.is_file():
            yaml_snippets[rel] = _safe_read_text(p, limit=50_000)

    return {
        **base,
        "readme_text": readme_text,
        "decision_text": decision_text,
        "readme": readme_text,
        "decision": decision_text,
        "results_links": base.get("results_links") or [],
        "artifact_links": base.get("results_links") or [],
        "yaml_snippets": yaml_snippets,
    }


def get_experiment_raw_file(
    experiment_id: str,
    filename: str,
    *,
    experiments_root: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> Optional[Dict[str, str]]:
    allowed_suffixes = (".md",)
    if not filename or filename != Path(filename).name:
        return None
    if not filename.endswith(allowed_suffixes):
        return None
    if filename.startswith("."):
        return None

    root = experiments_root or experiments_root_path()
    repo = repo_root or default_repo_root()
    exp_dir = (root / experiment_id).resolve()
    try:
        exp_dir.relative_to(root.resolve())
    except ValueError:
        return None

    target = (exp_dir / filename).resolve()
    try:
        target.relative_to(exp_dir)
    except ValueError:
        return None
    if not target.is_file():
        return None

    return {
        "filename": filename,
        "path": str(target.relative_to(repo)),
        "content": _safe_read_text(target),
    }
