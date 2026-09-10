"""Program-owned court gate. Agents do not write ``verdict``.

``gate_status`` is computed from front-matter + on-disk artifacts.
``verdict`` is only written by ``mlbot research close --declare`` (promote
needs ``--yes`` and ``court_ok``).

See ``docs/agent/rd_playbook.md``.
"""

from __future__ import annotations

import csv
import json
import os
import re
import statistics
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.research.evidence_staleness import evaluate_staleness
from src.research.experiment_index import (
    ExperimentMeta,
    find_decision_file,
    iter_experiment_dirs,
    load_known_segments,
    read_experiment_meta,
)
from src.research.harness_registry import harness_mismatch_issue

GATE_STATUSES: Tuple[str, ...] = (
    "court_ok",
    "court_fail",
    "artifacts_missing",
)

CANONICAL_CRYPTO: Tuple[str, ...] = (
    "bear_2022",
    "bull_2023_2024",
    "recent_range_to_bear",
)

_RESULTS_PATH_RE = re.compile(r"results/[A-Za-z0-9_./-]+")


@dataclass
class GateRow:
    experiment_id: str
    gate_status: str
    suggested_verdict: Optional[str]
    reasons: List[str] = field(default_factory=list)
    kpi_files: List[str] = field(default_factory=list)
    kpi: Dict[str, float] = field(default_factory=dict)
    evidence_stale: bool = False
    feature_store_layer: Optional[str] = None
    stale_reasons: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_court(
    meta: ExperimentMeta,
    *,
    family: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """Return (gate_status, reasons) from metadata alone (no artifacts)."""
    reasons: List[str] = []
    mismatch = harness_mismatch_issue(family or meta.strategy, meta.harness)
    if mismatch:
        reasons.append(mismatch)
    if meta.kill_switch is True:
        reasons.append("kill_switch on — cannot rank edge")
    if reasons:
        return "court_fail", reasons

    segs = set(meta.segments)
    if not meta.harness:
        reasons.append("harness missing")
    if not segs:
        reasons.append("segments missing")
    elif meta.harness == "event_backtest" or (
        meta.strategy
        and meta.strategy
        in {"srb", "tpc", "bpc", "me", "largebar_fade", "ma_cross"}
    ):
        missing = [s for s in CANONICAL_CRYPTO if s not in segs]
        if missing:
            reasons.append("canonical crypto segments missing: " + ",".join(missing))
    elif len(segs) < 3:
        reasons.append("need at least 3 segments")

    if reasons:
        return "artifacts_missing", reasons
    return "court_ok", []


def _load_kpi_file(path: Path) -> Dict[str, float]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, float] = {}
    mapping = {
        "cagr": ("cagr",),
        "calmar": ("calmar",),
        "win_rate": ("win_rate", "wr"),
        "maxdd": ("max_drawdown_pct", "maxdd", "max_dd"),
        "sharpe": ("sharpe_daily_annualized", "sharpe_r", "sharpe"),
    }
    for dest, keys in mapping.items():
        for key in keys:
            if key in data and data[key] is not None:
                try:
                    out[dest] = float(data[key])
                    break
                except (TypeError, ValueError):
                    continue
    return out


def _kpi_from_trades_csv(path: Path) -> Dict[str, float]:
    """Win rate / Sharpe from event_backtest ``event_trades_*.csv`` (pnl_r)."""
    vals: List[float] = []
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                raw = row.get("pnl_r")
                if raw is None or raw == "":
                    continue
                try:
                    vals.append(float(raw))
                except ValueError:
                    continue
    except OSError:
        return {}
    if not vals:
        return {}
    out: Dict[str, float] = {
        "win_rate": sum(1 for v in vals if v > 0) / len(vals),
    }
    if len(vals) >= 2:
        sd = statistics.stdev(vals)
        if sd > 0:
            out["sharpe"] = statistics.mean(vals) / sd
    return out


def _complete_kpi(kpi: Dict[str, float]) -> Dict[str, float]:
    if "calmar" not in kpi and "cagr" in kpi and "maxdd" in kpi:
        dd = abs(float(kpi["maxdd"]))
        if dd > 1e-12:
            kpi["calmar"] = float(kpi["cagr"]) / dd
    return kpi


def _neighbor_kpi_files(path: Path) -> List[Path]:
    """Sibling result.json / trades csv next to a cited capital_report."""
    extra: List[Path] = []
    folder = path.parent if path.is_file() else path
    if not folder.is_dir():
        return extra
    for name in ("spot_daily_mtm_report.json", "result.json", "capital_report.json"):
        hit = folder / name
        if hit.is_file() and hit != path:
            extra.append(hit)
    extra.extend(sorted(folder.glob("event_trades_*.csv")))
    return extra


_KPI_SKIP_DIRS = frozenset(
    {"variants", "__pycache__", ".git", "node_modules", ".venv", "venv"}
)


def _walk_kpi_files(exp_dir: Path, *, max_depth: int = 4) -> List[Path]:
    """Shallow walk. Skip variants/ so a 10k-file tree does not hang Lab."""
    files: List[Path] = []
    if not exp_dir.is_dir():
        return files
    root = exp_dir.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        depth = 0 if str(rel) == "." else len(rel.parts)
        dirnames[:] = [
            d for d in dirnames if d not in _KPI_SKIP_DIRS and not d.startswith(".")
        ]
        if depth > max_depth:
            dirnames[:] = []
            continue
        for name in ("spot_daily_mtm_report.json", "result.json", "capital_report.json"):
            if name in filenames:
                files.append(Path(dirpath) / name)
    return files


def find_kpi_artifacts(exp_dir: Path, *, repo_root: Path) -> Tuple[List[Path], Dict[str, float]]:
    """Look under the experiment dir and paths cited in DECISION.md."""
    files: List[Path] = list(_walk_kpi_files(exp_dir))

    decision = find_decision_file(exp_dir)
    if decision:
        try:
            text = decision.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        for raw in _RESULTS_PATH_RE.findall(text):
            cited = (repo_root / raw).resolve()
            if cited.is_file() and cited.suffix in {".json", ".csv"}:
                files.append(cited)
                files.extend(_neighbor_kpi_files(cited))
            elif cited.is_dir():
                for name in (
                    "spot_daily_mtm_report.json",
                    "result.json",
                    "capital_report.json",
                ):
                    hit = cited / name
                    if hit.is_file():
                        files.append(hit)
                files.extend(sorted(cited.glob("event_trades_*.csv")))

    # Dedup, keep stable order
    seen: List[Path] = []
    for path in files:
        if path not in seen:
            seen.append(path)

    kpi: Dict[str, float] = {}
    for path in seen:
        if path.suffix == ".csv":
            got = _kpi_from_trades_csv(path)
        else:
            got = _load_kpi_file(path)
        for key, val in got.items():
            kpi.setdefault(key, val)
    return seen[:20], _complete_kpi(kpi)


_EXP_ID_RE = re.compile(r"^\d{8}_")
_PHASE3_ROW_KEYS = ("cagr", "calmar", "win_rate", "maxdd", "sharpe")


def _folder_kpi(folder: Path) -> Dict[str, float]:
    kpi: Dict[str, float] = {}
    for name in ("spot_daily_mtm_report.json", "result.json", "capital_report.json"):
        hit = folder / name
        if hit.is_file():
            for key, val in _load_kpi_file(hit).items():
                kpi.setdefault(key, val)
    for trades in sorted(folder.glob("event_trades_*.csv")):
        for key, val in _kpi_from_trades_csv(trades).items():
            kpi.setdefault(key, val)
    return _complete_kpi(kpi)


def _segment_children(folder: Path, known: set) -> List[Path]:
    if not folder.is_dir():
        return []
    try:
        return sorted(p for p in folder.iterdir() if p.is_dir() and p.name in known)
    except OSError:
        return []


def _variant_dirs(root: Path, known: set) -> List[Path]:
    if not root.is_dir():
        return []
    out: List[Path] = []
    try:
        children = list(root.iterdir())
    except OSError:
        return []
    for child in children:
        if child.is_dir() and _segment_children(child, known):
            out.append(child)
    return out


def _row_from_folder(
    folder: Path, *, variant: str, segment: str
) -> Optional[Dict[str, Any]]:
    kpi = _folder_kpi(folder)
    if not kpi:
        return None
    row: Dict[str, Any] = {"variant": variant, "segment": segment}
    for key in _PHASE3_ROW_KEYS:
        if key in kpi:
            row[key] = kpi[key]
    return row


def _scan_segment_tree(root: Path, known: set) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    variants = _variant_dirs(root, known)
    if variants:
        for vdir in variants:
            for seg_dir in _segment_children(vdir, known):
                row = _row_from_folder(
                    seg_dir, variant=vdir.name, segment=seg_dir.name
                )
                if row:
                    rows.append(row)
        return rows
    variant = ""
    if not _EXP_ID_RE.match(root.name) and root.name not in {"experiments", "results"}:
        variant = root.name
    for seg_dir in _segment_children(root, known):
        row = _row_from_folder(seg_dir, variant=variant, segment=seg_dir.name)
        if row:
            rows.append(row)
    return rows


def _segment_tree_from_cited(folder: Path, known: set) -> Optional[Path]:
    """Walk up so a ``baseline/bear_2022`` cite still finds sibling variants.

    Stop at this experiment's result folder. Do not climb into
    ``results/<family>/experiments`` and mix other runs.
    """
    found: Optional[Path] = None
    cur = folder
    for _ in range(6):
        if cur.name in {"experiments", "results"}:
            break
        if _variant_dirs(cur, known):
            return cur
        if _segment_children(cur, known):
            found = cur
        if _EXP_ID_RE.match(cur.name):
            break
        if cur.parent == cur:
            break
        cur = cur.parent
    return found


def find_segment_kpi_rows(
    exp_dir: Path, *, repo_root: Path
) -> List[Dict[str, Any]]:
    """Per-segment (and per-variant) court KPI from the cited results tree."""
    known = set(CANONICAL_CRYPTO) | set(load_known_segments(repo_root))
    folders: List[Path] = []
    decision = find_decision_file(exp_dir)
    if decision:
        try:
            text = decision.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        for raw in _RESULTS_PATH_RE.findall(text):
            cited = (repo_root / raw).resolve()
            folder = cited.parent if cited.is_file() else cited
            if folder.is_dir():
                folders.append(folder)

    meta = read_experiment_meta(exp_dir)
    if meta.strategy:
        convention = (
            repo_root
            / "results"
            / meta.strategy
            / "experiments"
            / exp_dir.name
        )
        if convention.is_dir():
            folders.append(convention)

    trees: List[Path] = []
    for folder in folders:
        tree = _segment_tree_from_cited(folder, known)
        if tree is not None and tree not in trees:
            trees.append(tree)

    rows: List[Dict[str, Any]] = []
    seen: set = set()
    seg_rank = {name: idx for idx, name in enumerate(CANONICAL_CRYPTO)}
    for tree in trees:
        for row in _scan_segment_tree(tree, known):
            key = (str(row.get("variant") or ""), str(row.get("segment") or ""))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    rows.sort(
        key=lambda r: (
            0
            if (r.get("variant") or "") in {"", "baseline", "base", "prod"}
            else 1,
            str(r.get("variant") or ""),
            seg_rank.get(str(r.get("segment") or ""), 99),
            str(r.get("segment") or ""),
        )
    )
    return rows


def evaluate_experiment(
    exp_dir: Path,
    *,
    repo_root: Path,
    known_segments: Iterable[str] = (),
    layer_cache: Optional[Dict[str, Optional[date]]] = None,
) -> GateRow:
    meta = read_experiment_meta(exp_dir, known_segments=known_segments)
    status, reasons = evaluate_court(meta, family=meta.strategy)
    files, kpi = find_kpi_artifacts(exp_dir, repo_root=repo_root)
    rel_files = []
    for path in files:
        try:
            rel_files.append(str(path.relative_to(repo_root)))
        except ValueError:
            rel_files.append(str(path))

    if status == "court_ok" and not files:
        status = "artifacts_missing"
        reasons = list(reasons) + ["no capital_report.json / result.json"]

    suggested: Optional[str] = None
    if status in {"court_fail", "artifacts_missing"}:
        suggested = "needs-more"
    # court_ok + artifacts → program still does not suggest promote or reject

    evidence_stale = False
    fs_layer: Optional[str] = None
    stale_reasons: List[str] = []
    if (meta.verdict or "").lower() == "promote":
        stale = evaluate_staleness(
            exp_dir, repo_root=repo_root, layer_cache=layer_cache
        )
        evidence_stale = stale.stale
        fs_layer = stale.layer
        stale_reasons = list(stale.reasons)
        if stale.stale:
            reasons = list(reasons) + list(stale.reasons)

    return GateRow(
        experiment_id=exp_dir.name,
        gate_status=status,
        suggested_verdict=suggested,
        reasons=reasons,
        kpi_files=rel_files,
        kpi=kpi,
        evidence_stale=evidence_stale,
        feature_store_layer=fs_layer,
        stale_reasons=stale_reasons,
    )


def build_gate_index(
    *, repo_root: Path, experiments_root: Optional[Path] = None
) -> Dict[str, Any]:
    root = experiments_root or (repo_root / "config" / "experiments")
    known = load_known_segments(repo_root)
    layer_cache: Dict[str, Optional[date]] = {}
    rows = [
        evaluate_experiment(
            p, repo_root=repo_root, known_segments=known, layer_cache=layer_cache
        )
        for p in iter_experiment_dirs(root)
    ]
    by_status: Dict[str, int] = {}
    for row in rows:
        by_status[row.gate_status] = by_status.get(row.gate_status, 0) + 1
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "owner": "program",
        "summary": {
            "count": len(rows),
            "by_gate_status": dict(sorted(by_status.items())),
        },
        "rows": [r.as_dict() for r in rows],
    }


def merge_gate_into_index(index: Dict[str, Any], gate_index: Dict[str, Any]) -> None:
    """Attach program gate fields onto EXPERIMENT_INDEX rows (in place)."""
    by_id = {r["experiment_id"]: r for r in gate_index.get("rows") or []}
    for row in index.get("rows") or []:
        g = by_id.get(row.get("id"))
        if not g:
            continue
        row["gate_status"] = g.get("gate_status")
        row["suggested_verdict"] = g.get("suggested_verdict")
        row["gate_reasons"] = g.get("reasons") or []
        row["evidence_stale"] = bool(g.get("evidence_stale"))
        row["feature_store_layer"] = g.get("feature_store_layer")
        row["stale_reasons"] = g.get("stale_reasons") or []
        if g.get("kpi") and not row.get("kpi"):
            row["kpi"] = g["kpi"]
    index["gate_summary"] = gate_index.get("summary") or {}


def allowed_declare(
    gate: GateRow, verdict: str, *, yes: bool
) -> Optional[str]:
    """None if allowed; otherwise a one-line refusal."""
    v = verdict.strip().lower()
    if v == "promote":
        if gate.gate_status != "court_ok":
            return (
                f"cannot declare promote: gate_status={gate.gate_status} "
                f"({'; '.join(gate.reasons) or 'court not ok'})"
            )
        if not yes:
            return "promote requires --yes after a human read of the three-segment KPI table"
        return None
    if v in {"reject", "park", "needs-more"}:
        return None
    return f"unknown verdict {verdict!r}"
