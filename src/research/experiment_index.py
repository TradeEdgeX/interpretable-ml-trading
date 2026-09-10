"""Machine-readable lineage over ``config/experiments/``.

Each experiment states its verdict in ``DECISION.md`` YAML front-matter. Prose
regex stays as a fallback so the ~300 pre-front-matter experiments still appear,
but only ``declared`` rows are trustworthy enough for a promote gate or for an
agent asking "has this been rejected already?".

Canonical schema (see ``config/experiments/README.md``)::

    ---
    topic: 20260818_ashare_full_tp15
    strategy: ashare_oversold
    harness: ashare_slot_policy
    segments: [bear_2022, bull_2023_2024, recent_range_to_bear]
    kill_switch: false
    verdict: reject
    kpi: {cagr: 12.2, calmar: 0.31, maxdd: -39.0, win_rate: 55.9, sharpe: 0.42}
    supersedes: [20260805_ashare_exit_policy]
    tags: [take-profit, ashare]
    ---
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from src.research.harness_registry import (
    KNOWN_HARNESSES,
    harness_mismatch_issue,
)

SCHEMA_VERSION = 1

VERDICTS: Tuple[str, ...] = ("promote", "reject", "park", "needs-more")

#: How an agent may use a row. Computed; never hand-written as a verdict.
RECORD_CLASSES: Tuple[str, ...] = ("trusted", "open", "legacy", "example")

#: Headline KPI keys per ``report-kpi-cagr-not-sum-r``.
KPI_KEYS: Tuple[str, ...] = ("cagr", "calmar", "win_rate", "maxdd", "sharpe")

#: Metrics that must not stand in for the headline KPI set.
KPI_FORBIDDEN: Tuple[str, ...] = ("total_r", "sum_r", "pnl_r", "totr")

_SEGMENT_FILES: Tuple[str, ...] = (
    "market_segment.yaml",
    "market_segment_crypto.yaml",
    "market_segment_ashare.yaml",
    "market_segment_hk.yaml",
)

_FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.S)

_DIR_RE = re.compile(r"^(\d{8})(?:_(\d{4}))?_(.+)$")

# Fallback only, for the ~300 experiments written before front-matter existed.
# Latin verdict tokens stay case-SENSITIVE on purpose: this repo writes verdicts
# as PROMOTE / REJECT, while "## Promote" headings and "promote-baseline"
# commands appear in every checklist and must not be read as a conclusion.
# Ordered: an explicit REJECT wins over a promote-criteria citation.
_VERDICT_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("reject", re.compile(r"\bREJECT\b|(?i:不\s*promote|不采纳|已否决)")),
    ("promote", re.compile(r"\bPROMOTE\b|(?i:已写入生产|promote prod)")),
    ("park", re.compile(r"\bPARK\b|暂缓")),
    ("needs-more", re.compile(r"(?i:needs-more|结论 TODO)")),
)


@dataclass(frozen=True)
class ExperimentMeta:
    """Declared (or inferred) experiment metadata."""

    verdict: Optional[str] = None
    # declared | inferred | ambiguous | missing
    verdict_source: str = "missing"
    strategy: Optional[str] = None
    harness: Optional[str] = None
    segments: Tuple[str, ...] = ()
    kill_switch: Optional[bool] = None
    kpi: Dict[str, float] = field(default_factory=dict)
    supersedes: Tuple[str, ...] = ()
    tags: Tuple[str, ...] = ()
    issues: Tuple[str, ...] = ()
    record_class: str = "legacy"

    @property
    def declared(self) -> bool:
        return self.verdict_source == "declared"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "verdict_source": self.verdict_source,
            "strategy": self.strategy,
            "harness": self.harness,
            "segments": list(self.segments),
            "kill_switch": self.kill_switch,
            "kpi": dict(self.kpi),
            "supersedes": list(self.supersedes),
            "tags": list(self.tags),
            "issues": list(self.issues),
            "record_class": self.record_class,
        }


def parse_front_matter(text: str) -> Tuple[Dict[str, Any], str]:
    """Split leading YAML front-matter from the markdown body.

    Returns ``({}, text)`` when the document has no front-matter.
    """
    if not text:
        return {}, ""
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return {}, text
    try:
        data = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return {}, text
    if not isinstance(data, dict):
        return {}, text
    return data, text[m.end() :]


def load_known_segments(repo_root: Path) -> frozenset[str]:
    """Union of segment ids across the canonical segment files."""
    ids: set[str] = set()
    for name in _SEGMENT_FILES:
        path = repo_root / "config" / name
        if not path.is_file():
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        for seg in (data or {}).get("segments") or []:
            if isinstance(seg, dict) and seg.get("id"):
                ids.add(str(seg["id"]))
    return frozenset(ids)


def _is_ascii_slug(value: str) -> bool:
    """Tags / query keys: lowercase ASCII + hyphen only."""
    if not value or not value.isascii():
        return False
    return all(ch.isalnum() or ch in "-_." for ch in value)


def _as_str_tuple(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
        return tuple(p for p in parts if p)
    if isinstance(value, Sequence):
        return tuple(str(v).strip() for v in value if str(v).strip())
    return ()


def _coerce_kpi(raw: Any, issues: List[str]) -> Dict[str, float]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        issues.append("kpi: must be a mapping")
        return {}
    out: Dict[str, float] = {}
    for key, val in raw.items():
        name = str(key).strip().lower()
        if name in KPI_FORBIDDEN:
            issues.append(f"kpi.{name}: not a headline KPI (use cagr/calmar/maxdd)")
            continue
        try:
            out[name] = float(val)
        except (TypeError, ValueError):
            issues.append(f"kpi.{name}: not numeric ({val!r})")
    return out


def validate_meta(
    raw: Dict[str, Any],
    *,
    known_segments: Iterable[str] = (),
    fallback_text: str = "",
) -> ExperimentMeta:
    """Build :class:`ExperimentMeta` from front-matter, recording issues.

    Issues are reported rather than raised: the index must still list a broken
    experiment so a human can see that it needs fixing.
    """
    issues: List[str] = []
    segment_ids = set(known_segments)

    verdict = str(raw.get("verdict") or "").strip().lower() or None
    if verdict and verdict not in VERDICTS:
        issues.append(f"verdict: unknown {verdict!r} (allowed: {', '.join(VERDICTS)})")
        verdict = None

    if verdict:
        source = "declared"
    else:
        verdict, source = infer_verdict_detail(fallback_text)

    harness = str(raw.get("harness") or "").strip() or None
    if harness and harness not in KNOWN_HARNESSES:
        issues.append(f"harness: unknown {harness!r}")
    strategy = str(raw.get("strategy") or "").strip() or None
    mismatch = harness_mismatch_issue(strategy, harness)
    if mismatch:
        issues.append(mismatch)

    segments = _as_str_tuple(raw.get("segments"))
    if segment_ids:
        for seg in segments:
            if seg not in segment_ids:
                issues.append(f"segments: unknown id {seg!r}")

    kill_switch = raw.get("kill_switch")
    if kill_switch is not None and not isinstance(kill_switch, bool):
        issues.append("kill_switch: must be true/false")
        kill_switch = None

    tags = _as_str_tuple(raw.get("tags"))
    for tag in tags:
        if not _is_ascii_slug(tag):
            issues.append(
                f"tags: {tag!r} must be an ASCII slug (e.g. stop-loss, not 止损)"
            )

    if raw and verdict and source == "declared":
        if harness is None:
            issues.append("harness: required when verdict is declared")
        if not segments:
            issues.append("segments: required when verdict is declared")
        if kill_switch is None:
            issues.append("kill_switch: required when verdict is declared")

    return ExperimentMeta(
        verdict=verdict,
        verdict_source=source,
        strategy=strategy,
        harness=harness,
        segments=segments,
        kill_switch=kill_switch,
        kpi=_coerce_kpi(raw.get("kpi"), issues),
        supersedes=_as_str_tuple(raw.get("supersedes")),
        tags=tags,
        issues=tuple(issues),
        record_class=classify_record(raw, verdict=verdict, source=source),
    )


def classify_record(
    raw: Dict[str, Any], *, verdict: Optional[str], source: str
) -> str:
    """trusted / open / legacy / example — program class, not a human verdict."""
    role = str((raw or {}).get("role") or "").strip().lower()
    if role in {"harness-example", "example", "fixture"}:
        return "example"
    if source == "declared" and verdict:
        return "trusted"
    std = str((raw or {}).get("standardization") or "").strip().lower()
    if std == "auto-scaffold":
        return "legacy"
    if raw and not verdict:
        return "open"
    return "legacy"


def infer_verdict_detail(decision_text: str) -> Tuple[Optional[str], str]:
    """Regex fallback for docs written before front-matter existed.

    A closed experiment routinely PROMOTEs one track while REJECTing others, so
    a document matching several verdicts has no single inferable outcome. Such
    docs are reported as ``ambiguous`` with no verdict rather than silently
    taking the first match — the whole point of the index is that unverified
    verdicts must not read as facts.
    """
    if not decision_text.strip():
        return None, "missing"
    matched = [name for name, pat in _VERDICT_PATTERNS if pat.search(decision_text)]
    if not matched:
        return None, "missing"
    if len(matched) > 1:
        return None, "ambiguous"
    return matched[0], "inferred"


def infer_verdict(decision_text: str) -> Optional[str]:
    """Unambiguous fallback verdict, or ``None``."""
    return infer_verdict_detail(decision_text)[0]


def find_decision_file(exp_dir: Path) -> Optional[Path]:
    primary = exp_dir / "DECISION.md"
    if primary.is_file():
        return primary
    for pattern in ("DECISION*.md", "*_experiment_*.md", "SUMMARY*.md"):
        for path in sorted(exp_dir.glob(pattern)):
            if path.is_file():
                return path
    return None


def read_experiment_meta(
    exp_dir: Path, *, known_segments: Iterable[str] = ()
) -> ExperimentMeta:
    """Parse the experiment's decision doc into :class:`ExperimentMeta`."""
    decision = find_decision_file(exp_dir)
    if not decision:
        return ExperimentMeta()
    try:
        text = decision.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ExperimentMeta()
    raw, body = parse_front_matter(text)
    return validate_meta(raw, known_segments=known_segments, fallback_text=body)


def _parse_dir_name(name: str) -> Dict[str, Optional[str]]:
    m = _DIR_RE.match(name)
    if not m:
        return {"date": None, "topic": name}
    date_s, _time_s, topic = m.group(1), m.group(2), m.group(3)
    return {"date": f"{date_s[:4]}-{date_s[4:6]}-{date_s[6:8]}", "topic": topic}


def scan_experiment(
    exp_dir: Path, *, repo_root: Path, known_segments: Iterable[str] = ()
) -> Optional[Dict[str, Any]]:
    """One index row for an experiment directory."""
    if not exp_dir.is_dir() or exp_dir.name.startswith("."):
        return None

    meta = read_experiment_meta(exp_dir, known_segments=known_segments)
    parsed = _parse_dir_name(exp_dir.name)
    decision = find_decision_file(exp_dir)

    row: Dict[str, Any] = {
        "id": exp_dir.name,
        "date": parsed["date"],
        "topic": parsed["topic"],
        "dir": str(exp_dir.relative_to(repo_root)),
        "decision_path": (
            str(decision.relative_to(repo_root)) if decision else None
        ),
        **meta.as_dict(),
    }
    if not row.get("strategy"):
        row["strategy"] = None
    return row


def iter_experiment_dirs(experiments_root: Path) -> List[Path]:
    if not experiments_root.is_dir():
        return []
    return sorted(
        (
            p
            for p in experiments_root.iterdir()
            if p.is_dir() and not p.name.startswith(".") and not p.name.startswith("_")
        ),
        key=lambda p: p.name,
    )


def build_index(
    *, repo_root: Path, experiments_root: Optional[Path] = None
) -> Dict[str, Any]:
    """Scan every experiment directory into a serialisable index."""
    root = experiments_root or (repo_root / "config" / "experiments")
    known_segments = load_known_segments(repo_root)

    rows: List[Dict[str, Any]] = []
    for exp_dir in iter_experiment_dirs(root):
        row = scan_experiment(
            exp_dir, repo_root=repo_root, known_segments=known_segments
        )
        if row:
            rows.append(row)

    by_verdict: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    by_class: Dict[str, int] = {}
    for row in rows:
        by_verdict[str(row.get("verdict"))] = by_verdict.get(str(row.get("verdict")), 0) + 1
        src = str(row.get("verdict_source"))
        by_source[src] = by_source.get(src, 0) + 1
        cls = str(row.get("record_class") or "legacy")
        by_class[cls] = by_class.get(cls, 0) + 1

    invalid = [r["id"] for r in rows if r.get("issues")]

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiments_root": str(root.relative_to(repo_root)),
        "summary": {
            "count": len(rows),
            "by_verdict": dict(sorted(by_verdict.items())),
            "by_source": dict(sorted(by_source.items())),
            "by_record_class": dict(sorted(by_class.items())),
            "invalid_count": len(invalid),
        },
        "invalid": invalid,
        "rows": rows,
    }


def filter_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    verdict: Optional[str] = None,
    strategy: Optional[str] = None,
    tag: Optional[str] = None,
    query: Optional[str] = None,
    declared_only: bool = False,
    record_class: Optional[str] = None,
    record_classes: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Query helper backing ``mlbot research index --verdict/--tag/--query``."""
    out: List[Dict[str, Any]] = []
    q = (query or "").strip().lower()
    allowed = None
    if record_classes:
        allowed = {c.lower() for c in record_classes}
    elif record_class:
        allowed = {record_class.lower()}
    elif declared_only:
        allowed = {"trusted"}
    for row in rows:
        if allowed and str(row.get("record_class") or "legacy").lower() not in allowed:
            continue
        if verdict and str(row.get("verdict") or "").lower() != verdict.lower():
            continue
        if strategy and str(row.get("strategy") or "").lower() != strategy.lower():
            continue
        if tag and tag.lower() not in [t.lower() for t in row.get("tags") or []]:
            continue
        if declared_only and row.get("verdict_source") != "declared":
            continue
        if q:
            hay = " ".join(
                str(row.get(k) or "")
                for k in ("id", "topic", "strategy", "harness", "dir")
            )
            hay = f"{hay} {' '.join(row.get('tags') or [])}".lower()
            if q not in hay:
                continue
        out.append(row)
    return out
