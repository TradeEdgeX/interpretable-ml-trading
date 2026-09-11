"""Structural check of a hypothesis template. Does not judge truth."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CANONICAL_SEGMENTS = (
    "bear_2022",
    "bull_2023_2024",
    "recent_range_to_bear",
)

KPI_TOKENS = (
    "cagr",
    "calmar",
    "sharpe",
    "maxdd",
    "win_rate",
    "年化",
    "胜率",
    "回撤",
)

CLASS_TOKENS = (
    "alpha",
    "beta",
    "fattail",
    "fat-tail",
    "肥尾",
    "无用",
    "趋势",
    "trend",
)

CLOSED_BAR_TOKENS = ("闭棒", "收盘", "closed-bar", "closed bar", "close-only")

# heading text (lower) → slot
_SLOT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "claim": ("原句", "claim", "人说的原句", "hypothesis sentence"),
    "sociology": ("社会学", "sociology"),
    "math": ("数学", "mathematics", "math"),
    "stats": ("统计学", "statistics", "统计现象", "统计"),
    "standard": ("验证标准", "validation standard", "证伪条件"),
    "range": ("数据范围", "data range", "回测数据范围"),
    "boxes": ("五格", "five boxes"),
}

_PLACEHOLDER = re.compile(
    r"^(?:todo|tbd|pending|待填|待写|\.+\s*$|—\s*$|-\s*$)$",
    re.I,
)
_SCAFFOLD_HINT = re.compile(
    r"不要写|待填|待写|AI 不许|必须能落到|哪一段、哪项|人说的那一句|先分类：|不能单独结案"
)
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.M)


@dataclass
class TemplateReport:
    ok: bool
    missing: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    slots: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "missing": list(self.missing),
            "issues": list(self.issues),
            "slots": {k: (v[:80] + "…") if len(v) > 80 else v for k, v in self.slots.items()},
        }


def _norm_heading(raw: str) -> str:
    s = raw.strip().lower()
    s = re.sub(r"[#：:：\s]+", " ", s)
    return s.strip(" .")


def _slot_for_heading(heading: str) -> Optional[str]:
    h = _norm_heading(heading)
    ranked: List[Tuple[int, str, str]] = []
    for slot, aliases in _SLOT_ALIASES.items():
        for alias in aliases:
            ranked.append((len(alias), slot, alias.lower()))
    for _n, slot, alias in sorted(ranked, reverse=True):
        if h == alias or alias in h:
            return slot
    return None


def split_sections(text: str) -> Dict[str, str]:
    """Map template slots to the markdown body under the matching heading."""
    matches = list(_HEADING_RE.finditer(text))
    slots: Dict[str, List[str]] = {}
    for i, m in enumerate(matches):
        slot = _slot_for_heading(m.group(2))
        if not slot:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        slots.setdefault(slot, []).append(body)
    return {k: "\n\n".join(v).strip() for k, v in slots.items()}


def _nonempty(body: str) -> bool:
    if not body or len(body.strip()) < 8:
        return False
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    useful = []
    for ln in lines:
        if ln.startswith("|") and set(ln.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
            continue
        cell = ln.strip("|* ")
        if _PLACEHOLDER.match(cell):
            continue
        if _SCAFFOLD_HINT.search(ln):
            continue
        if cell.startswith("（") and cell.endswith("）"):
            continue
        if cell.startswith("(") and cell.endswith(")"):
            continue
        useful.append(ln)
    return len(" ".join(useful)) >= 8


def _has_token(text: str, tokens: Tuple[str, ...]) -> bool:
    low = text.lower()
    return any(t.lower() in low for t in tokens)


def validate_template_text(text: str) -> TemplateReport:
    """Check the template is fillable enough to orchestrate. Not a verdict."""
    slots = split_sections(text)
    # Prose fallback: 人说的原句 often sits above the first heading.
    if not _nonempty(slots.get("claim", "")):
        m = re.search(r"人说的原句[：:]\s*(.+)", text)
        if m and len(m.group(1).strip()) >= 8:
            slots["claim"] = m.group(1).strip()

    missing: List[str] = []
    issues: List[str] = []
    required = ("claim", "sociology", "math", "stats", "standard", "range")
    for slot in required:
        if not _nonempty(slots.get(slot, "")):
            missing.append(slot)

    if not _nonempty(slots.get("boxes", "")):
        box_hits = sum(
            1
            for key in ("机制", "预期市况", "合同", "证伪", "落地", "mechanism", "regimes", "contract")
            if key in text
        )
        if box_hits < 4:
            missing.append("boxes")

    joined = "\n".join(slots.get(k, "") for k in ("stats", "standard", "range", "boxes"))
    joined += "\n" + text
    if not _has_token(joined, KPI_TOKENS):
        issues.append("validation standard must name a court KPI (CAGR / Calmar / WR / MaxDD / Sharpe)")
    if not _has_token(joined, CANONICAL_SEGMENTS) and not any(
        tok in joined
        for tok in (
            "market_segment.yaml",
            "market_segment_ashare.yaml",
            "market_segment_crypto.yaml",
            "bear_2021",
            "bull_924",
            "chop_recent",
        )
    ):
        issues.append(
            "data range must name canonical segments "
            "(bear_2022 / bull_2023_2024 / recent_range_to_bear) "
            "or market_segment.yaml / market_segment_ashare.yaml"
        )
    if not _has_token(slots.get("stats", "") + slots.get("boxes", ""), CLASS_TOKENS):
        issues.append("statistics must classify the claim (alpha / fat-tail / beta / 无用)")
    if not _has_token(slots.get("math", "") + slots.get("range", "") + text, CLOSED_BAR_TOKENS):
        issues.append("math or data range must say the clock is closed-bar")

    ok = not missing and not issues
    return TemplateReport(ok=ok, missing=missing, issues=issues, slots=slots)


def validate_decision_path(path: Path) -> TemplateReport:
    text = path.read_text(encoding="utf-8")
    return validate_template_text(text)
