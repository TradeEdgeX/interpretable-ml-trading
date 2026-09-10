"""Research-court Q&A. Canonical file: ``docs/agent/rd_qa.yaml``.

Lab ``/rd/qa`` and agents read this. Do not fork a second FAQ in the UI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

QA_REL = Path("docs/agent/rd_qa.yaml")
REQUIRED_IDS = (
    "rule-tss",
    "rule-overfit",
    "no-pool",
    "ic-predict",
    "ic-lowdim",
    "model-vs-rule",
    "phase1-no-close",
    "three-segments",
    "phase12-insample",
    "rule-holdout-oos",
    "embargo-vs-tss",
    "tree-forward-rr",
    "purge-overlap",
    "sleeve-clock",
    "taker-burst-markout",
    "hft-not-oms",
)


def qa_path(repo_root: Path) -> Path:
    return Path(repo_root) / QA_REL


def load_rd_qa(repo_root: Path) -> Dict[str, Any]:
    path = qa_path(repo_root)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a mapping")
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path} has no items")
    return data


def qa_payload(repo_root: Path) -> Dict[str, Any]:
    data = load_rd_qa(repo_root)
    items: List[Dict[str, Any]] = []
    for raw in data.get("items") or []:
        if not isinstance(raw, dict) or not raw.get("id"):
            continue
        items.append(
            {
                "id": str(raw["id"]),
                "tags": [str(t) for t in (raw.get("tags") or [])],
                "q": dict(raw.get("q") or {}),
                "a": dict(raw.get("a") or {}),
            }
        )
    return {
        "schema_version": int(data.get("schema_version") or 1),
        "source": str(data.get("source") or QA_REL),
        "lab_path": str(data.get("lab_path") or "/rd/qa"),
        "related": [str(x) for x in (data.get("related") or [])],
        "title": dict(data.get("title") or {}),
        "blurb": dict(data.get("blurb") or {}),
        "items": items,
    }


def render_qa_markdown(data: Dict[str, Any], lang: str) -> str:
    loc = "en" if lang == "en" else "zh"
    title = (data.get("title") or {}).get(loc) or "Q&A"
    blurb = (data.get("blurb") or {}).get(loc) or ""
    source = data.get("source") or QA_REL
    lab = data.get("lab_path") or "/rd/qa"
    related = data.get("related") or []
    lines = [
        f"# {title}",
        "",
        blurb,
        "",
        f"Lab: `{lab}` · source: `{source}`",
        "",
    ]
    if related:
        lines.append("Related:")
        for rel in related:
            lines.append(f"- `{rel}`")
        lines.append("")
    for item in data.get("items") or []:
        q = ((item.get("q") or {}).get(loc) or "").strip()
        a = ((item.get("a") or {}).get(loc) or "").strip()
        if not q:
            continue
        lines.append(f"## {q}")
        lines.append("")
        lines.append(a)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
