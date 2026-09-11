"""Safe listing of ``results/`` for the court Lab."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KiB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MiB"
    return f"{n / 1024**3:.2f} GiB"


def shallow_dir_bytes(d: Path) -> int:
    total = 0
    try:
        for child in d.iterdir():
            if child.is_file():
                try:
                    total += child.stat().st_size
                except OSError:
                    pass
            elif child.is_dir():
                try:
                    for sub in child.iterdir():
                        if sub.is_file():
                            try:
                                total += sub.stat().st_size
                            except OSError:
                                pass
                except OSError:
                    pass
    except OSError:
        pass
    return total


def safe_browse_target(results_root: Path, rel_under: str) -> Optional[Path]:
    root = results_root.resolve()
    rel = rel_under.strip().strip("/").replace("\\", "/")
    if ".." in rel.split("/"):
        return None
    cand = (root / rel).resolve() if rel else root
    try:
        cand.relative_to(root)
    except ValueError:
        return None
    return cand


def list_browse(results_root: Path, rel_under: str = "") -> Optional[Dict[str, Any]]:
    root = results_root.resolve()
    rel = rel_under.strip().strip("/").replace("\\", "/")
    if not root.exists():
        return {
            "results_root": str(root),
            "path": rel,
            "parent": "/".join(rel.split("/")[:-1]) if rel else None,
            "crumbs": [{"label": "results", "path": ""}],
            "entries": [],
            "missing_root": True,
        }

    listed = safe_browse_target(root, rel)
    if listed is None or not listed.exists():
        return None

    entries: List[Dict[str, Any]] = []
    try:
        children = sorted(
            listed.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
        )
    except OSError:
        children = []

    for p in children:
        if p.name.startswith("."):
            continue
        try:
            rel_item = p.relative_to(root).as_posix()
        except ValueError:
            continue
        if p.is_file():
            try:
                size = fmt_bytes(p.stat().st_size)
            except OSError:
                size = "—"
            entries.append(
                {
                    "name": p.name,
                    "kind": "file",
                    "rel_path": rel_item,
                    "size": size,
                    "href": f"/results/{rel_item}",
                }
            )
            continue
        sz_b = shallow_dir_bytes(p)
        entries.append(
            {
                "name": p.name,
                "kind": "dir",
                "rel_path": rel_item,
                "size": fmt_bytes(sz_b) if sz_b > 0 else "0 B",
                "href": None,
            }
        )

    crumbs: List[Dict[str, str]] = [{"label": "results", "path": ""}]
    if rel:
        acc: List[str] = []
        for part in rel.split("/"):
            if not part:
                continue
            acc.append(part)
            crumbs.append({"label": part, "path": "/".join(acc)})

    return {
        "results_root": str(root),
        "path": rel,
        "parent": "/".join(rel.split("/")[:-1]) if rel else None,
        "crumbs": crumbs,
        "entries": entries,
        "missing_root": False,
    }
