from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8") or "{}")
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    _atomic_write_text(
        path, json.dumps(obj or {}, ensure_ascii=False, indent=2, default=str)
    )


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj or {}, ensure_ascii=False, default=str)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


@dataclass(frozen=True)
class ConstitutionStatePaths:
    base_dir: Path
    slots_path: Optional[Path] = None
    slots_db_path: Optional[Path] = None
    add_position_path: Optional[Path] = None
    add_position_db_path: Optional[Path] = None

    def resolve(self, p: Optional[str]) -> Optional[Path]:
        if not p:
            return None
        pp = Path(str(p))
        if pp.is_absolute():
            return pp
        return (self.base_dir / pp).resolve()
