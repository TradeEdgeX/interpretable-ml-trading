"""Repo / experiments / results roots for the court Lab."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    raw = (os.environ.get("MLBOT_REPO_ROOT") or "").strip()
    if raw:
        return Path(raw).resolve()
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "setup.py").exists() or (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()


def experiments_root(root: Path | None = None) -> Path:
    return (root or repo_root()) / "config" / "experiments"


def results_root(root: Path | None = None) -> Path:
    raw = (os.environ.get("MLBOT_RESULTS_ROOT") or "").strip()
    if raw:
        return Path(raw).resolve()
    return (root or repo_root()) / "results"


def static_dir() -> Path:
    return Path(__file__).resolve().parent / "static"
