from __future__ import annotations

import re
from pathlib import Path


def _slug(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def make_primitives_task_id(
    *,
    config_dir: str | Path,
    timeframe: str,
    horizon_hours: float,
    bar_hours: float,
    version: str = "v1",
) -> str:
    """
    Deterministic task_id for nnmultihead primitives tasks.

    This is intentionally independent from the tree-model strategy pipeline.
    """
    cfg = _slug(Path(config_dir).name)
    tf = _slug(timeframe)
    h = _slug(str(int(round(float(horizon_hours)))))
    b = _slug(str(int(round(float(bar_hours)))))
    ver = _slug(version)
    return f"primitives__{cfg}__tf_{tf}__bar_{b}h__h_{h}h__{ver}"
