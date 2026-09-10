"""Write last_calibration.plateaus baselines into archetype yaml (no rule value change)."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from scripts.plateau_stability import PlateauRange


def _plateau_range_from_payload(payload: Dict[str, Any]) -> Optional[PlateauRange]:
    start = payload.get("start_threshold")
    end = payload.get("end_threshold")
    if start is None or end is None:
        rec = payload.get("recommended") or payload.get("mid")
        if rec is None:
            return None
        return PlateauRange(start=float(rec), end=float(rec), mid=float(rec))
    mid = payload.get("mid") or payload.get("recommended")
    if mid is None:
        mid = (float(start) + float(end)) / 2.0
    return PlateauRange(start=float(start), end=float(end), mid=float(mid))


def merge_plateau_baseline(
    raw_yaml: Dict[str, Any],
    *,
    feature: str,
    operator: str,
    plateau: PlateauRange,
    timestamp_iso: Optional[str] = None,
    data_source: str = "research_plateau",
    action: str = "BASELINE",
    reason: str = "backfill from research plateau.json",
) -> Dict[str, Any]:
    """Return updated yaml dict with last_calibration.plateaus entry merged."""
    out = copy.deepcopy(raw_yaml)
    last_cal = out.get("last_calibration")
    if not isinstance(last_cal, dict):
        last_cal = {}
        out["last_calibration"] = last_cal
    last_cal["timestamp"] = timestamp_iso or datetime.now(timezone.utc).isoformat()
    last_cal["data_source"] = data_source
    plateaus = last_cal.get("plateaus")
    if not isinstance(plateaus, list):
        plateaus = []
        last_cal["plateaus"] = plateaus
    plateaus[:] = [
        p
        for p in plateaus
        if not (
            isinstance(p, dict)
            and str(p.get("feature", "")) == feature
            and str(p.get("operator", "")) == operator
        )
    ]
    plateaus.append(
        {
            "feature": feature,
            "operator": operator,
            "plateau": {
                "start": float(plateau.start),
                "end": float(plateau.end),
                "mid": float(plateau.mid),
            },
            "action": action,
            "reason": reason,
        }
    )
    return out


def merge_plateau_baseline_from_json(
    raw_yaml: Dict[str, Any],
    plateau_json: Dict[str, Any],
    *,
    feature: Optional[str] = None,
    operator: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Merge plateau baseline using fields from research plateau.json."""
    feat = feature or str(plateau_json.get("feature", ""))
    op = operator or str(plateau_json.get("operator", ""))
    if not feat or not op:
        raise ValueError("plateau json missing feature/operator")
    plateau = _plateau_range_from_payload(plateau_json)
    if plateau is None:
        raise ValueError("plateau json has no start/end/recommended thresholds")
    return merge_plateau_baseline(
        raw_yaml,
        feature=feat,
        operator=op,
        plateau=plateau,
        **kwargs,
    )


def load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping yaml: {path}")
    return data


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(
        yaml.dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
