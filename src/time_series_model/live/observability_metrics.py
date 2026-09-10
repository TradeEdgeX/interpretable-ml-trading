from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional


def compute_tick_gap_seconds(
    *, now_ns: int, last_tick_ts_ns: Optional[int]
) -> Optional[float]:
    if last_tick_ts_ns is None:
        return None
    try:
        gap_ns = int(now_ns) - int(last_tick_ts_ns)
        return float(gap_ns) / 1e9 if gap_ns >= 0 else 0.0
    except Exception:
        return None


def compute_feature_missing_rate(
    *, required_keys: List[str], features: Dict[str, Any]
) -> Optional[float]:
    req = [str(k) for k in (required_keys or []) if str(k)]
    if not req:
        return None
    feats = features or {}
    missing = sum(1 for k in req if k not in feats)
    return float(missing) / float(len(req))


def compute_evidence_true_rate(evidence: Optional[Dict[str, bool]]) -> Optional[float]:
    if not evidence:
        return None
    vals = [bool(v) for v in evidence.values()]
    if not vals:
        return None
    return float(sum(1 for v in vals if v)) / float(len(vals))


def compute_router_mode_entropy(modes: List[str]) -> Optional[float]:
    xs = [str(x).upper() for x in (modes or []) if str(x)]
    if not xs:
        return None
    c = Counter(xs)
    n = float(sum(c.values()))
    if n <= 0:
        return None
    ent = 0.0
    for k, v in c.items():
        p = float(v) / n
        if p > 0:
            ent -= p * math.log(p + 1e-12)
    return float(ent)
