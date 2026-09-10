from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import json
from pathlib import Path


@dataclass(frozen=True)
class EvidenceRule:
    name: str
    kind: str
    any_key_contains: List[str]


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


_EVIDENCE_QUANTILES_CACHE: Tuple[str, Dict[str, Any]] | None = None


def load_evidence_quantiles(path: str | None) -> Dict[str, Any] | None:
    """
    Load evidence quantiles from JSON path. Cached by path.
    Expected format:
      {
        "BTCUSDT": {
          "vpin": {"0.1": 0.01, "0.9": 0.12},
          "cvd_change_5": {"0.5": 0.0}
        }
      }
    """
    global _EVIDENCE_QUANTILES_CACHE
    if not path:
        return None
    p = Path(str(path))
    if not p.exists():
        return None
    if _EVIDENCE_QUANTILES_CACHE and _EVIDENCE_QUANTILES_CACHE[0] == str(p):
        return _EVIDENCE_QUANTILES_CACHE[1]
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            _EVIDENCE_QUANTILES_CACHE = (str(p), obj)
            return obj
    except Exception:
        return None
    return None


def _get_quantile_threshold(
    *,
    quantiles: Dict[str, Any] | None,
    key: str,
    q: float,
) -> Optional[float]:
    if not quantiles or not key:
        return None
    qkey = f"{float(q):.2f}".rstrip("0").rstrip(".")
    qkey_alt = f"q{int(round(float(q) * 100))}"
    entry = quantiles.get(key)
    if isinstance(entry, dict):
        for k in (qkey, qkey_alt, str(q)):
            if k in entry:
                try:
                    return float(entry[k])
                except Exception:
                    return None
    return None


def compute_execution_evidence(
    *,
    features: Dict[str, Any],
    rules: List[Dict[str, Any]] | None,
    quantiles: Dict[str, Any] | None = None,
) -> Dict[str, bool]:
    """
    Build boolean evidence flags from a feature dict.

    Supported rule kinds (v1):
    - any_key_contains: evidence true if any feature key contains any substring.
    - key_exists: evidence true if key exists in features
    - value_gt/value_gte/value_lt/value_lte: numeric comparisons on a key
    - abs_gt: abs(value(key)) > threshold
    - quantile_gt/quantile_gte/quantile_lt/quantile_lte/quantile_abs_gt:
        threshold is derived from quantiles[key][q]

    Missing key policy:
    - on_missing: "false" (default) | "true" | "error"
    """
    feats = features or {}
    keys = [str(k) for k in feats.keys()]

    out: Dict[str, bool] = {}
    for r in rules or []:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name") or "").strip()
        if not name:
            continue
        kind = str(r.get("kind") or "any_key_contains").strip()
        on_missing = str(r.get("on_missing") or "false").strip().lower()

        def _missing() -> bool:
            if on_missing == "true":
                return True
            if on_missing == "error":
                raise KeyError(f"Missing feature key for evidence '{name}'")
            return False

        if kind == "any_key_contains":
            subs = r.get("any_key_contains") or []
            subs = [str(x) for x in subs if str(x)]
            ok = False
            for s in subs:
                if any(s in k for k in keys):
                    ok = True
                    break
            out[name] = bool(ok)
            continue

        if kind == "key_exists":
            key = str(r.get("key") or "")
            out[name] = bool(key and (key in feats))
            continue

        if kind in (
            "value_gt",
            "value_gte",
            "value_lt",
            "value_lte",
            "abs_gt",
            "quantile_gt",
            "quantile_gte",
            "quantile_lt",
            "quantile_lte",
            "quantile_abs_gt",
        ):
            key = str(r.get("key") or "")
            if not key or key not in feats:
                out[name] = _missing()
                continue
            v = _to_float(feats.get(key))
            if v is None:
                out[name] = _missing()
                continue
            thr = None
            if kind.startswith("quantile_"):
                q = _to_float(r.get("quantile"))
                thr = _get_quantile_threshold(quantiles=quantiles, key=key, q=q or 0.0)
            else:
                thr = _to_float(r.get("threshold"))
            if thr is None:
                # No threshold means invalid rule => false (strict)
                out[name] = False
                continue
            if kind in ("value_gt", "quantile_gt"):
                out[name] = bool(v > thr)
            elif kind in ("value_gte", "quantile_gte"):
                out[name] = bool(v >= thr)
            elif kind in ("value_lt", "quantile_lt"):
                out[name] = bool(v < thr)
            elif kind in ("value_lte", "quantile_lte"):
                out[name] = bool(v <= thr)
            else:  # abs_gt
                out[name] = bool(abs(v) > thr)
            continue

        # Unknown kinds default to false to keep the contract strict/extendable.
        out[name] = False

    return out
