"""Promote-evidence staleness: FeatureStore layer newer than the experiment date.

Canonical trigger: ``20260907_srb_eth_exclusion_counterfactual`` — the
``size_1x`` promote archive stopped reproducing after an in-place FeatureStore
rebuild (same layer hash, newer parquet mtimes).

A row is **stale** only when:

1. the experiment has a ``promote`` verdict (declared or inferred),
2. a FeatureStore layer can be resolved, and
3. that layer's newest write date is **strictly after** the experiment
   directory date (``YYYYMMDD_…``).

No layer / no date → ``unknown`` (not stale). Rejects are scored the same
way but the CLI default only lists promote rows. This module never writes
``verdict``.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.feature_store.layer_naming import detect_layer_for_strategy
from src.research.experiment_index import (
    find_decision_file,
    parse_front_matter,
    read_experiment_meta,
)

LAYER_NAME_RE = re.compile(r"\b(features_[a-z0-9]+_\d+[TDh]_[a-f0-9]{6,})\b")

# Families whose Phase 3 numbers come from a crypto FeatureStore layer.
_FS_STRATEGIES = frozenset(
    {
        "srb",
        "tpc",
        "bpc",
        "me",
        "largebar_fade",
        "sr_fade",
        "sr_confirm",
        "chop_grid",
        "ma_cross",
    }
)


@dataclass
class StaleRow:
    experiment_id: str
    verdict: Optional[str]
    status: str  # stale | fresh | unknown
    experiment_date: Optional[str]
    layer: Optional[str]
    layer_date: Optional[str]
    reasons: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def stale(self) -> bool:
        return self.status == "stale"


def experiment_date_from_id(experiment_id: str) -> Optional[date]:
    if len(experiment_id) < 8 or not experiment_id[:8].isdigit():
        return None
    try:
        return date(int(experiment_id[:4]), int(experiment_id[4:6]), int(experiment_id[6:8]))
    except ValueError:
        return None


def _parse_iso_date(raw: Any) -> Optional[date]:
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None


def layer_rebuilt_at(
    store_root: Path,
    layer: str,
    *,
    cache: Optional[Dict[str, Optional[date]]] = None,
) -> Optional[date]:
    """Newest write date of a FeatureStore layer (UTC calendar day)."""
    if cache is not None and layer in cache:
        return cache[layer]
    newest: Optional[float] = None
    meta_path = store_root / f"{layer}.meta.json"
    if meta_path.is_file():
        try:
            blob = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            blob = {}
        stamped = _parse_iso_date(blob.get("last_write_at") or blob.get("built_at"))
        if stamped is not None:
            newest = datetime.combine(
                stamped, datetime.min.time(), tzinfo=timezone.utc
            ).timestamp()
        else:
            newest = meta_path.stat().st_mtime
    layer_dir = store_root / layer
    if layer_dir.is_dir():
        for path in layer_dir.rglob("*.parquet"):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if newest is None or mtime > newest:
                newest = mtime
        if newest is None:
            try:
                newest = layer_dir.stat().st_mtime
            except OSError:
                newest = None
    resolved: Optional[date] = None
    if newest is not None:
        resolved = datetime.fromtimestamp(newest, tz=timezone.utc).date()
    if cache is not None:
        cache[layer] = resolved
    return resolved


def cited_layers(text: str) -> List[str]:
    return sorted(set(LAYER_NAME_RE.findall(text or "")))


def resolve_layer_name(
    exp_dir: Path,
    *,
    repo_root: Path,
    store_root: Optional[Path] = None,
) -> Optional[str]:
    """Front-matter → cited layer in the decision doc → strategy auto-detect."""
    store = store_root or (repo_root / "feature_store")
    decision = find_decision_file(exp_dir)
    text = ""
    raw: Dict[str, Any] = {}
    if decision and decision.is_file():
        try:
            text = decision.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        raw, _ = parse_front_matter(text)
    explicit = str(raw.get("feature_store_layer") or "").strip()
    if explicit:
        return explicit
    for name in cited_layers(text):
        if (store / name).is_dir() or (store / f"{name}.meta.json").is_file():
            return name
    meta = read_experiment_meta(exp_dir)
    strategy = meta.strategy
    if strategy and strategy in _FS_STRATEGIES and store.is_dir():
        return detect_layer_for_strategy(
            strategy, features_store_root=str(store)
        )
    return None


def evaluate_staleness(
    exp_dir: Path,
    *,
    repo_root: Path,
    store_root: Optional[Path] = None,
    layer_cache: Optional[Dict[str, Optional[date]]] = None,
) -> StaleRow:
    store = store_root or (repo_root / "feature_store")
    meta = read_experiment_meta(exp_dir)
    exp_date = experiment_date_from_id(exp_dir.name)
    layer = resolve_layer_name(exp_dir, repo_root=repo_root, store_root=store)
    reasons: List[str] = []
    if exp_date is None:
        reasons.append("experiment date missing from directory name")
    if not layer:
        reasons.append("feature store layer unresolved")
        return StaleRow(
            experiment_id=exp_dir.name,
            verdict=meta.verdict,
            status="unknown",
            experiment_date=exp_date.isoformat() if exp_date else None,
            layer=None,
            layer_date=None,
            reasons=reasons,
        )
    layer_date = layer_rebuilt_at(store, layer, cache=layer_cache)
    if layer_date is None:
        reasons.append(f"layer {layer} has no mtime")
        return StaleRow(
            experiment_id=exp_dir.name,
            verdict=meta.verdict,
            status="unknown",
            experiment_date=exp_date.isoformat() if exp_date else None,
            layer=layer,
            layer_date=None,
            reasons=reasons,
        )
    if exp_date is None:
        return StaleRow(
            experiment_id=exp_dir.name,
            verdict=meta.verdict,
            status="unknown",
            experiment_date=None,
            layer=layer,
            layer_date=layer_date.isoformat(),
            reasons=reasons,
        )
    if layer_date > exp_date:
        reasons.append(
            f"feature store layer {layer} written {layer_date.isoformat()} "
            f"> experiment date {exp_date.isoformat()}"
        )
        return StaleRow(
            experiment_id=exp_dir.name,
            verdict=meta.verdict,
            status="stale",
            experiment_date=exp_date.isoformat(),
            layer=layer,
            layer_date=layer_date.isoformat(),
            reasons=reasons,
        )
    return StaleRow(
        experiment_id=exp_dir.name,
        verdict=meta.verdict,
        status="fresh",
        experiment_date=exp_date.isoformat(),
        layer=layer,
        layer_date=layer_date.isoformat(),
        reasons=[],
    )


def scan_promote_staleness(
    *,
    repo_root: Path,
    experiments_root: Optional[Path] = None,
    store_root: Optional[Path] = None,
    verdicts: Sequence[str] = ("promote",),
) -> List[StaleRow]:
    from src.research.experiment_index import iter_experiment_dirs

    root = experiments_root or (repo_root / "config" / "experiments")
    store = store_root or (repo_root / "feature_store")
    cache: Dict[str, Optional[date]] = {}
    wanted = {v.lower() for v in verdicts}
    rows: List[StaleRow] = []
    for exp_dir in iter_experiment_dirs(root):
        meta = read_experiment_meta(exp_dir)
        if (meta.verdict or "").lower() not in wanted:
            continue
        rows.append(
            evaluate_staleness(
                exp_dir,
                repo_root=repo_root,
                store_root=store,
                layer_cache=cache,
            )
        )
    return rows
