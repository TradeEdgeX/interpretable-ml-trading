"""Runtime veto from a trained adverse-excursion tree (gate.yaml + model.joblib)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import yaml

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_gate_model_path(raw: str, *, strategies_root: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    for base in (_repo_root(), strategies_root, Path.cwd()):
        cand = (base / p).resolve()
        if cand.is_file():
            return cand
    return (_repo_root() / p).resolve()


class AdverseTreeGateVeto:
    """Veto entries when P(adverse | gate features) exceeds ``reject_if_prob_bad_gt``."""

    def __init__(
        self,
        *,
        clf: Any,
        feature_names: List[str],
        reject_threshold: float,
    ) -> None:
        self._clf = clf
        self._feature_names = list(feature_names)
        self._reject_threshold = float(reject_threshold)
        self._bad_class_idx = self._bad_class_index()

    def _bad_class_index(self) -> int:
        classes = list(getattr(self._clf, "classes_", [0, 1]))
        try:
            return classes.index(0)
        except ValueError:
            return 0

    @classmethod
    def from_gate_yaml(
        cls,
        gate_yaml: Path,
        *,
        strategies_root: Path | str = "config/strategies",
    ) -> Optional["AdverseTreeGateVeto"]:
        if not gate_yaml.is_file():
            return None
        raw = yaml.safe_load(gate_yaml.read_text(encoding="utf-8")) or {}
        if not bool(raw.get("enabled", False)):
            return None
        model_rel = raw.get("gate_model")
        if not model_rel:
            return None
        names = list(raw.get("gate_feature_names") or [])
        if not names:
            logger.warning(
                "adverse tree gate enabled but gate_feature_names empty: %s", gate_yaml
            )
            return None
        model_path = _resolve_gate_model_path(
            str(model_rel), strategies_root=Path(strategies_root)
        )
        if not model_path.is_file():
            logger.warning("adverse tree gate model not found: %s", model_path)
            return None
        reject_thr = float(raw.get("reject_if_prob_bad_gt", 0.55))
        clf = joblib.load(model_path)
        logger.info(
            "Loaded adverse tree gate: %s features=%d reject_if_prob_bad_gt=%.2f",
            model_path,
            len(names),
            reject_thr,
        )
        return cls(clf=clf, feature_names=names, reject_threshold=reject_thr)

    def evaluate(self, features: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Return (passed, reasons). passed=False => veto entry."""
        vec: List[float] = []
        for name in self._feature_names:
            raw = features.get(name)
            if raw is None:
                return True, []
            try:
                v = float(raw)
            except (TypeError, ValueError):
                return True, []
            if v != v:
                return True, []
            vec.append(v)
        x = np.asarray([vec], dtype=float)
        p_bad: float
        if hasattr(self._clf, "predict_proba"):
            proba = np.asarray(self._clf.predict_proba(x)[0], dtype=float)
            p_bad = float(proba[self._bad_class_idx])
        else:
            p_bad = 1.0 - float(self._clf.predict(x)[0])
        if p_bad > self._reject_threshold:
            return False, [
                f"tree_gate_veto:p_bad={p_bad:.3f}>thr={self._reject_threshold:.3f}"
            ]
        return True, []
