from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.tree import DecisionTreeClassifier, _tree  # type: ignore
except Exception:  # pragma: no cover
    DecisionTreeClassifier = None  # type: ignore
    _tree = None  # type: ignore


@dataclass(frozen=True)
class TreeGateTrainConfig:
    max_depth: int = 4
    min_samples_leaf: int = 100
    class_weight: Optional[str] = "balanced"  # keeps veto/allow sane under imbalance
    random_state: int = 42


@dataclass(frozen=True)
class TreeGateArtifact:
    gate_name: str
    feature_names: List[str]
    model_json: Dict[str, Any]
    rules: Dict[str, Any]
    metrics: Dict[str, Any]


def _require_sklearn() -> None:
    if DecisionTreeClassifier is None:
        raise ImportError("scikit-learn is required for TreeGateClassifier")


def train_tree_gate(
    X: np.ndarray,
    y: np.ndarray,
    *,
    gate_name: str,
    feature_names: Sequence[str],
    cfg: TreeGateTrainConfig = TreeGateTrainConfig(),
) -> Any:
    """
    Train a gate classifier:
      y=1 => allow
      y=0 => veto
    """
    _require_sklearn()
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X/y length mismatch")
    if X.shape[1] != len(feature_names):
        raise ValueError("feature_names length mismatch")

    clf = DecisionTreeClassifier(
        max_depth=int(cfg.max_depth),
        min_samples_leaf=int(cfg.min_samples_leaf),
        class_weight=cfg.class_weight,
        random_state=int(cfg.random_state),
    )
    clf.fit(X, y)
    return clf


def _tree_to_rules(
    clf: Any,
    feature_names: Sequence[str],
    *,
    depth_limit: int = 4,
    node_id: int = 0,
    depth: int = 0,
) -> Dict[str, Any]:
    """
    Export a compact, auditable rule tree (YAML/JSON friendly).
    """
    if _tree is None:
        return {}
    tree = clf.tree_
    if depth >= depth_limit:
        # leaf summary
        value = tree.value[node_id][0]
        n0, n1 = float(value[0]), (
            float(value[1]) if len(value) > 1 else (float(value[0]), 0.0)
        )
        allow_rate = float(n1 / max(1.0, n0 + n1))
        pred = int(np.argmax(value))
        return {
            "type": "leaf",
            "pred": int(pred),
            "allow_rate": allow_rate,
            "n": float(n0 + n1),
        }

    feat = int(tree.feature[node_id])
    thr = float(tree.threshold[node_id])
    left = int(tree.children_left[node_id])
    right = int(tree.children_right[node_id])

    # Leaf?
    if left == right:
        value = tree.value[node_id][0]
        n0, n1 = float(value[0]), (
            float(value[1]) if len(value) > 1 else (float(value[0]), 0.0)
        )
        allow_rate = float(n1 / max(1.0, n0 + n1))
        pred = int(np.argmax(value))
        return {
            "type": "leaf",
            "pred": int(pred),
            "allow_rate": allow_rate,
            "n": float(n0 + n1),
        }

    name = str(feature_names[feat]) if 0 <= feat < len(feature_names) else f"f{feat}"
    return {
        "type": "split",
        "feature": name,
        "threshold": thr,
        "left": _tree_to_rules(
            clf, feature_names, depth_limit=depth_limit, node_id=left, depth=depth + 1
        ),
        "right": _tree_to_rules(
            clf, feature_names, depth_limit=depth_limit, node_id=right, depth=depth + 1
        ),
    }


def gate_predict(clf: Any, X: np.ndarray) -> np.ndarray:
    """
    Return gate decisions:
      1 => allow
      0 => veto
    """
    X = np.asarray(X, dtype=float)
    return np.asarray(clf.predict(X), dtype=int)


def evaluate_gate_effect(
    *,
    allow: np.ndarray,
    ret_used: np.ndarray,
    tail_q: float = 0.05,
) -> Dict[str, Any]:
    """
    Gate KPIs (do NOT optimize Sharpe here):
    - activation_rate: allow ratio
    - veto_loss_avoided: avg(ret | veto) - avg(ret | allow)  (more negative veto => good)
    - false_reject_rate: fraction of vetoed trades with positive ret
    - tail_loss_reduction: q-tail mean reduction on allowed trades vs all trades
    """
    a = np.asarray(allow, dtype=int)
    r = np.asarray(ret_used, dtype=float)
    if a.size == 0:
        return {"activation_rate": 0.0}
    mask_all = np.isfinite(r)
    a = a[mask_all]
    r = r[mask_all]
    if r.size == 0:
        return {"activation_rate": 0.0}
    allow_mask = a == 1
    veto_mask = a == 0
    act = float(np.mean(allow_mask.astype(float)))

    def _avg(x: np.ndarray) -> float:
        return float(np.mean(x)) if x.size else 0.0

    veto_loss_avoided = float(_avg(r[veto_mask]) - _avg(r[allow_mask]))
    false_reject_rate = (
        float(np.mean((r[veto_mask] > 0.0).astype(float))) if np.any(veto_mask) else 0.0
    )

    q = float(np.clip(tail_q, 1e-6, 0.5))
    thr = float(np.quantile(r, q))
    tail_all = r[r <= thr]
    tail_allow = r[allow_mask & (r <= thr)]
    tail_loss_reduction = (
        float(_avg(tail_all) - _avg(tail_allow)) if tail_all.size else 0.0
    )

    return {
        "activation_rate": act,
        "veto_loss_avoided": veto_loss_avoided,
        "false_reject_rate": false_reject_rate,
        "tail_loss_reduction": tail_loss_reduction,
        "n": int(r.size),
        "n_allow": int(np.sum(allow_mask)),
        "n_veto": int(np.sum(veto_mask)),
    }


def export_tree_gate_artifact(
    *,
    clf: Any,
    gate_name: str,
    feature_names: Sequence[str],
    metrics: Dict[str, Any],
    rules_depth_limit: int = 4,
) -> TreeGateArtifact:
    rules = _tree_to_rules(clf, feature_names, depth_limit=int(rules_depth_limit))
    model_json = {
        "type": "sklearn.DecisionTreeClassifier",
        "params": getattr(clf, "get_params", lambda: {})(),
    }
    return TreeGateArtifact(
        gate_name=str(gate_name),
        feature_names=[str(x) for x in feature_names],
        model_json=dict(model_json),
        rules=dict(rules),
        metrics=dict(metrics or {}),
    )


def save_tree_gate_artifact(artifact: TreeGateArtifact, *, out_dir: str | Path) -> None:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "meta.json").write_text(
        json.dumps(
            {
                "gate_name": artifact.gate_name,
                "feature_names": artifact.feature_names,
                "model": artifact.model_json,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (p / "rules.json").write_text(
        json.dumps(artifact.rules, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (p / "metrics.json").write_text(
        json.dumps(artifact.metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
