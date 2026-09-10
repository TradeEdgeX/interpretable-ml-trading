from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class OODBin:
    name: str
    lo: float
    hi: float


@dataclass(frozen=True)
class OODToArchetypeTableConfig:
    version: int
    name: str

    symbol_col: str
    timestamp_col: str
    ood_score_col: str
    archetype_col: str
    label_col: str

    archetypes: List[str]
    bins: List[OODBin]

    temperature: float
    min_samples_per_cell: int
    default_weight_if_missing: float


def _default_ood_to_archetype_table_config() -> OODToArchetypeTableConfig:
    """In-code default when config/ood was removed. Research-only; live uses constitution."""
    return OODToArchetypeTableConfig(
        version=1,
        name="ood_to_archetype_table_default",
        symbol_col="symbol",
        timestamp_col="timestamp",
        ood_score_col="ood_score",
        archetype_col="active_archetype",
        label_col="y_surv",
        archetypes=[
            "TrendContinuationTC",
            "TrendExpansionTE",
            "FailureReversionFR",
            "ExhaustionTurnET",
        ],
        bins=[
            OODBin(name="low_ood", lo=0.0, hi=0.4),
            OODBin(name="mid_ood", lo=0.4, hi=0.7),
            OODBin(name="high_ood", lo=0.7, hi=1.0),
        ],
        temperature=0.15,
        min_samples_per_cell=200,
        default_weight_if_missing=0.0,
    )


def load_ood_to_archetype_table_config(
    path: str | Path = "config/ood/ood_to_archetype_table.yaml",
) -> OODToArchetypeTableConfig:
    p = Path(path)
    if not p.exists():
        return _default_ood_to_archetype_table_config()

    obj = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cols = obj.get("columns") or {}
    bins_obj = obj.get("bins") or []
    ws = obj.get("weights") or {}

    bins: List[OODBin] = []
    if isinstance(bins_obj, list):
        for b in bins_obj:
            if not isinstance(b, dict):
                continue
            bins.append(
                OODBin(
                    name=str(b.get("name") or ""),
                    lo=float(b.get("lo", 0.0)),
                    hi=float(b.get("hi", 1.0)),
                )
            )
    return OODToArchetypeTableConfig(
        version=int(obj.get("version", 1)),
        name=str(obj.get("name", "ood_to_archetype_table")),
        symbol_col=str(cols.get("symbol_col", "symbol")),
        timestamp_col=str(cols.get("timestamp_col", "timestamp")),
        ood_score_col=str(cols.get("ood_score_col", "ood_score")),
        archetype_col=str(cols.get("archetype_col", "active_archetype")),
        label_col=str(cols.get("label_col", "y_surv")),
        archetypes=[str(x) for x in (obj.get("archetypes") or [])],
        bins=bins,
        temperature=float(ws.get("temperature", 0.15)),
        min_samples_per_cell=int(ws.get("min_samples_per_cell", 200)),
        default_weight_if_missing=float(ws.get("default_weight_if_missing", 0.0)),
    )


def _softmax(xs: List[float], *, temperature: float) -> List[float]:
    t = float(temperature)
    if t <= 1e-9:
        t = 1e-9
    x = np.asarray(xs, dtype=float) / t
    x = x - np.max(x)
    e = np.exp(x)
    s = float(np.sum(e))
    if s <= 0:
        return [0.0 for _ in xs]
    return [float(v / s) for v in e.tolist()]


def build_conditional_survival_table(
    df: pd.DataFrame,
    *,
    cfg: OODToArchetypeTableConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Output table columns:
      bin, archetype, n, survival_rate, weight
    """
    for c in [cfg.ood_score_col, cfg.archetype_col, cfg.label_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required col: {c}")

    work = df.copy()
    work[cfg.ood_score_col] = pd.to_numeric(
        work[cfg.ood_score_col], errors="coerce"
    ).fillna(0.0)
    work[cfg.label_col] = (
        pd.to_numeric(work[cfg.label_col], errors="coerce").fillna(0.0) > 0.5
    ).astype(int)
    work[cfg.archetype_col] = work[cfg.archetype_col].astype(str)
    work = work[np.isfinite(work[cfg.ood_score_col].to_numpy())].reset_index(drop=True)

    rows = []
    meta: Dict[str, Any] = {"bins": [], "archetypes": list(cfg.archetypes)}

    for b in cfg.bins:
        m = (work[cfg.ood_score_col] >= float(b.lo)) & (
            work[cfg.ood_score_col] < float(b.hi)
            if float(b.hi) < 1.0
            else work[cfg.ood_score_col] <= float(b.hi)
        )
        sub = work[m]
        rates = []
        ns = []
        for a in cfg.archetypes:
            g = sub[sub[cfg.archetype_col] == str(a)]
            n = int(len(g))
            sr = float(g[cfg.label_col].mean()) if n else float("nan")
            ns.append(n)
            # missing cells get default_weight later
            rates.append(sr if np.isfinite(sr) else float("nan"))

        # Prepare softmax inputs: only valid cells with enough samples
        x_for_softmax = []
        valid_mask = []
        for sr, n in zip(rates, ns):
            ok = bool(np.isfinite(sr) and n >= int(cfg.min_samples_per_cell))
            valid_mask.append(ok)
            x_for_softmax.append(float(sr) if ok else float("-inf"))

        # Replace -inf with a very small number for stable softmax
        finite_vals = [v for v in x_for_softmax if np.isfinite(v)]
        base = min(finite_vals) if finite_vals else 0.0
        x = [v if np.isfinite(v) else base - 10.0 for v in x_for_softmax]
        w = (
            _softmax(x, temperature=float(cfg.temperature))
            if any(valid_mask)
            else [0.0 for _ in x]
        )

        for a, n, sr, ww, ok in zip(cfg.archetypes, ns, rates, w, valid_mask):
            weight = float(ww) if ok else float(cfg.default_weight_if_missing)
            rows.append(
                {
                    "bin": str(b.name),
                    "ood_lo": float(b.lo),
                    "ood_hi": float(b.hi),
                    "archetype": str(a),
                    "n": int(n),
                    "survival_rate": float(sr) if np.isfinite(sr) else None,
                    "weight": float(weight),
                    "ok": bool(ok),
                }
            )

        meta["bins"].append({"name": b.name, "lo": float(b.lo), "hi": float(b.hi)})

    table = pd.DataFrame(rows)
    return table, meta


def export_weights_yaml(
    table: pd.DataFrame,
    *,
    cfg: OODToArchetypeTableConfig,
) -> Dict[str, Any]:
    """
    Export a deployable mapping:
      ood_bin -> {weights, survival_rate, n}
    """
    out = {
        "version": int(cfg.version),
        "name": str(cfg.name),
        "archetypes": list(cfg.archetypes),
        "bins": [],
        "weights_rule": {
            "type": "softmax_on_survival_rate",
            "temperature": float(cfg.temperature),
            "min_samples_per_cell": int(cfg.min_samples_per_cell),
            "default_weight_if_missing": float(cfg.default_weight_if_missing),
        },
    }
    for b in cfg.bins:
        dfb = table[table["bin"] == str(b.name)].copy()
        weights = {str(r["archetype"]): float(r["weight"]) for _, r in dfb.iterrows()}
        surv = {
            str(r["archetype"]): (
                float(r["survival_rate"]) if pd.notna(r["survival_rate"]) else None
            )
            for _, r in dfb.iterrows()
        }
        nmap = {str(r["archetype"]): int(r["n"]) for _, r in dfb.iterrows()}
        out["bins"].append(
            {
                "name": str(b.name),
                "lo": float(b.lo),
                "hi": float(b.hi),
                "weights": weights,
                "survival_rate": surv,
                "n": nmap,
            }
        )
    return out


def load_any(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)
