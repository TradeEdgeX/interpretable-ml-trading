"""
nnmultihead primitives factor-eval

Goal:
- Evaluate candidate feature columns against *path primitives* labels:
  dir_y, mfe_atr, mae_atr, t_to_mfe (computed from OHLC + atr).
- Produce a Pool-B style YAML: features_pool_b_primitives.yaml

Why this exists:
- Tree-side factor-eval targets strategy labels (returns / win-rate).
- nnmultihead needs features that are predictive for primitives heads (router inputs).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.feature_store.feature_store import FeatureStore, FeatureStoreSpec
from src.time_series_model.models.nn.path_primitives_labels import (
    PathPrimitivesLabelConfig,
    compute_path_primitives_labels,
)


EPS = 1e-12


@dataclass(frozen=True)
class GroupStats:
    n_groups: int
    n_samples_total: int
    mean: float
    std: float
    t_stat: float
    ir: float


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _feature_deps_map(feature_deps_yaml: Dict[str, Any]) -> Dict[str, Any]:
    return feature_deps_yaml.get("features", {}) or {}


def _build_output_col_to_feature_func(
    feature_deps_yaml: Dict[str, Any],
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    features = _feature_deps_map(feature_deps_yaml)
    for feat_name, info in features.items():
        for col in info.get("output_columns") or []:
            out[str(col)] = str(feat_name)
    return out


def _expand_candidates_to_output_cols(
    *,
    candidates: List[str],
    feature_deps_yaml: Dict[str, Any],
) -> List[str]:
    """
    Accept candidates as either:
    - feature functions (xxx_f)
    - output columns
    Expand to output-column list for evaluation.
    """
    features = _feature_deps_map(feature_deps_yaml)
    out_cols: List[str] = []
    for item in candidates:
        item = str(item)
        if item in features:
            cols = features[item].get("output_columns") or []
            out_cols.extend([str(c) for c in cols])
        else:
            out_cols.append(item)
    # de-dupe while preserving order
    seen = set()
    deduped = []
    for c in out_cols:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def _read_feature_store_range(
    *,
    features_store_root: str,
    layer: str,
    symbols: List[str],
    timeframe: str,
    start: Optional[str],
    end: Optional[str],
) -> pd.DataFrame:
    store = FeatureStore(str(features_store_root))
    parts = []
    for sym in symbols:
        spec = FeatureStoreSpec(
            layer=str(layer), symbol=str(sym), timeframe=str(timeframe)
        )
        start_ts = pd.Timestamp(start) if start else pd.Timestamp("1970-01-01")
        end_ts = pd.Timestamp(end) if end else pd.Timestamp("2100-01-01")
        df_sym = store.read_range(spec, start=start_ts, end=end_ts)
        if df_sym.empty:
            raise ValueError(f"Empty FeatureStore read for symbol={sym}, layer={layer}")
        if "symbol" not in df_sym.columns:
            df_sym = df_sym.copy()
            df_sym["symbol"] = sym
        parts.append(df_sym)
    df = pd.concat(parts, axis=0, ignore_index=False)
    # FeatureStore convention: timestamp is often the index name, not a column.
    if "timestamp" not in df.columns:
        if getattr(df.index, "name", None) == "timestamp":
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df.index, utc=False, errors="coerce")
        else:
            raise KeyError(
                "Expected FeatureStore data to have a 'timestamp' column or index named 'timestamp'"
            )
    return df


def _month_key(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    return t.dt.to_period("M").astype(str)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    # fast-ish spearman via rank transform
    if x.size < 3:
        return np.nan
    xr = pd.Series(x).rank(pct=False).to_numpy(dtype=float)
    yr = pd.Series(y).rank(pct=False).to_numpy(dtype=float)
    if not (np.isfinite(xr).all() and np.isfinite(yr).all()):
        m = np.isfinite(xr) & np.isfinite(yr)
        xr = xr[m]
        yr = yr[m]
    if xr.size < 3:
        return np.nan
    cx = xr - xr.mean()
    cy = yr - yr.mean()
    denom = float(np.sqrt(np.sum(cx * cx) * np.sum(cy * cy)))
    if denom <= 0:
        return np.nan
    return float(np.sum(cx * cy) / denom)


def _binary_auc(x: np.ndarray, y01: np.ndarray) -> float:
    # sklearn is optional; fallback to rank-based AUC if missing
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore

        if len(np.unique(y01)) < 2:
            return np.nan
        return float(roc_auc_score(y01, x))
    except Exception:
        # Mann–Whitney U: AUC = (rank_sum_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
        y01 = y01.astype(int)
        if len(np.unique(y01)) < 2:
            return np.nan
        r = pd.Series(x).rank().to_numpy(dtype=float)
        pos = y01 == 1
        n_pos = int(pos.sum())
        n_neg = int((~pos).sum())
        if n_pos == 0 or n_neg == 0:
            return np.nan
        rank_sum_pos = float(r[pos].sum())
        return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _aggregate_group_values(vals: List[float], ns: List[int]) -> GroupStats:
    if not vals:
        return GroupStats(
            n_groups=0,
            n_samples_total=int(np.sum(ns)) if ns else 0,
            mean=float("nan"),
            std=float("nan"),
            t_stat=float("nan"),
            ir=float("nan"),
        )

    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    n_groups = int(v.size)
    if n_groups <= 1:
        return GroupStats(
            n_groups=n_groups,
            n_samples_total=int(np.sum(ns)),
            mean=float(v[0]) if n_groups == 1 else float("nan"),
            std=float("nan"),
            t_stat=float("nan"),
            ir=float("nan"),
        )

    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1))
    # If across-group std is ~0, the "IR/t-stat" is not meaningful; treat as no signal.
    if not np.isfinite(std) or std <= 1e-8:
        return GroupStats(
            n_groups=n_groups,
            n_samples_total=int(np.sum(ns)),
            mean=mean,
            std=std,
            t_stat=0.0,
            ir=0.0,
        )
    t_stat = float(mean / (std / np.sqrt(n_groups)))
    ir = float(mean / std)
    return GroupStats(
        n_groups=n_groups,
        n_samples_total=int(np.sum(ns)),
        mean=mean,
        std=std,
        t_stat=t_stat,
        ir=ir,
    )


def _eval_factor_against_targets(
    *,
    df: pd.DataFrame,
    factor_col: str,
    min_samples_per_group: int,
) -> Dict[str, Any]:
    """
    Evaluate one factor column against labels already present in df:
    dir_y, mfe_atr, mae_atr, t_to_mfe
    Using per-(symbol,month) groups => stability (ICIR-like).
    """
    x_all = pd.to_numeric(df[factor_col], errors="coerce").to_numpy(dtype=float)
    out: Dict[str, Any] = {"factor_col": factor_col}

    group_cols = ["symbol", "month"]
    grouped = df[group_cols].copy()
    grouped["_i"] = np.arange(len(df), dtype=int)

    targets = [
        ("dir_y", "auc"),
        ("dir_y", "spearman"),
        ("mfe_atr", "spearman"),
        ("mae_atr", "spearman"),
        ("t_to_mfe", "spearman"),
    ]

    for tgt, metric in targets:
        vals: List[float] = []
        ns: List[int] = []
        for (_, _), g in grouped.groupby(group_cols, sort=False):
            idx = g["_i"].to_numpy(dtype=int)
            y = pd.to_numeric(df.iloc[idx][tgt], errors="coerce").to_numpy(dtype=float)
            x = x_all[idx]
            m = np.isfinite(x) & np.isfinite(y)
            if int(m.sum()) < int(min_samples_per_group):
                continue
            x = x[m]
            y = y[m]
            if metric == "spearman":
                v = _spearman(x, y)
            elif metric == "auc":
                # dir_y expected 0/1
                y01 = (y > 0.5).astype(int)
                auc_raw = _binary_auc(x, y01)
                # Center AUC around 0.5 so "no-skill" => 0.0; prevents constant factors looking great.
                v = float(auc_raw - 0.5) if np.isfinite(auc_raw) else np.nan
            else:
                v = np.nan
            if np.isfinite(v):
                vals.append(float(v))
                ns.append(int(len(x)))

        stats = _aggregate_group_values(vals, ns)
        key = f"{tgt}__{metric}"
        out[f"{key}__n_groups"] = stats.n_groups
        out[f"{key}__n_samples"] = stats.n_samples_total
        out[f"{key}__mean"] = stats.mean
        out[f"{key}__std"] = stats.std
        out[f"{key}__tstat"] = stats.t_stat
        out[f"{key}__ir"] = stats.ir

    # simple missingness
    out["nan_rate"] = float(np.mean(~np.isfinite(x_all)))
    return out


def _select_qualified_feature_funcs(
    *,
    metrics_df: pd.DataFrame,
    output_col_to_feat_func: Dict[str, str],
    min_abs_ir: float,
    min_abs_tstat: float,
    max_nan_rate: float,
) -> List[str]:
    """
    Select feature compute functions (xxx_f) if any of their output columns qualifies
    on any target metric.
    """
    qualified_funcs: List[str] = []
    seen = set()

    # columns to consider for qualification
    score_cols = [
        "dir_y__auc__ir",
        "dir_y__spearman__ir",
        "mfe_atr__spearman__ir",
        "mae_atr__spearman__ir",
        "t_to_mfe__spearman__ir",
    ]
    tstat_cols = [c.replace("__ir", "__tstat") for c in score_cols]

    for _, r in metrics_df.iterrows():
        try:
            nan_rate = float(r.get("nan_rate", 1.0))
        except Exception:
            nan_rate = 1.0
        if nan_rate > float(max_nan_rate):
            continue

        passed = False
        for sc, tc in zip(score_cols, tstat_cols):
            ir = r.get(sc, np.nan)
            ts = r.get(tc, np.nan)
            if np.isfinite(ir) and np.isfinite(ts):
                if (abs(float(ir)) >= float(min_abs_ir)) and (
                    abs(float(ts)) >= float(min_abs_tstat)
                ):
                    passed = True
                    break
        if not passed:
            continue

        factor_col = str(r["factor_col"])
        feat_func = output_col_to_feat_func.get(factor_col, None)
        if feat_func is None:
            continue
        if feat_func not in seen:
            seen.add(feat_func)
            qualified_funcs.append(feat_func)

    return qualified_funcs


def _write_pool_b_yaml(
    *,
    out_path: str | Path,
    requested_features: List[str],
    comment: str,
) -> None:
    cfg = {
        "feature_pipeline": {
            "requested_features": requested_features,
            "invert_features": [],
            "post_processors": [],
            "selector": None,
        },
        "_comment": comment.strip(),
    }
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Primitives factor-eval (nnmultihead).")
    p.add_argument(
        "--config-dir", required=True, help="nnmultihead config dir (for provenance)"
    )
    p.add_argument(
        "--candidates-yaml",
        required=True,
        help="YAML with feature_pipeline.requested_features (e.g., strategy/features_all.yaml)",
    )
    p.add_argument("--feature-deps", default="config/feature_dependencies.yaml")
    p.add_argument(
        "--symbols", required=True, help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT"
    )
    p.add_argument("--timeframe", default="240T")
    p.add_argument("--features-store-root", default="feature_store")
    p.add_argument(
        "--features-store-layer", required=True, help="FeatureStore layer id"
    )
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)

    p.add_argument("--horizon-hours", type=float, default=80.0)
    p.add_argument("--bar-hours", type=float, default=4.0)

    p.add_argument("--min-samples-per-group", type=int, default=200)
    p.add_argument("--max-nan-rate", type=float, default=0.5)
    p.add_argument("--min-abs-ir", type=float, default=0.05)
    p.add_argument("--min-abs-tstat", type=float, default=1.96)

    p.add_argument(
        "--output-dir", default=None, help="Directory for CSV/JSON artifacts"
    )
    p.add_argument(
        "--export-yaml", default=None, help="Write features_pool_b_primitives.yaml here"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_dir = Path(args.config_dir).resolve()
    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    if not symbols:
        raise ValueError("No symbols provided")

    feature_deps = _load_yaml(args.feature_deps)
    out_col_to_feat = _build_output_col_to_feature_func(feature_deps)

    candidates_yaml = _load_yaml(args.candidates_yaml)
    candidates = (candidates_yaml.get("feature_pipeline") or {}).get(
        "requested_features"
    ) or []
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(
            "candidates-yaml must contain feature_pipeline.requested_features as a non-empty list"
        )
    candidate_output_cols = _expand_candidates_to_output_cols(
        candidates=[str(x) for x in candidates], feature_deps_yaml=feature_deps
    )

    # Read features
    df = _read_feature_store_range(
        features_store_root=str(args.features_store_root),
        layer=str(args.features_store_layer),
        symbols=symbols,
        timeframe=str(args.timeframe),
        start=args.start_date,
        end=args.end_date,
    )
    df = df.copy()
    df["month"] = _month_key(df["timestamp"])

    # Compute primitives labels (group-safe)
    horizon_bars = int(round(float(args.horizon_hours) / float(args.bar_hours)))
    if horizon_bars <= 0:
        raise ValueError(f"Invalid horizon_bars computed: {horizon_bars}")
    label_cfg = PathPrimitivesLabelConfig(horizon_bars=horizon_bars)
    labels = compute_path_primitives_labels(df, cfg=label_cfg, group_col="symbol")
    for c in labels.columns:
        df[c] = labels[c]

    # Evaluate all candidate columns that exist
    existing_cols = [c for c in candidate_output_cols if c in df.columns]
    missing_cols = [c for c in candidate_output_cols if c not in df.columns]
    if not existing_cols:
        raise ValueError(
            "None of the candidate columns exist in the FeatureStore layer"
        )

    rows: List[Dict[str, Any]] = []
    for col in existing_cols:
        try:
            rows.append(
                _eval_factor_against_targets(
                    df=df,
                    factor_col=str(col),
                    min_samples_per_group=int(args.min_samples_per_group),
                )
            )
        except Exception as e:
            rows.append({"factor_col": str(col), "error": str(e)})

    metrics_df = pd.DataFrame(rows)

    # Default output locations
    if args.output_dir is None:
        out_dir = Path("results") / "pools" / cfg_dir.name / "pool_b_primitives"
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = out_dir / "primitives_factor_eval_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    summary = {
        "config_dir": str(cfg_dir),
        "candidates_yaml": str(Path(args.candidates_yaml).resolve()),
        "features_store_root": str(Path(args.features_store_root).resolve()),
        "features_store_layer": str(args.features_store_layer),
        "timeframe": str(args.timeframe),
        "symbols": symbols,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "horizon_hours": float(args.horizon_hours),
        "bar_hours": float(args.bar_hours),
        "horizon_bars": int(horizon_bars),
        "n_candidate_output_cols": int(len(candidate_output_cols)),
        "n_existing_output_cols": int(len(existing_cols)),
        "n_missing_output_cols": int(len(missing_cols)),
        "missing_output_cols_sample": missing_cols[:50],
        "metrics_csv": str(metrics_csv),
    }
    (out_dir / "primitives_factor_eval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    qualified_funcs = _select_qualified_feature_funcs(
        metrics_df=metrics_df,
        output_col_to_feat_func=out_col_to_feat,
        min_abs_ir=float(args.min_abs_ir),
        min_abs_tstat=float(args.min_abs_tstat),
        max_nan_rate=float(args.max_nan_rate),
    )

    if args.export_yaml is None:
        export_yaml = out_dir / "features_pool_b_primitives.yaml"
    else:
        export_yaml = Path(args.export_yaml)

    comment = f"""
Primitives Pool B (nnmultihead) exported by factor_primitives_eval.

Targets: dir_y (AUC & Spearman), mfe_atr/mae_atr/t_to_mfe (Spearman).
Grouping: per-(symbol, month) IC stability (IR = mean/std).

Selection thresholds:
- abs(IR) >= {float(args.min_abs_ir)}
- abs(t-stat) >= {float(args.min_abs_tstat)}
- nan_rate <= {float(args.max_nan_rate)}

Artifacts:
- metrics: {metrics_csv}
- summary: {out_dir / 'primitives_factor_eval_summary.json'}
""".strip()

    _write_pool_b_yaml(
        out_path=export_yaml, requested_features=qualified_funcs, comment=comment
    )

    print(f"✅ Wrote metrics CSV: {metrics_csv}")
    print(f"✅ Wrote Pool B YAML: {export_yaml}  (n={len(qualified_funcs)})")


if __name__ == "__main__":
    main()
