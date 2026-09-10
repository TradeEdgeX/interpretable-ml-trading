"""Holdout IC screen: prune feature nodes by |IC| and best_lag (forward_rr target)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

from src.research.stat_kernels.ic import rank_ic, resolve_target_col, shift_target_by_horizon
from src.time_series_model.strategies.models.feature_direction import expand_invert_features

DEFAULT_TARGET = "forward_rr"
DEFAULT_FEATURE_DEPS = Path("config/feature_dependencies.yaml")

_BASE_SKIP = {
    "datetime",
    "timestamp",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
    "_symbol",
    "label",
    "forward_rr",
    "split",
    "sample_weight",
    "atr",
    "atr14",
    "signal",
    "binary_signal",
    "pred",
    "score",
}

InvertMode = Literal["none", "auto"]
WritebackMode = Literal["nodes", "columns"]


def _skip_ic_feature_column(name: str, target: str) -> bool:
    """Exclude meta/target/leakage columns from IC feature screening."""
    if name in _BASE_SKIP:
        return True
    if name == target:
        return True
    if name.startswith("forward_rr_h"):
        return True
    return False


def load_feature_deps(path: Path | None = None) -> dict[str, Any]:
    deps_path = path or DEFAULT_FEATURE_DEPS
    data = yaml.safe_load(deps_path.read_text(encoding="utf-8")) or {}
    return data.get("features", data)


def column_to_nodes(deps: dict[str, Any]) -> dict[str, set[str]]:
    col_to_nodes: dict[str, set[str]] = {}
    for node, info in deps.items():
        if not isinstance(info, dict):
            continue
        for col in info.get("output_columns", []) or []:
            col_to_nodes.setdefault(str(col), set()).add(str(node))
    return col_to_nodes


def holdout_mask(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    if "datetime" in df.columns:
        ts = pd.to_datetime(df["datetime"])
    elif isinstance(df.index, pd.DatetimeIndex):
        ts = pd.Series(df.index, index=df.index)
    else:
        raise ValueError("Need datetime column or DatetimeIndex")
    return (ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))


def target_at_horizon(
    sub: pd.DataFrame, target: str, horizon: int
) -> pd.Series | None:
    tcol, need_shift = resolve_target_col(sub, target, horizon)
    if tcol is None:
        return None
    y = pd.to_numeric(sub[tcol], errors="coerce")
    if need_shift and horizon > 1:
        y = shift_target_by_horizon(y, horizon, sub)
    return y


def parse_allowed_best_lags(raw: str | list[int] | None) -> frozenset[int] | None:
    """Parse comma-separated best_lag whitelist (e.g. ``10,20`` for swing)."""
    if raw is None:
        return None
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if not parts:
            return None
        return frozenset(int(p) for p in parts)
    if not raw:
        return None
    return frozenset(int(x) for x in raw)


def parse_horizons(raw: str | list[int]) -> list[int]:
    if isinstance(raw, list):
        return [int(x) for x in raw]
    return [int(x) for x in str(raw).split(",") if str(x).strip()]


def trim_nodes(
    requested_nodes: list[str],
    *,
    top_n: int | None,
    pool_nodes: set[str] | None,
    always_include: list[str] | None,
) -> list[str]:
    always = list(always_include or [])
    nodes = [n for n in requested_nodes if n not in always]
    if pool_nodes:
        nodes = [n for n in nodes if n in pool_nodes]
    if top_n is not None and top_n > 0:
        cap = max(0, top_n - len(always))
        nodes = nodes[:cap]
    return always + nodes


def split_always_include(
    always: list[str] | None,
    deps: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Split always_include into compute nodes vs model column singletons."""
    nodes: list[str] = []
    columns: list[str] = []
    node_set = set(deps.keys())
    col_to_nodes = column_to_nodes(deps)
    for name in always or []:
        if name in node_set:
            nodes.append(name)
        elif name in col_to_nodes:
            columns.append(name)
        else:
            nodes.append(name)
    return nodes, columns


def expand_requested_features_to_columns(
    requested: list[str],
    deps: dict[str, Any],
) -> set[str]:
    """Expand features.yaml requested_features (nodes + column singletons) to columns."""
    cols: set[str] = set()
    col_to_nodes = column_to_nodes(deps)
    node_set = set(deps.keys())
    for name in requested:
        if name in node_set:
            info = deps.get(name) or {}
            for col in info.get("output_columns") or [name]:
                cols.add(str(col))
        elif name in col_to_nodes:
            cols.add(str(name))
        else:
            cols.add(str(name))
    return cols


def trim_columns(
    rows: list[dict],
    *,
    top_n: int | None,
    pool_columns: set[str] | None,
    always_include_columns: list[str] | None,
) -> list[dict]:
    """Keep top-|IC| column rows (not whole compute nodes)."""
    always = list(always_include_columns or [])
    always_set = set(always)
    ranked = [r for r in rows if r["feature"] not in always_set]
    if pool_columns:
        ranked = [r for r in ranked if r["feature"] in pool_columns]
    if top_n is not None and top_n > 0:
        cap = max(0, top_n - len(always))
        ranked = ranked[:cap]
    selected = list(ranked)
    selected_names = {r["feature"] for r in selected}
    for col in always:
        if col in selected_names:
            continue
        row = next((r for r in rows if r["feature"] == col), None)
        if row is not None:
            selected.append(row)
            selected_names.add(col)
    return selected


def build_compute_request(
    selected_columns: list[str],
    always_include_nodes: list[str],
) -> list[str]:
    """Mixed requested_features: compute nodes + column singletons (sr_breakout hal_low pattern)."""
    out: list[str] = []
    seen: set[str] = set()
    for name in always_include_nodes + selected_columns:
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


def column_rows_with_nodes(
    selected_rows: list[dict],
    deps: dict[str, Any],
) -> list[dict]:
    col_to_nodes = column_to_nodes(deps)
    enriched: list[dict] = []
    for row in selected_rows:
        nodes = sorted(col_to_nodes.get(row["feature"], set()))
        enriched.append({**row, "node": nodes[0] if nodes else None})
    return enriched


def invert_columns_for_rows(rows: list[dict]) -> list[str]:
    return sorted(
        {str(r["feature"]) for r in rows if float(r.get("rank_ic", 0.0)) < 0.0}
    )


def default_model_features_yaml(strategy: str, project_root: Path) -> Path:
    return (
        project_root
        / "config"
        / "strategies"
        / "tree_strategies"
        / strategy
        / "archetypes"
        / "model_features.yaml"
    )


def ic_sign(rank_ic_val: float) -> str:
    if not np.isfinite(rank_ic_val):
        return "?"
    return "+" if rank_ic_val >= 0 else "-"


def screen_features(
    df: pd.DataFrame,
    *,
    holdout_start: str,
    holdout_end: str,
    horizons: list[int],
    min_ic: float,
    max_lag: int,
    min_n: int,
    target: str = DEFAULT_TARGET,
    allowed_best_lags: frozenset[int] | None = None,
    reject_peak_at: int | None = None,
    feature_deps: dict[str, Any] | None = None,
) -> tuple[list[dict], list[dict], list[str]]:
    """Return (column_rows, node_summaries, requested_nodes)."""
    if target not in df.columns:
        raise KeyError(
            f"Target column {target!r} missing from parquet; run prepare-only first"
        )

    mask = holdout_mask(df, holdout_start, holdout_end)
    sub = df.loc[mask].copy()
    feature_cols = [
        c
        for c in sub.columns
        if not _skip_ic_feature_column(str(c), target)
        and not str(c).startswith(("binary_signal", "signal_"))
        and pd.api.types.is_numeric_dtype(sub[c])
    ]

    rows: list[dict] = []
    for feat in feature_cols:
        x = pd.to_numeric(sub[feat], errors="coerce")
        best_lag: int | None = None
        best_ic = 0.0
        best_p = float("nan")
        best_n = 0
        per_h: dict[str, float] = {}
        for h in horizons:
            y = target_at_horizon(sub, target, h)
            if y is None:
                per_h[str(h)] = float("nan")
                continue
            rho, p, n = rank_ic(x, y, min_n=min_n)
            per_h[str(h)] = rho
            if np.isfinite(rho) and abs(rho) > abs(best_ic):
                best_ic = float(rho)
                best_lag = h
                best_p = p
                best_n = n
        if best_lag is None or best_lag > max_lag or abs(best_ic) < min_ic:
            continue
        if allowed_best_lags is not None and best_lag not in allowed_best_lags:
            continue
        if reject_peak_at is not None and best_lag == reject_peak_at:
            continue
        rows.append(
            {
                "feature": feat,
                "best_lag": best_lag,
                "rank_ic": best_ic,
                "ic_sign": ic_sign(best_ic),
                "p_value": best_p,
                "n": best_n,
                "ic_by_horizon": per_h,
            }
        )

    rows.sort(key=lambda r: abs(r["rank_ic"]), reverse=True)
    deps = feature_deps if feature_deps is not None else load_feature_deps()
    col_to_nodes = column_to_nodes(deps)
    node_best: dict[str, dict] = {}
    for row in rows:
        for node in col_to_nodes.get(row["feature"], set()):
            prev = node_best.get(node)
            if prev is None or abs(row["rank_ic"]) > abs(prev["rank_ic"]):
                node_best[node] = {
                    "node": node,
                    "via_column": row["feature"],
                    "best_lag": row["best_lag"],
                    "rank_ic": row["rank_ic"],
                    "ic_sign": row["ic_sign"],
                }
    node_summaries = sorted(
        node_best.values(),
        key=lambda r: abs(r["rank_ic"]),
        reverse=True,
    )
    requested = sorted({n["node"] for n in node_summaries})
    return rows, node_summaries, requested


def invert_columns_for_nodes(
    node_summaries: list[dict],
    requested_nodes: list[str],
) -> list[str]:
    """Output column names to invert (auto mode): negative IC at best lag."""
    keep = set(requested_nodes)
    inv: list[str] = []
    for ns in node_summaries:
        if ns["node"] not in keep:
            continue
        if ns.get("rank_ic", 0) < 0:
            inv.append(str(ns["via_column"]))
    return sorted(set(inv))


def build_monotone_payload(
    requested_nodes: list[str],
    node_summaries: list[dict],
    *,
    feature_deps: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Review-only monotone constraint hints per expanded output column."""
    deps = feature_deps if feature_deps is not None else load_feature_deps()
    sign_by_node = {ns["node"]: ns.get("rank_ic", 0.0) for ns in node_summaries}
    expanded: list[dict[str, Any]] = []
    for node in requested_nodes:
        cols = (deps.get(node) or {}).get("output_columns") or [node]
        rank_ic_val = sign_by_node.get(node)
        if rank_ic_val is None or not np.isfinite(rank_ic_val):
            constraint = 0
        elif rank_ic_val >= 0:
            constraint = 1
        else:
            constraint = -1
        for col in cols:
            expanded.append(
                {
                    "column": str(col),
                    "node": node,
                    "monotone_constraint": constraint,
                }
            )
    return {
        "requested_features": requested_nodes,
        "expanded_columns": expanded,
        "monotone_constraints": [e["monotone_constraint"] for e in expanded],
        "note": (
            "Review-only. Align column order with trainer numeric feature columns "
            "before pasting into model_params.monotone_constraints."
        ),
    }


def build_monotone_payload_columns(selected_rows: list[dict]) -> dict[str, Any]:
    expanded: list[dict[str, Any]] = []
    for row in selected_rows:
        rank_ic_val = row.get("rank_ic", 0.0)
        if not np.isfinite(rank_ic_val):
            constraint = 0
        elif rank_ic_val >= 0:
            constraint = 1
        else:
            constraint = -1
        expanded.append(
            {
                "column": str(row["feature"]),
                "monotone_constraint": constraint,
            }
        )
    return {
        "selected_columns": [r["feature"] for r in selected_rows],
        "expanded_columns": expanded,
        "monotone_constraints": [e["monotone_constraint"] for e in expanded],
        "note": (
            "Review-only. Align column order with trainer numeric feature columns "
            "before pasting into model_params.monotone_constraints."
        ),
    }


def _leading_comment_block(raw: str) -> str:
    lines = raw.splitlines()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or stripped == "":
            out.append(line)
        else:
            break
    block = "\n".join(out).rstrip()
    return block + "\n\n" if block else ""


def _writeback_features_yaml(
    feat_yaml: Path,
    *,
    requested_features: list[str],
    invert_cols: list[str],
    max_lag: int,
    min_ic: float,
    target: str,
    writeback_mode: WritebackMode,
    top_n: int | None,
    allowed_best_lags: frozenset[int] | None = None,
) -> None:
    """Update requested_features / invert_features, preserving other keys.

    - Preserves the leading comment block.
    - Preserves existing feature_pipeline sub-keys (exclude_columns, selector,
      post_processors, ensure_signal_column, ...).
    - Replaces (not appends) the description; no duplicate-key accumulation.
    """
    raw = feat_yaml.read_text(encoding="utf-8")
    doc = yaml.safe_load(raw) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"features yaml is not a mapping: {feat_yaml}")

    fp = doc.get("feature_pipeline")
    if not isinstance(fp, dict):
        fp = {}
    fp["requested_features"] = requested_features
    if invert_cols:
        fp["invert_features"] = expand_invert_features(invert_cols)
    else:
        fp.pop("invert_features", None)
    doc["feature_pipeline"] = fp
    lag_clause = (
        f"best_lag in {{{','.join(str(x) for x in sorted(allowed_best_lags))}}}"
        if allowed_best_lags
        else f"best_lag<={max_lag}"
    )
    if writeback_mode == "columns":
        n_part = f"top-{top_n} columns" if top_n else "IC columns"
        doc["description"] = (
            f"IC-pruned @ holdout {lag_clause}, |IC|>={min_ic}, "
            f"target={target}, {n_part} (column singletons).\n"
        )
    else:
        doc["description"] = (
            f"IC-pruned @ holdout {lag_clause}, |IC|>={min_ic}, target={target}.\n"
        )

    comment = _leading_comment_block(raw)
    body = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
    feat_yaml.write_text(comment + body, encoding="utf-8")


def _writeback_model_features_yaml(
    archetype_yaml: Path,
    *,
    selected_rows: list[dict],
    holdout_start: str,
    holdout_end: str,
    target: str,
    min_ic: float,
    max_lag: int,
    top_n: int | None,
    allowed_best_lags: frozenset[int] | None,
    reject_peak_at: int | None,
    ic_prune_json: Path,
    deps: dict[str, Any],
    project_root: Path | None = None,
) -> None:
    """Write tree archetype column manifest (rule-style feature: entries)."""
    archetype_yaml.parent.mkdir(parents=True, exist_ok=True)
    enriched = column_rows_with_nodes(selected_rows, deps)
    columns_block: list[dict[str, Any]] = []
    for row in enriched:
        entry: dict[str, Any] = {
            "feature": row["feature"],
            "best_lag": row["best_lag"],
            "rank_ic": round(float(row["rank_ic"]), 6),
            "ic_sign": row["ic_sign"],
        }
        if row.get("node"):
            entry["node"] = row["node"]
        columns_block.append(entry)

    root = project_root or Path(__file__).resolve().parents[3]
    try:
        json_ref = str(ic_prune_json.relative_to(root))
    except ValueError:
        json_ref = str(ic_prune_json)

    doc = {
        "description": (
            "Holdout IC-pruned model input columns. "
            "Regenerate via mlbot research ic-prune (writeback_mode=columns)."
        ),
        "provenance": {
            "holdout_start": holdout_start,
            "holdout_end": holdout_end,
            "target": target,
            "min_ic": min_ic,
            "max_lag": max_lag,
            "allowed_best_lags": sorted(allowed_best_lags) if allowed_best_lags else None,
            "reject_peak_at": reject_peak_at,
            "top_n": top_n,
            "ic_prune_json": json_ref,
        },
        "columns": columns_block,
    }
    body = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
    archetype_yaml.write_text(body, encoding="utf-8")


def run_ic_prune(
    *,
    parquet: str | Path,
    output_dir: str | Path,
    holdout_start: str = "2025-10-01",
    holdout_end: str = "2026-04-01",
    horizons: str | list[int] = "1,2,3,4,5",
    max_lag: int = 5,
    allowed_best_lags: str | list[int] | None = None,
    reject_peak_at: int | None = None,
    min_ic: float = 0.02,
    min_n: int = 200,
    target: str = DEFAULT_TARGET,
    write_features_yaml: str | Path | None = None,
    writeback_mode: WritebackMode = "columns",
    top_n_nodes: int | None = None,
    top_n_columns: int | None = 20,
    intersect_features_yaml: str | Path | None = None,
    always_include: list[str] | None = None,
    invert_mode: InvertMode = "none",
    emit_monotone_constraints: str | Path | None = None,
    write_model_features_yaml: str | Path | None = None,
    strategy: str | None = None,
    project_root: Path | None = None,
    feature_deps_path: Path | None = None,
) -> dict[str, Path | None]:
    """Run holdout IC screen; return paths to artifacts."""
    root = project_root or Path(__file__).resolve().parents[3]
    pq = Path(parquet)
    if not pq.is_absolute():
        pq = (root / pq).resolve()
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    deps_path = feature_deps_path
    if deps_path and not deps_path.is_absolute():
        deps_path = (root / deps_path).resolve()
    feature_deps = load_feature_deps(deps_path)

    df = pd.read_parquet(pq)
    horizon_list = parse_horizons(horizons)
    lag_whitelist = parse_allowed_best_lags(allowed_best_lags)
    rows, node_summaries, requested_nodes = screen_features(
        df,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        horizons=horizon_list,
        min_ic=min_ic,
        max_lag=max_lag,
        min_n=min_n,
        target=target,
        allowed_best_lags=lag_whitelist,
        reject_peak_at=reject_peak_at,
        feature_deps=feature_deps,
    )

    pool_nodes: set[str] | None = None
    pool_columns: set[str] | None = None
    if intersect_features_yaml:
        pool_path = Path(intersect_features_yaml)
        if not pool_path.is_absolute():
            pool_path = (root / pool_path).resolve()
        pool_data = yaml.safe_load(pool_path.read_text(encoding="utf-8")) or {}
        pool_requested = (
            pool_data.get("feature_pipeline", {}).get("requested_features") or []
        )
        pool_nodes = set(pool_requested)
        pool_columns = expand_requested_features_to_columns(pool_requested, feature_deps)

    always = list(always_include or ["atr_f"])
    always_nodes, always_columns = split_always_include(always, feature_deps)

    selected_columns: list[str] = []
    selected_rows: list[dict] = []
    requested_features: list[str]
    top_n_applied: int | None

    if writeback_mode == "columns":
        selected_rows = trim_columns(
            rows,
            top_n=top_n_columns,
            pool_columns=pool_columns,
            always_include_columns=always_columns,
        )
        selected_columns = [r["feature"] for r in selected_rows]
        requested_features = build_compute_request(selected_columns, always_nodes)
        top_n_applied = top_n_columns
        keep_nodes = set(always_nodes)
        col_to_nodes = column_to_nodes(feature_deps)
        for col in selected_columns:
            keep_nodes.update(col_to_nodes.get(col, set()))
        node_summaries = [ns for ns in node_summaries if ns["node"] in keep_nodes]
    else:
        requested_nodes = trim_nodes(
            requested_nodes,
            top_n=top_n_nodes,
            pool_nodes=pool_nodes,
            always_include=always,
        )
        requested_features = requested_nodes
        top_n_applied = top_n_nodes
        keep_nodes = set(requested_nodes)
        node_summaries = [ns for ns in node_summaries if ns["node"] in keep_nodes]

    invert_cols: list[str] = []
    if invert_mode == "auto":
        if writeback_mode == "columns":
            invert_cols = invert_columns_for_rows(selected_rows)
        else:
            invert_cols = invert_columns_for_nodes(node_summaries, requested_features)

    payload: dict[str, Any] = {
        "holdout": [holdout_start, holdout_end],
        "target": target,
        "horizons": horizon_list,
        "max_lag": max_lag,
        "allowed_best_lags": sorted(lag_whitelist) if lag_whitelist else None,
        "reject_peak_at": reject_peak_at,
        "min_ic": min_ic,
        "invert_mode": invert_mode,
        "writeback_mode": writeback_mode,
        "n_pass_columns": len(rows),
        "n_selected_columns": len(selected_columns),
        "n_pass_nodes": len({ns["node"] for ns in node_summaries}),
        "columns": rows,
        "nodes": node_summaries,
        "requested_features": requested_features,
    }
    if writeback_mode == "columns":
        payload["selected_columns"] = selected_columns
        payload["top_n_columns"] = top_n_columns
    else:
        payload["top_n_nodes"] = top_n_nodes
    if invert_cols:
        payload["invert_features"] = invert_cols

    json_path = out_dir / "ic_prune_holdout.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lag_pass = (
        f"best_lag ∈ {{{','.join(str(x) for x in sorted(lag_whitelist))}}}"
        if lag_whitelist
        else f"best_lag≤{max_lag}"
    )
    md = [
        f"# holdout IC prune ({lag_pass}, target={target})",
        "",
        f"- parquet: `{pq}`",
        f"- holdout: {holdout_start} → {holdout_end}",
        f"- pass: |IC|≥{min_ic}, {lag_pass}",
    ]
    if reject_peak_at is not None:
        md.append(f"- reject peak @ lag {reject_peak_at}")
    md += [
        f"- writeback_mode: {writeback_mode}",
        f"- IC pass columns: {len(rows)}",
        f"- invert_mode: {invert_mode}",
        "",
    ]
    if writeback_mode == "columns":
        md.append(
            f"- selected columns: {len(selected_columns)} "
            f"(top_n_columns={top_n_columns})"
        )
        md.extend(["", "## selected model columns", ""])
        md.append("| feature | best_lag | rank_ic | sign | n |")
        md.append("|---|---:|---:|:--:|---:|")
        for r in selected_rows:
            md.append(
                f"| {r['feature']} | {r['best_lag']} | {r['rank_ic']:.4f} | "
                f"{r['ic_sign']} | {r['n']} |"
            )
    else:
        md.append(f"- selected nodes: {len(requested_features)}")
    md.extend(
        [
            "",
            "## all IC-pass columns (ranked)",
            "",
            "| feature | best_lag | rank_ic | sign | n |",
            "|---|---:|---:|:--:|---:|",
        ]
    )
    for r in rows[:60]:
        md.append(
            f"| {r['feature']} | {r['best_lag']} | {r['rank_ic']:.4f} | "
            f"{r['ic_sign']} | {r['n']} |"
        )
    if len(rows) > 60:
        md.append(f"| ... | | | | ({len(rows) - 60} more) |")
    md.extend(["", "## nodes (_f)", ""])
    md.append("| node | via_column | best_lag | rank_ic | sign |")
    md.append("|---|---|---:|---:|:--:|")
    for ns in node_summaries:
        md.append(
            f"| {ns['node']} | {ns['via_column']} | {ns['best_lag']} | "
            f"{ns['rank_ic']:.4f} | {ns['ic_sign']} |"
        )
    if invert_cols:
        md.extend(["", "## invert_features (output columns)", ""])
        for col in invert_cols:
            md.append(f"- {col}")
    md_path = out_dir / "ic_prune_holdout.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    mono_path: Path | None = None
    if emit_monotone_constraints not in (None, False, "false", "False"):
        mono_path = Path(str(emit_monotone_constraints))
        if not mono_path.is_absolute():
            mono_path = (root / mono_path).resolve()
        mono_path.parent.mkdir(parents=True, exist_ok=True)
        mono_payload = (
            build_monotone_payload_columns(selected_rows)
            if writeback_mode == "columns"
            else build_monotone_payload(
                requested_features, node_summaries, feature_deps=feature_deps
            )
        )
        mono_path.write_text(
            yaml.safe_dump(mono_payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        print(f"wrote {mono_path}")

    feat_out: Path | None = None
    if write_features_yaml not in (None, False, "false", "False"):
        feat_yaml = Path(str(write_features_yaml))
        if not feat_yaml.is_absolute():
            feat_yaml = root / feat_yaml
        _writeback_features_yaml(
            feat_yaml,
            requested_features=requested_features,
            invert_cols=invert_cols,
            max_lag=max_lag,
            min_ic=min_ic,
            target=target,
            writeback_mode=writeback_mode,
            top_n=top_n_applied,
            allowed_best_lags=lag_whitelist,
        )
        feat_out = feat_yaml
        if writeback_mode == "columns":
            print(
                f"updated {feat_yaml} ({len(selected_columns)} column singletons + "
                f"{len(always_nodes)} compute nodes)"
            )
        else:
            print(f"updated {feat_yaml} ({len(requested_features)} nodes)")

    archetype_out: Path | None = None
    skip_archetype = write_model_features_yaml in (False, "false", "False")
    if writeback_mode == "columns" and selected_rows and not skip_archetype:
        archetype_yaml: Path | None = None
        if write_model_features_yaml not in (None, True, "true"):
            archetype_yaml = Path(str(write_model_features_yaml))
        elif strategy:
            archetype_yaml = default_model_features_yaml(strategy, root)
        if archetype_yaml is not None:
            if not archetype_yaml.is_absolute():
                archetype_yaml = root / archetype_yaml
            _writeback_model_features_yaml(
                archetype_yaml,
                selected_rows=selected_rows,
                holdout_start=holdout_start,
                holdout_end=holdout_end,
                target=target,
                min_ic=min_ic,
                max_lag=max_lag,
                top_n=top_n_columns,
                allowed_best_lags=lag_whitelist,
                reject_peak_at=reject_peak_at,
                ic_prune_json=json_path,
                deps=feature_deps,
                project_root=root,
            )
            archetype_out = archetype_yaml
            print(f"wrote {archetype_yaml} ({len(selected_rows)} columns)")

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return {
        "json": json_path,
        "md": md_path,
        "features_yaml": feat_out,
        "model_features_yaml": archetype_out,
        "monotone_constraints": mono_path,
    }
