"""
Feature group search (greedy forward selection) for strategy configs.

This is designed to be reproducible and compatible with existing training/backtest pipeline:
we generate temporary strategy config dirs with different `features.yaml` contents, then call
`scripts/train_strategy_pipeline.py` and aggregate `strategy_pipeline_metrics.csv`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import yaml


@dataclass(frozen=True)
class SearchConfig:
    base_strategy_dir: Path
    timeframe: str
    symbol: str
    start_date: str
    end_date: str
    test_size: float
    seeds: List[int]
    output_dir: Path
    deterministic: bool
    no_docker: bool
    fast_features: bool = False
    invert_eval: str = "conservative"  # none|conservative|all


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, obj: dict) -> None:
    path.write_text(
        yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _stable_dedup(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in items or []:
        if not isinstance(x, str):
            continue
        s = x.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _load_base_invert_features(strategy_dir: Path) -> List[str]:
    feats = _load_yaml(strategy_dir / "features.yaml")
    fp = feats.get("feature_pipeline") or {}
    inv = fp.get("invert_features") or []
    if isinstance(inv, list):
        return _stable_dedup([str(x) for x in inv])
    return []


def _backup_if_exists(path: Path) -> Optional[Path]:
    """Create a timestamped backup if file exists; returns backup path if created."""
    if not path.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak.{ts}")
    shutil.copy2(path, bak)
    return bak


def _writeback_features_yaml(
    *,
    base_strategy_dir: Path,
    out_path: Path,
    requested_features: List[str],
    meta: dict,
    invert_candidates: Optional[List[str]] = None,
) -> dict:
    """
    Write a features.yaml compatible file with requested_features set to the provided list.
    Keeps existing keys from base strategy's features.yaml where possible (e.g., ensure_signal_column).
    Adds `feature_group_search` metadata for provenance.
    """
    base_yaml_path = base_strategy_dir / "features.yaml"
    base_obj = _load_yaml(base_yaml_path) if base_yaml_path.exists() else {}

    # Normalize structure
    base_obj.setdefault("feature_pipeline", {})
    fp = base_obj["feature_pipeline"] or {}
    fp["requested_features"] = list(requested_features)

    # Write back only the final invert_features, not an unbounded candidate list.
    # NOTE:
    # - feature_pipeline.requested_features are *feature compute functions* (e.g. "..._f")
    # - feature_pipeline.invert_features are *output column names* to multiply by -1 before training/inference.
    #
    # Therefore we must NOT prune invert_features by requested_features (different namespaces).
    # It's safe to keep extra entries: the trainer only applies inversion to columns that
    # are actually present in the selected feature columns.
    inv_cand = invert_candidates
    if inv_cand is None:
        inv_cand = fp.get("invert_features") or []
    inv_cand = [str(x).strip() for x in (inv_cand or []) if str(x).strip()]
    # Deduplicate but preserve order (stable writeback)
    seen = set()
    inv_final = []
    for name in inv_cand:
        if name in seen:
            continue
        seen.add(name)
        inv_final.append(name)
    fp["invert_features"] = inv_final

    base_obj["feature_pipeline"] = fp

    # Ensure name is present and indicate it's suggested
    base_name = base_obj.get("name") or base_strategy_dir.name
    base_obj["name"] = f"{base_name}__suggested"

    # Attach provenance metadata
    base_obj["feature_group_search"] = meta

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bak = _backup_if_exists(out_path)
    _write_yaml(out_path, base_obj)
    return {"written": str(out_path), "backup": str(bak) if bak else None}


def _load_invert_candidates(path: Path) -> List[str]:
    """
    Load invert candidates from a YAML file.
    Supports:
      - YAML list: [feat1, feat2, ...]
      - full features config: {feature_pipeline: {invert_features: [...]}}
      - {invert_candidates: [...]} (preferred naming for pool artifacts)
    """
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if obj is None:
        return []
    if isinstance(obj, list):
        return [str(x).strip() for x in obj if str(x).strip()]
    if isinstance(obj, dict):
        if "invert_candidates" in obj and isinstance(obj["invert_candidates"], list):
            return [str(x).strip() for x in obj["invert_candidates"] if str(x).strip()]
        fp = obj.get("feature_pipeline")
        if isinstance(fp, dict) and isinstance(fp.get("invert_features"), list):
            return [str(x).strip() for x in fp["invert_features"] if str(x).strip()]
    raise ValueError(
        "invert-candidates-yaml must be a YAML list or contain feature_pipeline.invert_features"
    )


def _strategy_name(strategy_dir: Path) -> str:
    feats = _load_yaml(strategy_dir / "features.yaml")
    name = feats.get("name") or strategy_dir.name
    return str(name)


def _safe_dir_component(s: str, *, max_len: int = 180) -> str:
    """
    Make a filesystem-safe directory component.

    Linux filesystems typically limit a single path component to 255 bytes.
    Our group/run suffixes can get extremely long, so we truncate and add a hash.
    """
    s = str(s or "").strip()
    if not s:
        return "x"
    safe = []
    for ch in s:
        if ch.isalnum() or ch in {"_", "-", ".", "="}:
            safe.append(ch)
        else:
            safe.append("_")
    out = "".join(safe)
    if len(out) <= max_len:
        return out
    h = hashlib.md5(out.encode("utf-8")).hexdigest()[:12]
    prefix = out[: max(16, max_len - (2 + len(h)))]
    return f"{prefix}__{h}"


def _make_temp_strategy(
    *,
    base_dir: Path,
    tmp_root: Path,
    name_suffix: str,
    requested_features: List[str],
    invert_features: Optional[List[str]] = None,
) -> Path:
    """
    Copy strategy dir to tmp_root/<base_name>__<suffix> and overwrite `features.yaml`.
    """
    base_name = base_dir.name
    safe_suffix = _safe_dir_component(name_suffix, max_len=180)
    out_dir = tmp_root / f"{base_name}__{safe_suffix}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    from src.config.strategy_layout import copy_strategy_package

    copy_strategy_package(base_dir, out_dir)

    feats = _load_yaml(out_dir / "features.yaml")
    feats["name"] = f"{feats.get('name', base_name)}__{safe_suffix}"
    feats.setdefault("feature_pipeline", {})
    feats["feature_pipeline"]["requested_features"] = list(requested_features)
    if invert_features is not None:
        # invert_features can be output columns OR feature nodes (*_f). The trainer will expand.
        feats["feature_pipeline"]["invert_features"] = list(invert_features)
    _write_yaml(out_dir / "features.yaml", feats)
    return out_dir


def _run_one_seed(
    *,
    strategy_dir: Path,
    cfg: SearchConfig,
    seed: int,
    out_root: Path,
) -> Path:
    """
    Run training pipeline; returns path to results.json.
    """
    _ensure_dir(out_root)
    # Resume behavior: if this seed output already has exactly one results.json, reuse it.
    # This is important for long searches that can be interrupted (or crash on a single candidate).
    strat_name = _strategy_name(strategy_dir)
    direct = out_root / strat_name / "results.json"
    if direct.is_file():
        return direct
    try:
        existing = list(out_root.glob(f"{strat_name}/**/results.json"))
        if len(existing) == 1:
            return existing[0]
    except Exception:
        pass
    args = [
        "--config",
        str(strategy_dir),
        "--symbol",
        cfg.symbol,
        "--data-path",
        "data/parquet_data",
        "--timeframe",
        cfg.timeframe,
        "--test-size",
        str(cfg.test_size),
        "--output-root",
        str(out_root),
        "--seed",
        str(seed),
    ]
    if cfg.deterministic:
        args.append("--deterministic")

    # Performance: enable FeatureStore (wide table) so multi-seed and multi-step searches
    # don't re-compute features from scratch for every run.
    #
    # IMPORTANT: Use a stable layer derived from the BASE strategy dir (not the tmp variant dir),
    # so different candidate combinations share the same FeatureStore dataset and can incrementally
    # fill missing columns (without clobbering existing ones).
    try:
        from src.feature_store.layer_naming import default_layer_from_config

        fs_layer = default_layer_from_config(cfg.base_strategy_dir)
        args.extend(
            [
                "--feature-store-dir",
                "feature_store",
                "--feature-store-layer",
                str(fs_layer),
            ]
        )
    except Exception:
        # Never fail the search due to optional perf optimization.
        pass

    cmd = ["python3", "scripts/train_strategy_pipeline.py"] + args
    env = os.environ.copy()
    # Train pipeline supports optional cropping via env vars.
    env["TRAIN_START_DATE"] = cfg.start_date
    env["TRAIN_END_DATE"] = cfg.end_date
    if bool(getattr(cfg, "fast_features", False)):
        # Feature search speed-up: allow feature functions to switch to cheaper approximations
        # (e.g., DTW without random templates; spectrum computed at lower frequency).
        env["FEATURE_FAST_MODE"] = "1"
    try:
        subprocess.run(cmd, check=True, cwd=str(Path.cwd()), env=env)
    except subprocess.CalledProcessError as e:
        # Safety net: training can fail non-deterministically (e.g. killed process / transient issues).
        # The search must keep going, so we create a placeholder results.json and continue.
        placeholder = out_root / _strategy_name(strategy_dir) / "results.json"
        _ensure_dir(placeholder.parent)
        payload = {
            "strategy": _strategy_name(strategy_dir),
            "model_type": "unknown",
            "task_type": "unknown",
            "avg_cv_metric": None,
            "n_features": None,
            "n_train_samples": 0,
            "n_test_samples": 0,
            "evaluation": {},
            "diagnostics": {
                "skip": {
                    "skipped": True,
                    "reason": "train_pipeline_nonzero_exit",
                    "returncode": int(getattr(e, "returncode", -1) or -1),
                    "cmd": [str(x) for x in cmd],
                }
            },
            "backtest": {
                "total_return_pct": 0.0,
                "sharpe": -999.0,
                "max_drawdown_pct": 0.0,
                "total_trades": 0,
                "reason": "train_pipeline_nonzero_exit",
            },
        }
        with open(placeholder, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(
            f"   ⚠️  train_strategy_pipeline failed (returncode={payload['diagnostics']['skip']['returncode']}), "
            f"created placeholder results at {placeholder}"
        )
        return placeholder

    results = list(out_root.glob(f"{strat_name}/**/results.json"))
    if len(results) != 1:
        # Safety net:
        # Even if the training pipeline decides to skip a run (e.g. insufficient samples),
        # feature-group-search must keep going. Create a placeholder results.json so downstream
        # aggregation/selection can proceed deterministically.
        if len(results) == 0:
            placeholder = out_root / _strategy_name(strategy_dir) / "results.json"
            _ensure_dir(placeholder.parent)
            payload = {
                "strategy": _strategy_name(strategy_dir),
                "model_type": "unknown",
                "task_type": "unknown",
                "avg_cv_metric": None,
                "n_features": None,
                "n_train_samples": 0,
                "n_test_samples": 0,
                "evaluation": {},
                "diagnostics": {
                    "skip": {
                        "skipped": True,
                        "reason": "missing_results_json_from_train_pipeline",
                    }
                },
                "backtest": {
                    "total_return_pct": 0.0,
                    "sharpe": -999.0,
                    "max_drawdown_pct": 0.0,
                    "win_rate": 0.0,
                    "total_trades": 0,
                    "skipped": True,
                    "reason": "missing_results_json_from_train_pipeline",
                },
            }
            placeholder.write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
            return placeholder
        raise RuntimeError(
            f"Expected 1 results.json under {out_root}, found {len(results)}"
        )
    return results[0]


def run_seed_sweep_for_strategy(
    *,
    strategy_dir: Path,
    cfg: SearchConfig,
    run_id: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (all_rows, summary_by_strategy).
    """
    rows = []
    for seed in cfg.seeds:
        seed_out = cfg.output_dir / "runs" / run_id / f"seed_{seed}"
        results_json = _run_one_seed(
            strategy_dir=strategy_dir, cfg=cfg, seed=seed, out_root=seed_out
        )
        payload = json.loads(results_json.read_text(encoding="utf-8"))
        row = {
            "strategy": payload.get("strategy"),
            "task": payload.get("task_type"),
            "train": payload.get("n_train_samples"),
            "CV": payload.get("avg_cv_metric"),
            "corr": (payload.get("evaluation") or {}).get("test_correlation"),
            "return%": (payload.get("backtest") or {}).get("total_return_pct"),
            "Sharpe": (payload.get("backtest") or {}).get("sharpe"),
            "DD%": (payload.get("backtest") or {}).get("max_drawdown_pct"),
            "trades": (payload.get("backtest") or {}).get("total_trades"),
            "seed": int(seed),
            "run_id": run_id,
        }
        rows.append(pd.DataFrame([row]))

    all_df = pd.concat(rows, ignore_index=True)
    metric_cols = [
        c
        for c in ["train", "CV", "corr", "return%", "Sharpe", "DD%", "trades"]
        if c in all_df.columns
    ]
    gb = all_df.groupby(["strategy", "task"], dropna=False)
    summary = gb[metric_cols].agg(["mean", "std", "min", "max"]).reset_index()
    summary.columns = [
        "_".join([c for c in col if c]).rstrip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return all_df, summary


def greedy_forward_search(
    *,
    cfg: SearchConfig,
    base_features: List[str],
    groups: Dict[str, List[str]],
    max_steps: int,
    objective: str,
    min_trades: int,
) -> dict:
    """
    Greedy forward selection over feature groups.

    objective: one of ["Sharpe_mean", "return%_mean", "corr_mean", "CV_mean"] (must exist in summary).
    """
    tmp_root = cfg.output_dir / "tmp_strategies"
    _ensure_dir(tmp_root)

    selected: List[str] = []
    remaining = list(groups.keys())
    history: List[dict] = []
    # For transparency/debuggability: store all evaluated candidates per step (not just the winner).
    candidates_history: List[dict] = []

    current_features = list(base_features)
    best_score = None
    baseline_score = None
    eps_improve = 1e-9
    stop_reason = "unknown"
    rejected_groups: List[str] = []
    baseline_summary = None

    # Always evaluate baseline (no added groups) so the search can legitimately return "select nothing".
    tmp_root = cfg.output_dir / "tmp_strategies"
    _ensure_dir(tmp_root)
    baseline_suffix = "baseline"
    baseline_dir = _make_temp_strategy(
        base_dir=cfg.base_strategy_dir,
        tmp_root=tmp_root,
        name_suffix=baseline_suffix,
        requested_features=current_features,
    )
    _, baseline_summ = run_seed_sweep_for_strategy(
        strategy_dir=baseline_dir,
        cfg=cfg,
        run_id=baseline_suffix,
    )
    if baseline_summ.empty:
        raise RuntimeError("Baseline seed sweep returned empty summary")
    baseline_row = baseline_summ.iloc[0].to_dict()
    if objective not in baseline_row:
        raise ValueError(
            f"Objective '{objective}' not found in baseline summary columns: {list(baseline_row.keys())}"
        )
    baseline_score = float(baseline_row[objective])
    best_score = float(baseline_score)
    baseline_summary = baseline_row

    for step in range(int(max_steps)):
        best_candidate = None
        best_candidate_score = None
        best_candidate_summary = None
        step_candidates: List[dict] = []

        for g in remaining:
            feats = current_features + groups[g]
            suffix = f"step{step+1}_add_{g}"
            strat_dir = _make_temp_strategy(
                base_dir=cfg.base_strategy_dir,
                tmp_root=tmp_root,
                name_suffix=suffix,
                requested_features=feats,
            )

            run_id = suffix
            _, summary = run_seed_sweep_for_strategy(
                strategy_dir=strat_dir, cfg=cfg, run_id=run_id
            )
            if summary.empty:
                step_candidates.append(
                    {
                        "step": step + 1,
                        "candidate_group": g,
                        "score": None,
                        "valid": False,
                        "reject_reason": "empty_summary",
                        "summary": {},
                    }
                )
                continue
            row = summary.iloc[0].to_dict()

            # trade sanity
            trades_col = "trades_mean"
            trades_ok = True
            if trades_col in row:
                trades_ok = float(row[trades_col]) >= float(min_trades)
            if not trades_ok:
                step_candidates.append(
                    {
                        "step": step + 1,
                        "candidate_group": g,
                        "score": float(row[objective]) if objective in row else None,
                        "valid": False,
                        "reject_reason": "min_trades",
                        "summary": row,
                    }
                )
                continue

            if objective not in row:
                raise ValueError(
                    f"Objective '{objective}' not found in summary columns: {list(row.keys())}"
                )

            score = float(row[objective])
            step_candidates.append(
                {
                    "step": step + 1,
                    "candidate_group": g,
                    "score": score,
                    "valid": True,
                    "reject_reason": None,
                    "summary": row,
                }
            )
            if best_candidate_score is None or score > best_candidate_score:
                best_candidate_score = score
                best_candidate = g
                best_candidate_summary = row

        if best_candidate is None:
            stop_reason = "no_valid_candidates"
            break

        # Record all candidate scores for this step (useful when search space is big / noisy).
        candidates_history.append(
            {
                "step": step + 1,
                "current_selected": list(selected),
                "candidates": step_candidates,
            }
        )

        # Default-safe behavior: stop if we can't improve the objective.
        # Note: this may miss "synergy" that requires temporary deterioration; if needed, upgrade to beam/SFFS.
        if best_score is not None and best_candidate_score is not None:
            if float(best_candidate_score) <= float(best_score) + eps_improve:
                stop_reason = "no_improvement"
                # At this point, none of the remaining groups can improve the objective.
                rejected_groups = list(remaining)
                break

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        current_features = current_features + groups[best_candidate]
        best_score = best_candidate_score
        history.append(
            {
                "step": step + 1,
                "added_group": best_candidate,
                "objective": objective,
                "score": best_score,
                "summary": best_candidate_summary,
            }
        )

    if stop_reason == "unknown":
        stop_reason = (
            "max_steps_reached" if len(selected) >= int(max_steps) else "completed"
        )
        if stop_reason == "max_steps_reached":
            rejected_groups = list(remaining)

    return {
        "base_strategy": cfg.base_strategy_dir.name,
        "base_features": base_features,
        "baseline": {"score": baseline_score, "summary": baseline_summary},
        "search_algo": "greedy",
        "selected_groups": selected,
        "final_features": current_features,
        "history": history,
        "candidates_history": candidates_history,
        "stop_reason": stop_reason,
        "rejected_groups": rejected_groups,
        "objective": objective,
        "min_trades": min_trades,
        "seeds": cfg.seeds,
    }


def _parse_int_list(csv: str) -> List[int]:
    out: List[int] = []
    for part in str(csv or "").split(","):
        part = str(part).strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            continue
    return out


def _score_from_summary(
    *,
    summary: pd.DataFrame,
    objective: str,
    min_trades: int,
) -> tuple[float | None, bool, str | None, dict]:
    """
    Convert the aggregated summary DataFrame into a (score, valid, reject_reason, summary_dict).
    Mirrors greedy_forward_search behavior.
    """
    if summary.empty:
        return None, False, "empty_summary", {}
    row = summary.iloc[0].to_dict()
    if objective not in row:
        return None, False, "objective_missing", row
    # Min-trades constraint (mean across seeds)
    try:
        trades = float(row.get("trades_mean", 0.0))
        if trades < float(min_trades):
            return -999.0, False, "min_trades", row
    except Exception:
        pass
    try:
        return float(row[objective]), True, None, row
    except Exception:
        return None, False, "objective_not_float", row


def successive_halving_prefilter(
    *,
    cfg: SearchConfig,
    base_features: List[str],
    groups: Dict[str, List[str]],
    objective: str,
    min_trades: int,
    stages: List[int],
    top_fraction: float,
    min_survivors: int,
    target_survivors: int,
    invert_candidates: Optional[List[str]] = None,
) -> dict:
    """
    Successive Halving prefilter (no selection, just pruning).

    We evaluate each group as a *single add* on top of base_features.
    Budget dimension: number of seeds (prefix of cfg.seeds).
    Returns survivors and per-group scores from the final stage (if evaluated).
    """
    tmp_root = cfg.output_dir / "tmp_strategies"
    _ensure_dir(tmp_root)

    max_n = max(1, len(cfg.seeds))
    stages = [max(1, min(int(x), max_n)) for x in (stages or []) if int(x) > 0]
    if not stages:
        stages = [1, max_n]
    stages = sorted(set(stages))
    if stages[-1] != max_n:
        stages.append(max_n)

    survivors = list(groups.keys())
    stage_tables: List[dict] = []
    base_inv = _load_base_invert_features(cfg.base_strategy_dir)
    inv_cand_set = set(
        [str(x).strip() for x in (invert_candidates or []) if str(x).strip()]
    )
    invert_by_group: Dict[str, List[str]] = {}

    # Inversion pick policy (conservative):
    # - Only pick inverted version when raw is VALID and "clearly negative"
    # - And inverted improves by a meaningful margin
    invert_min_negative_score = -0.05
    invert_min_improvement = 0.05

    for si, nseeds in enumerate(stages):
        seeds_subset = cfg.seeds[:nseeds]
        cfg_stage = SearchConfig(
            base_strategy_dir=cfg.base_strategy_dir,
            timeframe=cfg.timeframe,
            symbol=cfg.symbol,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            test_size=cfg.test_size,
            seeds=list(seeds_subset),
            output_dir=cfg.output_dir,
            deterministic=cfg.deterministic,
            no_docker=cfg.no_docker,
            fast_features=cfg.fast_features,
            invert_eval=cfg.invert_eval,
        )

        scored: List[tuple[str, float]] = []
        rows: List[dict] = []
        for g in survivors:
            feats = list(base_features) + (groups.get(g) or [])
            # Evaluate raw candidate
            run_id = f"prefilter_add_{g}__halving_s{nseeds}"
            strat_dir = _make_temp_strategy(
                base_dir=cfg.base_strategy_dir,
                tmp_root=tmp_root,
                name_suffix=run_id,
                requested_features=feats,
                invert_features=base_inv,
            )
            _, summ = run_seed_sweep_for_strategy(
                strategy_dir=strat_dir, cfg=cfg_stage, run_id=run_id
            )
            score, valid, reject_reason, row = _score_from_summary(
                summary=summ, objective=objective, min_trades=min_trades
            )

            # Optional verification: try inverted version for Pool-B invert candidates (output columns).
            # NOTE:
            # - `invert_candidates` are OUTPUT COLUMN NAMES (from Pool-B YAML invert_features),
            #   not feature nodes (*_f) and not group names.
            # - Many groups (semantic singletons) may directly add output columns; for those we can
            #   test inversion precisely at the column level.
            # - For feature-node groups, we generally cannot infer which output columns correspond
            #   without consulting the feature DAG; therefore inversion trials are primarily effective
            #   for singleton output-column groups.
            picked_inverted = False
            inv_score = None
            inv_valid = False
            inv_reject_reason = None
            inv_row: dict = {}
            inv_cols_for_group = _stable_dedup(
                [
                    str(x)
                    for x in ([str(g)] + [str(xx) for xx in (groups.get(g) or [])])
                    if str(x).strip() in inv_cand_set
                ]
            )
            try_invert = str(
                getattr(cfg, "invert_eval", "conservative")
            ).strip().lower() in {"conservative", "all"} and bool(inv_cols_for_group)
            if try_invert:
                inv_list = _stable_dedup(list(base_inv) + list(inv_cols_for_group))
                run_id_inv = f"{run_id}__inv"
                strat_dir_inv = _make_temp_strategy(
                    base_dir=cfg.base_strategy_dir,
                    tmp_root=tmp_root,
                    name_suffix=run_id_inv,
                    requested_features=feats,
                    invert_features=inv_list,
                )
                _, summ_inv = run_seed_sweep_for_strategy(
                    strategy_dir=strat_dir_inv, cfg=cfg_stage, run_id=run_id_inv
                )
                inv_score, inv_valid, inv_reject_reason, inv_row = _score_from_summary(
                    summary=summ_inv, objective=objective, min_trades=min_trades
                )
                mode = str(getattr(cfg, "invert_eval", "conservative")).strip().lower()
                if mode == "all":
                    # Always pick the better of raw vs inverted (when both valid).
                    if (
                        bool(valid)
                        and inv_valid
                        and (score is not None)
                        and (inv_score is not None)
                        and float(inv_score) > float(score) + 1e-9
                    ):
                        picked_inverted = True
                else:
                    # Conservative: choose inverted ONLY when raw is valid and clearly negative.
                    if (
                        bool(valid)
                        and inv_valid
                        and (score is not None)
                        and (inv_score is not None)
                        and (float(score) <= float(invert_min_negative_score))
                        and (
                            float(inv_score)
                            >= float(score) + float(invert_min_improvement)
                        )
                    ):
                        picked_inverted = True

                if picked_inverted:
                    score, valid, reject_reason, row = (
                        inv_score,
                        inv_valid,
                        inv_reject_reason,
                        inv_row,
                    )
                    # Record only the OUTPUT COLUMNS we inverted for this group.
                    invert_by_group[str(g)] = list(inv_cols_for_group)
            rows.append(
                {
                    "stage": si + 1,
                    "stage_seeds": list(seeds_subset),
                    "candidate_group": g,
                    "score": score,
                    "valid": bool(valid),
                    "reject_reason": reject_reason,
                    "summary": row,
                    "picked_inverted": bool(picked_inverted),
                    "raw_score": (
                        None
                        if score is None
                        else float(score) if not picked_inverted else None
                    ),
                    "inv_score": None if inv_score is None else float(inv_score),
                    "inv_valid": bool(inv_valid),
                    "inv_reject_reason": inv_reject_reason,
                }
            )
            if valid and score is not None:
                scored.append((g, float(score)))

        stage_tables.append(
            {"stage": si + 1, "seeds": list(seeds_subset), "candidates": rows}
        )

        if not scored:
            survivors = []
            break

        scored.sort(key=lambda x: x[1], reverse=True)
        keep_n = max(int(min_survivors), int(len(scored) * float(top_fraction)))
        keep_n = max(1, min(keep_n, len(scored)))
        survivors = [g for g, _ in scored[:keep_n]]

        # Cap to target_survivors once we reach the full budget
        if nseeds == max_n:
            survivors = survivors[: max(1, int(target_survivors))]

    # Extract final-stage scores
    final_scores: Dict[str, float] = {}
    if stage_tables:
        last = stage_tables[-1]
        for r in last.get("candidates", []):
            if r.get("valid") and r.get("score") is not None:
                final_scores[str(r["candidate_group"])] = float(r["score"])

    return {
        "stages": stages,
        "top_fraction": float(top_fraction),
        "min_survivors": int(min_survivors),
        "target_survivors": int(target_survivors),
        "survivors": survivors,
        "final_scores": final_scores,
        "stage_tables": stage_tables,
        "invert_by_group": invert_by_group,
        "invert_policy": {
            "invert_min_negative_score": invert_min_negative_score,
            "invert_min_improvement": invert_min_improvement,
        },
    }


def sffs_prune_selected(
    *,
    cfg: SearchConfig,
    base_features: List[str],
    groups: Dict[str, List[str]],
    selected_groups: List[str],
    objective: str,
    min_trades: int,
    max_backward_steps: int,
    base_invert_features: Optional[List[str]] = None,
    invert_by_group: Optional[Dict[str, List[str]]] = None,
) -> dict:
    """
    Prune-only SFFS stage: starting from a fixed selected set, repeatedly remove any single
    group if removal improves the objective.
    """
    tmp_root = cfg.output_dir / "tmp_strategies"
    _ensure_dir(tmp_root)

    eps_improve = 1e-9
    max_backward_steps = max(1, int(max_backward_steps))
    base_inv = _stable_dedup(list(base_invert_features or []))
    inv_map = invert_by_group or {}

    def _feats_for(sel: List[str]) -> List[str]:
        feats = list(base_features)
        for g in sel:
            feats = feats + (groups.get(g) or [])
        return feats

    sel = [g for g in (selected_groups or []) if g in groups]
    history: List[dict] = []

    # Score initial
    init_feats = _feats_for(sel)
    run_id0 = f"prune_init__{'__'.join(sel) if sel else 'none'}"
    strat_dir0 = _make_temp_strategy(
        base_dir=cfg.base_strategy_dir,
        tmp_root=tmp_root,
        name_suffix=run_id0,
        requested_features=init_feats,
        invert_features=_stable_dedup(
            list(base_inv) + [str(x) for g in sel for x in (inv_map.get(str(g)) or [])]
        ),
    )
    _, summ0 = run_seed_sweep_for_strategy(
        strategy_dir=strat_dir0, cfg=cfg, run_id=run_id0
    )
    best_score, _, _, best_row = _score_from_summary(
        summary=summ0, objective=objective, min_trades=min_trades
    )
    if best_score is None:
        best_score = -1e18

    steps = 0
    while steps < max_backward_steps and len(sel) > 1:
        steps += 1
        best_rm = None
        best_rm_score = None
        best_rm_summary = None
        for rm in list(sel):
            kept = [g for g in sel if g != rm]
            feats = _feats_for(kept)
            run_id = f"prune_try__keep_{'__'.join(kept) if kept else 'none'}__rm_{rm}"
            strat_dir = _make_temp_strategy(
                base_dir=cfg.base_strategy_dir,
                tmp_root=tmp_root,
                name_suffix=run_id,
                requested_features=feats,
                invert_features=_stable_dedup(
                    list(base_inv)
                    + [str(x) for g in kept for x in (inv_map.get(str(g)) or [])]
                ),
            )
            _, summ = run_seed_sweep_for_strategy(
                strategy_dir=strat_dir, cfg=cfg, run_id=run_id
            )
            score, valid, reject_reason, row = _score_from_summary(
                summary=summ, objective=objective, min_trades=min_trades
            )
            if not valid or score is None:
                continue
            if best_rm_score is None or float(score) > float(best_rm_score):
                best_rm = rm
                best_rm_score = float(score)
                best_rm_summary = row
        if best_rm is None or best_rm_score is None:
            break
        if float(best_rm_score) <= float(best_score) + eps_improve:
            break
        sel = [g for g in sel if g != best_rm]
        best_score = float(best_rm_score)
        history.append(
            {
                "phase": "prune",
                "removed_group": best_rm,
                "objective": objective,
                "score": best_rm_score,
                "summary": best_rm_summary or {},
            }
        )

    return {"selected_groups": sel, "score": float(best_score), "history": history}


def pipeline_sh_beam_sffs(
    *,
    cfg: SearchConfig,
    base_features: List[str],
    groups: Dict[str, List[str]],
    max_steps: int,
    objective: str,
    min_trades: int,
    stages: List[int],
    top_fraction: float,
    min_survivors: int,
    target_survivors: int,
    beam_width: int,
    sffs_max_backward_steps: int,
    invert_candidates: Optional[List[str]] = None,
) -> dict:
    """
    Pipeline:
      1) Successive Halving prefilter (single-add ranking) -> survivors
      2) Beam search on survivors -> best path
      3) SFFS prune-only on best path -> final
    """
    # Baseline (full seeds)
    tmp_root = cfg.output_dir / "tmp_strategies"
    _ensure_dir(tmp_root)
    base_inv = _load_base_invert_features(cfg.base_strategy_dir)
    baseline_suffix = "baseline"
    baseline_dir = _make_temp_strategy(
        base_dir=cfg.base_strategy_dir,
        tmp_root=tmp_root,
        name_suffix=baseline_suffix,
        requested_features=list(base_features),
        invert_features=base_inv,
    )
    _, baseline_summ = run_seed_sweep_for_strategy(
        strategy_dir=baseline_dir, cfg=cfg, run_id=baseline_suffix
    )
    base_score, _, _, base_row = _score_from_summary(
        summary=baseline_summ, objective=objective, min_trades=min_trades
    )
    if base_score is None:
        raise RuntimeError("Baseline evaluation failed")

    pre = successive_halving_prefilter(
        cfg=cfg,
        base_features=base_features,
        groups=groups,
        objective=objective,
        min_trades=min_trades,
        stages=stages,
        top_fraction=top_fraction,
        min_survivors=min_survivors,
        target_survivors=target_survivors,
        invert_candidates=invert_candidates,
    )
    surv_groups = {k: groups[k] for k in pre["survivors"] if k in groups}
    inv_map = pre.get("invert_by_group") or {}

    beam_res = beam_search(
        cfg=cfg,
        base_features=base_features,
        groups=surv_groups,
        max_steps=max_steps,
        objective=objective,
        min_trades=min_trades,
        beam_width=beam_width,
        base_invert_features=base_inv,
        invert_by_group=inv_map,
    )

    pruned = sffs_prune_selected(
        cfg=cfg,
        base_features=base_features,
        groups=surv_groups,
        selected_groups=beam_res.get("selected_groups") or [],
        objective=objective,
        min_trades=min_trades,
        max_backward_steps=sffs_max_backward_steps,
        base_invert_features=base_inv,
        invert_by_group=inv_map,
    )

    final_selected = pruned["selected_groups"]
    final_features = list(base_features)
    for g in final_selected:
        final_features = final_features + (surv_groups.get(g) or [])
    final_invert_features = _stable_dedup(
        list(base_inv)
        + [str(x) for g in final_selected for x in (inv_map.get(str(g)) or [])]
    )

    return {
        "base_strategy": cfg.base_strategy_dir.name,
        "base_features": base_features,
        "baseline": {"score": float(base_score), "summary": base_row},
        "search_algo": "pipeline_sh_beam_sffs",
        "algo_params": {
            "halving_stages": pre["stages"],
            "halving_top_fraction": pre["top_fraction"],
            "halving_min_survivors": pre["min_survivors"],
            "pipeline_survivors": pre["target_survivors"],
            "beam_width": int(beam_width),
            "beam_max_steps": int(max_steps),
            "sffs_max_backward_steps": int(sffs_max_backward_steps),
        },
        "prefilter": pre,
        "beam": {
            "selected_groups": beam_res.get("selected_groups") or [],
            "stop_reason": beam_res.get("stop_reason"),
        },
        "prune": pruned,
        "selected_groups": final_selected,
        "final_features": final_features,
        "final_invert_features": final_invert_features,
        "history": (beam_res.get("history") or []) + (pruned.get("history") or []),
        "candidates_history": (beam_res.get("candidates_history") or []),
        "stop_reason": "completed",
        "rejected_groups": [],
        "objective": objective,
        "min_trades": min_trades,
        "seeds": cfg.seeds,
    }


def successive_halving_search(
    *,
    cfg: SearchConfig,
    base_features: List[str],
    groups: Dict[str, List[str]],
    max_steps: int,
    objective: str,
    min_trades: int,
    stages: List[int],
    top_fraction: float,
    min_survivors: int,
) -> dict:
    """
    Successive Halving over candidates per greedy step.

    Budget dimension: number of seeds (prefix of cfg.seeds).
    """
    tmp_root = cfg.output_dir / "tmp_strategies"
    _ensure_dir(tmp_root)

    selected: List[str] = []
    remaining = list(groups.keys())
    history: List[dict] = []
    candidates_history: List[dict] = []

    current_features = list(base_features)
    eps_improve = 1e-9
    stop_reason = "unknown"
    rejected_groups: List[str] = []

    # Baseline evaluation (full seeds)
    baseline_suffix = "baseline"
    baseline_dir = _make_temp_strategy(
        base_dir=cfg.base_strategy_dir,
        tmp_root=tmp_root,
        name_suffix=baseline_suffix,
        requested_features=current_features,
    )
    _, baseline_summ = run_seed_sweep_for_strategy(
        strategy_dir=baseline_dir, cfg=cfg, run_id=baseline_suffix
    )
    base_score, _, _, base_row = _score_from_summary(
        summary=baseline_summ, objective=objective, min_trades=min_trades
    )
    if base_score is None:
        raise RuntimeError("Baseline evaluation failed")
    best_score = float(base_score)
    baseline_score = float(base_score)
    baseline_summary = base_row

    # Normalize stages
    max_n = max(1, len(cfg.seeds))
    stages = [max(1, min(int(x), max_n)) for x in (stages or []) if int(x) > 0]
    if not stages:
        stages = [1, max_n]
    stages = sorted(set(stages))
    if stages[-1] != max_n:
        stages.append(max_n)

    for step in range(int(max_steps)):
        survivors = list(remaining)
        step_candidates: List[dict] = []

        for si, nseeds in enumerate(stages):
            seeds_subset = cfg.seeds[:nseeds]
            cfg_stage = SearchConfig(
                base_strategy_dir=cfg.base_strategy_dir,
                timeframe=cfg.timeframe,
                symbol=cfg.symbol,
                start_date=cfg.start_date,
                end_date=cfg.end_date,
                test_size=cfg.test_size,
                seeds=list(seeds_subset),
                output_dir=cfg.output_dir,
                deterministic=cfg.deterministic,
                no_docker=cfg.no_docker,
                fast_features=cfg.fast_features,
                invert_eval=cfg.invert_eval,
            )

            stage_results: List[tuple[str, float]] = []
            for g in survivors:
                feats = current_features + groups[g]
                suffix = f"step{step+1}_add_{g}__halving_s{nseeds}"
                strat_dir = _make_temp_strategy(
                    base_dir=cfg.base_strategy_dir,
                    tmp_root=tmp_root,
                    name_suffix=suffix,
                    requested_features=feats,
                )
                _, summ = run_seed_sweep_for_strategy(
                    strategy_dir=strat_dir, cfg=cfg_stage, run_id=suffix
                )
                score, valid, reject_reason, row = _score_from_summary(
                    summary=summ, objective=objective, min_trades=min_trades
                )
                step_candidates.append(
                    {
                        "step": step + 1,
                        "stage": si + 1,
                        "stage_seeds": list(seeds_subset),
                        "candidate_group": g,
                        "score": score,
                        "valid": bool(valid),
                        "reject_reason": reject_reason,
                        "summary": row,
                    }
                )
                if valid and score is not None:
                    stage_results.append((g, float(score)))
                else:
                    if reject_reason:
                        rejected_groups.append(g)

            if not stage_results:
                survivors = []
                break
            stage_results.sort(key=lambda x: x[1], reverse=True)
            keep_n = max(
                int(min_survivors), int(len(stage_results) * float(top_fraction))
            )
            keep_n = max(1, min(keep_n, len(stage_results)))
            survivors = [g for g, _ in stage_results[:keep_n]]
            if len(survivors) <= 1:
                break

        candidates_history.append(
            {
                "step": step + 1,
                "current_selected": list(selected),
                "candidates": step_candidates,
            }
        )

        # Pick best valid entry from the final stage among survivors
        last_stage = len(stages)
        best_candidate = None
        best_candidate_score = None
        best_candidate_summary = None
        surv_set = set(survivors)
        for e in step_candidates:
            if e.get("stage") != last_stage:
                continue
            g = e.get("candidate_group")
            if g not in surv_set:
                continue
            if not e.get("valid"):
                continue
            sc = e.get("score")
            if sc is None:
                continue
            if best_candidate_score is None or float(sc) > float(best_candidate_score):
                best_candidate = str(g)
                best_candidate_score = float(sc)
                best_candidate_summary = e.get("summary") or {}

        if best_candidate is None or best_candidate_score is None:
            stop_reason = "no_valid_candidates"
            break
        if float(best_candidate_score) <= float(best_score) + eps_improve:
            stop_reason = "no_improvement"
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        current_features = current_features + groups[best_candidate]
        best_score = float(best_candidate_score)
        history.append(
            {
                "step": step + 1,
                "added_group": best_candidate,
                "objective": objective,
                "score": best_score,
                "summary": best_candidate_summary,
            }
        )
        if not remaining:
            stop_reason = "exhausted_candidates"
            break

    if stop_reason == "unknown":
        stop_reason = (
            "max_steps_reached" if len(selected) >= int(max_steps) else "completed"
        )
        if stop_reason == "max_steps_reached":
            rejected_groups = list(remaining)

    return {
        "base_strategy": cfg.base_strategy_dir.name,
        "base_features": base_features,
        "baseline": {"score": baseline_score, "summary": baseline_summary},
        "search_algo": "successive_halving",
        "algo_params": {
            "stages": stages,
            "top_fraction": float(top_fraction),
            "min_survivors": int(min_survivors),
        },
        "selected_groups": selected,
        "final_features": current_features,
        "history": history,
        "candidates_history": candidates_history,
        "stop_reason": stop_reason,
        "rejected_groups": rejected_groups,
        "objective": objective,
        "min_trades": min_trades,
        "seeds": cfg.seeds,
    }


def beam_search(
    *,
    cfg: SearchConfig,
    base_features: List[str],
    groups: Dict[str, List[str]],
    max_steps: int,
    objective: str,
    min_trades: int,
    beam_width: int,
    base_invert_features: Optional[List[str]] = None,
    invert_by_group: Optional[Dict[str, List[str]]] = None,
) -> dict:
    """
    Beam search over group additions.
    Keeps top-K partial solutions at each depth.
    """
    tmp_root = cfg.output_dir / "tmp_strategies"
    _ensure_dir(tmp_root)

    eps_improve = 1e-9
    beam_width = max(1, int(beam_width))

    baseline_suffix = "baseline"
    base_inv = _stable_dedup(list(base_invert_features or []))
    inv_map = invert_by_group or {}
    baseline_dir = _make_temp_strategy(
        base_dir=cfg.base_strategy_dir,
        tmp_root=tmp_root,
        name_suffix=baseline_suffix,
        requested_features=list(base_features),
        invert_features=base_inv,
    )
    _, baseline_summ = run_seed_sweep_for_strategy(
        strategy_dir=baseline_dir, cfg=cfg, run_id=baseline_suffix
    )
    base_score, _, _, base_row = _score_from_summary(
        summary=baseline_summ, objective=objective, min_trades=min_trades
    )
    if base_score is None:
        raise RuntimeError("Baseline evaluation failed")
    baseline_score = float(base_score)
    baseline_summary = base_row

    # Beam items: (selected_groups, current_features, score, summary)
    beam: List[tuple[List[str], List[str], float, dict]] = [
        ([], list(base_features), float(base_score), base_row)
    ]
    best_item = beam[0]
    best_score = float(best_item[2])

    all_group_keys = list(groups.keys())
    candidates_history: List[dict] = []
    rejected_groups: List[str] = []
    stop_reason = "unknown"

    for step in range(int(max_steps)):
        expansions: List[tuple[List[str], List[str], float, dict]] = []
        step_candidates: List[dict] = []

        for sel, feats, _, _ in beam:
            rem = [g for g in all_group_keys if g not in set(sel)]
            for g in rem:
                new_sel = sel + [g]
                new_feats = feats + groups[g]
                inv_feats = list(base_inv)
                for gg in new_sel:
                    inv_feats.extend([str(x) for x in (inv_map.get(str(gg)) or [])])
                inv_feats = _stable_dedup(inv_feats)
                sig = "__".join(new_sel)
                run_id = f"beam_step{step+1}_sel_{sig}"
                strat_dir = _make_temp_strategy(
                    base_dir=cfg.base_strategy_dir,
                    tmp_root=tmp_root,
                    name_suffix=run_id,
                    requested_features=new_feats,
                    invert_features=inv_feats,
                )
                _, summ = run_seed_sweep_for_strategy(
                    strategy_dir=strat_dir, cfg=cfg, run_id=run_id
                )
                score, valid, reject_reason, row = _score_from_summary(
                    summary=summ, objective=objective, min_trades=min_trades
                )
                step_candidates.append(
                    {
                        "step": step + 1,
                        "parent_path": list(sel),
                        "candidate_group": g,
                        "score": score,
                        "valid": bool(valid),
                        "reject_reason": reject_reason,
                        "summary": row,
                    }
                )
                if valid and score is not None:
                    expansions.append((new_sel, new_feats, float(score), row))
                else:
                    if reject_reason:
                        rejected_groups.append(g)

        candidates_history.append(
            {
                "step": step + 1,
                "current_selected": [b[0] for b in beam],
                "candidates": step_candidates,
            }
        )

        if not expansions:
            stop_reason = "no_valid_candidates"
            break

        expansions.sort(key=lambda x: x[2], reverse=True)
        beam = expansions[:beam_width]

        if beam[0][2] > best_item[2] + eps_improve:
            best_item = beam[0]
            best_score = float(best_item[2])
        else:
            stop_reason = "no_improvement"
            break

        if not beam:
            stop_reason = "no_valid_candidates"
            break

    best_selected, best_features, _, _ = best_item
    history: List[dict] = []
    for i, g in enumerate(best_selected):
        history.append({"step": i + 1, "added_group": g, "objective": objective})

    if stop_reason == "unknown":
        stop_reason = (
            "max_steps_reached" if len(best_selected) >= int(max_steps) else "completed"
        )

    return {
        "base_strategy": cfg.base_strategy_dir.name,
        "base_features": base_features,
        "baseline": {"score": baseline_score, "summary": baseline_summary},
        "search_algo": "beam",
        "algo_params": {"beam_width": beam_width},
        "selected_groups": best_selected,
        "final_features": best_features,
        "history": history,
        "candidates_history": candidates_history,
        "stop_reason": stop_reason,
        "rejected_groups": rejected_groups,
        "objective": objective,
        "min_trades": min_trades,
        "seeds": cfg.seeds,
    }


def sffs_search(
    *,
    cfg: SearchConfig,
    base_features: List[str],
    groups: Dict[str, List[str]],
    max_steps: int,
    objective: str,
    min_trades: int,
    max_backward_per_step: int,
    base_invert_features: Optional[List[str]] = None,
    invert_by_group: Optional[Dict[str, List[str]]] = None,
) -> dict:
    """
    Sequential Floating Forward Selection (SFFS):
    forward add best group, then backward remove any selected group if it improves score.
    """
    tmp_root = cfg.output_dir / "tmp_strategies"
    _ensure_dir(tmp_root)

    eps_improve = 1e-9
    max_backward_per_step = max(1, int(max_backward_per_step))

    selected: List[str] = []
    remaining = list(groups.keys())
    history: List[dict] = []
    candidates_history: List[dict] = []
    rejected_groups: List[str] = []
    stop_reason = "unknown"

    # Baseline
    baseline_suffix = "baseline"
    base_inv = _stable_dedup(list(base_invert_features or []))
    inv_map = invert_by_group or {}
    baseline_dir = _make_temp_strategy(
        base_dir=cfg.base_strategy_dir,
        tmp_root=tmp_root,
        name_suffix=baseline_suffix,
        requested_features=list(base_features),
        invert_features=base_inv,
    )
    _, baseline_summ = run_seed_sweep_for_strategy(
        strategy_dir=baseline_dir, cfg=cfg, run_id=baseline_suffix
    )
    base_score, _, _, base_row = _score_from_summary(
        summary=baseline_summ, objective=objective, min_trades=min_trades
    )
    if base_score is None:
        raise RuntimeError("Baseline evaluation failed")
    baseline_score = float(base_score)
    baseline_summary = base_row
    best_score = float(base_score)

    current_features = list(base_features)

    def _eval_featureset(
        feats: List[str], inv_feats: List[str], run_id: str
    ) -> tuple[float | None, bool, str | None, dict]:
        strat_dir = _make_temp_strategy(
            base_dir=cfg.base_strategy_dir,
            tmp_root=tmp_root,
            name_suffix=run_id,
            requested_features=feats,
            invert_features=inv_feats,
        )
        _, summ = run_seed_sweep_for_strategy(
            strategy_dir=strat_dir, cfg=cfg, run_id=run_id
        )
        return _score_from_summary(
            summary=summ, objective=objective, min_trades=min_trades
        )

    for step in range(int(max_steps)):
        step_candidates: List[dict] = []

        # Forward add
        best_add = None
        best_add_score = None
        best_add_summary = None
        for g in list(remaining):
            new_sel = selected + [g]
            new_feats = current_features + groups[g]
            inv_feats = list(base_inv)
            for gg in new_sel:
                inv_feats.extend([str(x) for x in (inv_map.get(str(gg)) or [])])
            inv_feats = _stable_dedup(inv_feats)
            sig = "__".join(new_sel)
            run_id = f"sffs_step{step+1}_fwd_sel_{sig}"
            score, valid, reject_reason, row = _eval_featureset(
                new_feats, inv_feats, run_id
            )
            step_candidates.append(
                {
                    "step": step + 1,
                    "phase": "forward",
                    "candidate_group": g,
                    "score": score,
                    "valid": bool(valid),
                    "reject_reason": reject_reason,
                    "summary": row,
                }
            )
            if valid and score is not None:
                if best_add_score is None or float(score) > float(best_add_score):
                    best_add = g
                    best_add_score = float(score)
                    best_add_summary = row
            else:
                if reject_reason:
                    rejected_groups.append(g)

        if best_add is None or best_add_score is None:
            stop_reason = "no_valid_candidates"
            candidates_history.append({"step": step + 1, "candidates": step_candidates})
            break
        if float(best_add_score) <= float(best_score) + eps_improve:
            stop_reason = "no_improvement"
            candidates_history.append({"step": step + 1, "candidates": step_candidates})
            break

        # Accept add
        selected.append(best_add)
        remaining.remove(best_add)
        current_features = current_features + groups[best_add]
        best_score = float(best_add_score)
        history.append(
            {
                "step": step + 1,
                "added_group": best_add,
                "objective": objective,
                "score": best_score,
                "summary": best_add_summary or {},
            }
        )

        # Backward floating remove
        for _ in range(max_backward_per_step):
            if len(selected) <= 1:
                break
            best_rm = None
            best_rm_score = None
            best_rm_summary = None
            for rm in list(selected):
                kept = [x for x in selected if x != rm]
                feats = list(base_features)
                for gg in kept:
                    feats = feats + groups[gg]
                inv_feats = list(base_inv)
                for gg in kept:
                    inv_feats.extend([str(x) for x in (inv_map.get(str(gg)) or [])])
                inv_feats = _stable_dedup(inv_feats)
                sig = "__".join(kept) if kept else "none"
                run_id = f"sffs_step{step+1}_bwd_sel_{sig}__rm_{rm}"
                score, valid, reject_reason, row = _eval_featureset(
                    feats, inv_feats, run_id
                )
                step_candidates.append(
                    {
                        "step": step + 1,
                        "phase": "backward",
                        "candidate_group": rm,
                        "score": score,
                        "valid": bool(valid),
                        "reject_reason": reject_reason,
                        "summary": row,
                    }
                )
                if valid and score is not None:
                    if best_rm_score is None or float(score) > float(best_rm_score):
                        best_rm = rm
                        best_rm_score = float(score)
                        best_rm_summary = row
            if best_rm is None or best_rm_score is None:
                break
            if float(best_rm_score) <= float(best_score) + eps_improve:
                break
            # Remove
            selected = [x for x in selected if x != best_rm]
            current_features = list(base_features)
            for gg in selected:
                current_features = current_features + groups[gg]
            best_score = float(best_rm_score)
            history.append(
                {
                    "step": step + 1,
                    "removed_group": best_rm,
                    "objective": objective,
                    "score": best_score,
                    "summary": best_rm_summary or {},
                }
            )

        candidates_history.append({"step": step + 1, "candidates": step_candidates})
        if not remaining:
            stop_reason = "exhausted_candidates"
            break

    if stop_reason == "unknown":
        stop_reason = (
            "max_steps_reached" if len(selected) >= int(max_steps) else "completed"
        )

    return {
        "base_strategy": cfg.base_strategy_dir.name,
        "base_features": base_features,
        "baseline": {"score": baseline_score, "summary": baseline_summary},
        "search_algo": "sffs",
        "algo_params": {"max_backward_per_step": max_backward_per_step},
        "selected_groups": selected,
        "final_features": current_features,
        "history": history,
        "candidates_history": candidates_history,
        "stop_reason": stop_reason,
        "rejected_groups": rejected_groups,
        "objective": objective,
        "min_trades": min_trades,
        "seeds": cfg.seeds,
    }


def _default_groups() -> Dict[str, List[str]]:
    """
    Conservative group list (keeps "feature explosion" manageable).
    All keys are group names; values are requested_features nodes.
    """
    return {
        # Cheap, non-tick
        "kline_core": [
            "macd_f",
            "rsi_f",
            "sma_200_f",
            "atr_f",
            "trend_r2_20_f",
            "bb_width_f",
            "wick_ratios_f",
        ],
        "sr_structure_min": [
            "poc_hal_features_close_f",
            "sqs_hal_high_f",
            "sqs_hal_low_f",
            "sr_strength_max_close_f",
        ],
        "volume_profile": ["volume_profile_volatility_features_f"],
        "volume_profile_scene": ["volume_profile_scene_semantic_scores_f"],
        "wpt_energy": ["wpt_volume_energy_f"],
        "wpt_scene": ["wpt_scene_semantic_scores_f"],
        "liquidity_void": ["liquidity_void_f"],
        "liquidity_void_scene": ["liquidity_void_scene_semantic_scores_f"],
        "compression": ["compression_score_f", "compression_energy_f"],
        "wick_scene": ["wick_scene_semantic_scores_f"],
        # Tick-heavy
        "footprint_basic": ["footprint_basic_f"],
        "fp_scene": ["fp_imbalance_scene_semantic_scores_f"],
        "vpin_block": ["vpin_features_f"],
        "vpin_scene": ["vpin_scene_semantic_scores_f"],
        "trade_cluster_semantic": ["trade_cluster_semantic_scores_f"],
        "trade_cluster_scene": ["trade_cluster_scene_semantic_scores_f"],
    }


def _normalize_groups(obj: object) -> Dict[str, List[str]]:
    """
    Normalize a groups mapping loaded from YAML/JSON.
    Supports either:
      - {group_name: [node1, node2, ...]}
      - {"groups": {group_name: [node1, node2, ...]}}
    """
    if obj is None:
        return {}
    if isinstance(obj, dict) and "groups" in obj and isinstance(obj["groups"], dict):
        obj = obj["groups"]
    if not isinstance(obj, dict):
        raise ValueError(
            "groups must be a mapping: {group_name: [requested_feature_nodes...]}"
        )
    out: Dict[str, List[str]] = {}
    for k, v in obj.items():
        if not isinstance(k, str) or not k.strip():
            continue
        if v is None:
            out[k] = []
            continue
        if not isinstance(v, list):
            raise ValueError(f"group '{k}' must be a list, got: {type(v)}")
        cleaned = [str(x).strip() for x in v if str(x).strip()]
        out[str(k).strip()] = cleaned
    return out


def _expand_semantic_groups_to_singletons(
    groups: Dict[str, List[str]],
    feature_deps_path: str = "config/feature_dependencies.yaml",
) -> Dict[str, List[str]]:
    """
    Expand semantic groups into singleton groups (one per output column).

    For each feature node in groups, if it has multiple output_columns,
    create a singleton group for each output column.

    This allows fine-grained selection of semantic scores (e.g., ignition vs exhaustion)
    which may have opposite effects on the strategy.

    Example:
        Input: {"trade_cluster_scene": ["trade_cluster_scene_semantic_scores_f"]}
        Output: {
            "trade_cluster_scene__compression": ["trade_cluster_compression_score"],
            "trade_cluster_scene__ignition": ["trade_cluster_ignition_score"],
            "trade_cluster_scene__absorption": ["trade_cluster_absorption_scene_score"],
            "trade_cluster_scene__exhaustion": ["trade_cluster_exhaustion_scene_score"],
        }

    Note: The feature node itself is still required as a dependency, but we only
    select specific output columns for training.
    """
    import yaml
    from pathlib import Path

    # Load feature dependencies
    deps_path = Path(feature_deps_path)
    if not deps_path.exists():
        print(f"   ⚠️  Feature dependencies file not found: {deps_path}")
        print(f"      Keeping semantic groups as-is (no expansion)")
        return groups

    with open(deps_path, "r", encoding="utf-8") as f:
        feature_deps = yaml.safe_load(f)
    features = feature_deps.get("features", {})

    expanded = {}
    expanded_count = 0

    for group_name, feature_nodes in groups.items():
        for node in feature_nodes:
            if node not in features:
                # Keep as-is if not found in feature_deps
                key = f"{group_name}__{node}"
                if key not in expanded:
                    expanded[key] = [node]
                continue

            output_cols = features[node].get("output_columns", [])
            if len(output_cols) <= 1:
                # Single output or no output_columns defined, keep as-is
                key = f"{group_name}__{node}"
                if key not in expanded:
                    expanded[key] = [node]
            else:
                # Multiple outputs: create singleton for each output column
                # Note: The feature node itself is still required for dependency resolution,
                # but we request specific output columns by name (feature loader will handle this)
                for col in output_cols:
                    # Extract semantic name from column
                    # e.g., "trade_cluster_ignition_score" -> "ignition"
                    # e.g., "vpin_compression_score" -> "compression"
                    semantic_name = col
                    # Try to extract a cleaner name
                    if "_score" in col:
                        # Remove common prefixes and suffixes
                        parts = (
                            col.replace("_score", "").replace("_scene", "").split("_")
                        )
                        # Find the semantic part (usually the last meaningful word)
                        if len(parts) >= 2:
                            # e.g., ["trade", "cluster", "ignition"] -> "ignition"
                            # e.g., ["vpin", "compression"] -> "compression"
                            # Skip common prefixes
                            skip_words = {
                                "trade",
                                "cluster",
                                "vpin",
                                "wpt",
                                "funding",
                                "fp",
                                "wick",
                                "volume",
                                "profile",
                                "liquidity",
                                "void",
                            }
                            for part in reversed(parts):
                                if part not in skip_words and len(part) > 3:
                                    semantic_name = part
                                    break
                            else:
                                semantic_name = parts[-1]

                    key = f"{group_name}__{semantic_name}"
                    # Ensure unique key
                    original_key = key
                    i = 2
                    while key in expanded:
                        key = f"{original_key}__{i}"
                        i += 1

                    # Store as output column name (feature loader will resolve to the feature node)
                    # The feature loader's resolve_dependencies will map output columns back to feature nodes
                    expanded[key] = [col]
                    expanded_count += 1

    if expanded_count > 0:
        print(f"   ✅ Expanded {expanded_count} semantic groups to singleton groups")
        print(f"      Total groups: {len(expanded)} (was {len(groups)})")

    return expanded


def _apply_feature_blacklist(
    *,
    base_features: List[str],
    groups: Dict[str, List[str]],
    blacklist: List[str],
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Remove blacklisted requested_feature nodes from base_features and groups."""
    bl = {b.strip() for b in blacklist if isinstance(b, str) and b.strip()}
    if not bl:
        return base_features, groups
    base_out = [f for f in base_features if f not in bl]
    groups_out: Dict[str, List[str]] = {}
    for g, feats in groups.items():
        groups_out[g] = [f for f in (feats or []) if f not in bl]
    return base_out, groups_out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base-strategy-config",
        required=True,
        help="Base strategy config dir (single strategy)",
    )
    p.add_argument("--symbol", required=True)
    p.add_argument("--timeframe", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--seeds", default="1,2,3", help="Comma-separated seeds")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--no-docker", action="store_true", default=True)
    p.add_argument("--objective", default="Sharpe_mean")
    p.add_argument("--min-trades", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument(
        "--invert-eval",
        default="conservative",
        choices=["none", "conservative", "all"],
        help=(
            "How to validate Pool-B invert candidates (output columns) during search. "
            "none=never try inversion; conservative=try and pick only when raw is clearly negative; "
            "all=always try raw vs inverted for candidate columns and pick the better one."
        ),
    )
    p.add_argument(
        "--fast-features",
        action="store_true",
        default=False,
        help=(
            "Enable faster feature computation for search runs (sets env FEATURE_FAST_MODE=1). "
            "This can disable DTW random templates and compute spectrum features at a lower frequency. "
            "Recommended for preset A/B; typically keep disabled for final verification (C)."
        ),
    )
    p.add_argument(
        "--preset",
        default="",
        choices=["", "A", "B", "C"],
        help=(
            "Budget preset to speed up feature-group-search. "
            "A=fast proxy (CV_mean, fewer seeds), B=medium, C=full (Sharpe_mean, more seeds). "
            "If set, this will override related args (seeds/objective/halving/beam/sffs/max-steps)."
        ),
    )
    p.add_argument(
        "--search-algo",
        default="greedy",
        choices=["greedy", "halving", "beam", "sffs", "pipeline"],
        help="Search algorithm: greedy (default), halving, beam, sffs, or pipeline (halving->beam->sffs).",
    )
    # Successive Halving params
    p.add_argument(
        "--halving-stages",
        default="1,3,5",
        help="Comma-separated seed counts to use as budgets, e.g. '1,3,5'. Will auto-append full seeds.",
    )
    p.add_argument(
        "--halving-top-fraction",
        type=float,
        default=0.25,
        help="Fraction of candidates to keep at each halving stage (0,1].",
    )
    p.add_argument(
        "--halving-min-survivors",
        type=int,
        default=5,
        help="Minimum number of survivors to keep at each halving stage.",
    )
    # Beam params
    p.add_argument(
        "--beam-width",
        type=int,
        default=3,
        help="Beam width (top-K paths to keep).",
    )
    # SFFS params
    p.add_argument(
        "--sffs-max-backward-per-step",
        type=int,
        default=2,
        help="Max backward removals to try per forward step in SFFS.",
    )
    # Pipeline params
    p.add_argument(
        "--pipeline-survivors",
        type=int,
        default=30,
        help="Pipeline only: target number of candidate groups to keep after halving prefilter.",
    )
    p.add_argument(
        "--groups-json", default=None, help="Optional groups override (JSON file path)"
    )
    p.add_argument(
        "--groups-yaml", default=None, help="Optional groups override (YAML file path)"
    )
    p.add_argument(
        "--pool-b-yaml",
        default=None,
        help=(
            "Optional Pool B YAML exported by factor-eval (features_pool_b.yaml). "
            "If provided, the tool will auto-generate extra singleton groups for any "
            "feature_pipeline.requested_features not already present in groups."
        ),
    )
    p.add_argument(
        "--base-features-yaml",
        default=None,
        help="Optional base features list YAML path",
    )
    p.add_argument(
        "--writeback-yaml",
        default=None,
        help="Optional output path to write a features_suggested.yaml (requested_features=final_features) with provenance metadata.",
    )
    p.add_argument(
        "--expand-semantic-singletons",
        action="store_true",
        default=False,
        help=(
            "If True, expand semantic groups into singleton groups (one per output column). "
            "This allows fine-grained selection of semantic scores (e.g., ignition vs exhaustion) "
            "which may have opposite effects on the strategy. "
            "Default: False (keep semantic groups as-is for backward compatibility)."
        ),
    )
    p.add_argument(
        "--invert-candidates-yaml",
        default=None,
        help=(
            "Optional YAML path providing invert candidates. Accepts either a YAML list, or a full "
            "features config containing feature_pipeline.invert_features. On writeback, the tool "
            "will set invert_features = invert_candidates ∩ final_requested_features."
        ),
    )
    p.add_argument(
        "--feature-blacklist",
        default="",
        help="Comma-separated requested_feature nodes to exclude from BOTH base and candidate groups",
    )
    return p.parse_args()


def _apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    """
    Apply A/B/C budget presets for faster iteration.

    Notes:
    - Presets intentionally override user-provided flags to ensure deterministic budgets.
      If you want full manual control, do not pass --preset.
    - halving_stages represent *seed budgets* (e.g. 1,2,3), and the implementation
      will auto-append the full seed count.
    """
    preset = (getattr(args, "preset", "") or "").strip().upper()
    if not preset:
        return args

    presets = {
        # A: very fast screening (use proxy objective + fewer seeds)
        "A": {
            "objective": "CV_mean",
            "seeds": "1,2",
            "halving_stages": "1,2",
            "halving_top_fraction": 0.35,
            "halving_min_survivors": 20,
            "pipeline_survivors": 25,
            "beam_width": 3,
            "max_steps": 4,
            "sffs_max_backward_per_step": 1,
            "fast_features": True,
        },
        # B: medium (more seeds + slightly wider search)
        "B": {
            "objective": "CV_mean",
            "seeds": "1,2,3",
            "halving_stages": "1,2,3",
            "halving_top_fraction": 0.5,
            "halving_min_survivors": 30,
            "pipeline_survivors": 40,
            "beam_width": 4,
            "max_steps": 5,
            "sffs_max_backward_per_step": 1,
            "fast_features": True,
        },
        # C: full verification (closest to your wide runs)
        "C": {
            "objective": "Sharpe_mean",
            "seeds": "1,2,3,4,5",
            "halving_stages": "1,3,5",
            "halving_top_fraction": 0.6,
            "halving_min_survivors": 40,
            "pipeline_survivors": 60,
            "beam_width": 5,
            "max_steps": 6,
            "sffs_max_backward_per_step": 2,
            "fast_features": False,
        },
    }

    cfg = presets.get(preset)
    if cfg is None:
        return args

    # Apply
    for k, v in cfg.items():
        setattr(args, k, v)

    print(
        f"⚙️  Applied preset {preset}: "
        f"objective={args.objective}, seeds={args.seeds}, "
        f"halving_stages={args.halving_stages}, top_fraction={args.halving_top_fraction}, "
        f"min_survivors={args.halving_min_survivors}, pipeline_survivors={getattr(args,'pipeline_survivors',None)}, "
        f"beam_width={getattr(args,'beam_width',None)}, max_steps={args.max_steps}, "
        f"sffs_max_backward_per_step={getattr(args,'sffs_max_backward_per_step',None)}"
    )
    return args


def _load_groups_with_source(
    *,
    strategy_dir_name: str,
    groups_json: str | None,
    groups_yaml: str | None,
) -> tuple[Dict[str, List[str]], str, bool]:
    """
    Resolve candidate groups + provenance.

    Priority:
      1) --groups-json
      2) --groups-yaml
      3) config/feature_groups_<strategy_dir_name>_semantic.yaml (auto, if exists)
      4) config/feature_groups.yaml (auto, if exists)
      5) _default_groups() (code fallback)
    """
    if groups_json:
        p = Path(groups_json)
        return (
            _normalize_groups(json.loads(p.read_text(encoding="utf-8"))),
            f"groups_json:{groups_json}",
            False,
        )
    if groups_yaml:
        p = Path(groups_yaml)
        return (
            _normalize_groups(yaml.safe_load(p.read_text(encoding="utf-8"))),
            f"groups_yaml:{groups_yaml}",
            False,
        )

    # Auto strategy-specific convention first: config/feature_groups_<strategy>_semantic.yaml
    strategy_yaml = Path("config") / f"feature_groups_{strategy_dir_name}_semantic.yaml"
    if strategy_yaml.exists():
        return (
            _normalize_groups(
                yaml.safe_load(strategy_yaml.read_text(encoding="utf-8"))
            ),
            f"groups_yaml:auto:{strategy_yaml.as_posix()}",
            True,
        )

    # Auto global convention: config/feature_groups.yaml
    auto_yaml = Path("config") / "feature_groups.yaml"
    if auto_yaml.exists():
        return (
            _normalize_groups(yaml.safe_load(auto_yaml.read_text(encoding="utf-8"))),
            f"groups_yaml:auto:{auto_yaml.as_posix()}",
            True,
        )

    return _default_groups(), "default_groups", False


def main() -> None:
    args = _apply_preset(_parse_args())
    base_dir = Path(args.base_strategy_config)
    out_dir = Path(args.output_dir)
    _ensure_dir(out_dir)

    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    cfg = SearchConfig(
        base_strategy_dir=base_dir,
        timeframe=str(args.timeframe),
        symbol=str(args.symbol),
        start_date=str(args.start_date),
        end_date=str(args.end_date),
        test_size=float(args.test_size),
        seeds=seeds,
        output_dir=out_dir,
        deterministic=bool(args.deterministic),
        no_docker=bool(args.no_docker),
        fast_features=bool(getattr(args, "fast_features", False)),
        invert_eval=str(getattr(args, "invert_eval", "conservative")),
    )

    groups, resolved_groups_source, groups_yaml_auto = _load_groups_with_source(
        strategy_dir_name=base_dir.name,
        groups_json=args.groups_json,
        groups_yaml=args.groups_yaml,
    )

    # Optional: expand semantic groups to singletons (one per output column)
    # This allows fine-grained selection of semantic scores (e.g., ignition vs exhaustion)
    if args.expand_semantic_singletons:
        groups = _expand_semantic_groups_to_singletons(groups)

    # Convention-by-default:
    # If user didn't pass pool-b-yaml / invert-candidates-yaml, try the conventional location
    # derived from the base strategy directory name:
    #   results/pools/<strategy_dir_name>/pool_b/features_pool_b.yaml
    # (We intentionally do NOT hard fail if absent; the tool can still run without Pool B.)
    conventional_pool_b = (
        Path("results") / "pools" / base_dir.name / "pool_b" / "features_pool_b.yaml"
    )
    if not args.pool_b_yaml and conventional_pool_b.exists():
        args.pool_b_yaml = str(conventional_pool_b)
    if not args.invert_candidates_yaml and conventional_pool_b.exists():
        args.invert_candidates_yaml = str(conventional_pool_b)

    # Optional: merge Pool B (factor-eval) candidates as singleton groups.
    # This is how we make a semantic groups file compatible with a dynamic factor pool.
    if args.pool_b_yaml:
        pool_obj = (
            yaml.safe_load(Path(args.pool_b_yaml).read_text(encoding="utf-8")) or {}
        )
        pool_fp = (
            pool_obj.get("feature_pipeline") if isinstance(pool_obj, dict) else None
        )
        pool_req = (
            pool_fp.get("requested_features") if isinstance(pool_fp, dict) else None
        )
        pool_req = pool_req if isinstance(pool_req, list) else []
        pool_inv = pool_fp.get("invert_features") if isinstance(pool_fp, dict) else None
        pool_inv = pool_inv if isinstance(pool_inv, list) else []

        used_nodes = set()
        for feats in (groups or {}).values():
            for f in feats or []:
                used_nodes.add(str(f))

        # Add singleton groups for any pool feature not already present in groups.
        for f in pool_req:
            f = str(f).strip()
            if not f:
                continue
            if f in used_nodes:
                continue
            key = f"poolb__{f}"
            # Ensure unique group name even if collisions occur
            if key in groups:
                i = 2
                while f"{key}__{i}" in groups:
                    i += 1
                key = f"{key}__{i}"
            groups[key] = [f]

        # Add singleton groups for Pool-B inverted OUTPUT columns, so we can validate sign per column.
        # This makes "Pool-B inverted factors" first-class candidates in the search space.
        for col in pool_inv:
            c = str(col).strip()
            if not c:
                continue
            # Avoid clobbering existing semantic singleton keys; use a dedicated namespace.
            key = f"poolb_invcol__{c}"
            if key in groups:
                i = 2
                while f"{key}__{i}" in groups:
                    i += 1
                key = f"{key}__{i}"
            groups[key] = [c]

    # Resolve base_features: these are "must-have" features required by label generator / backtest
    # They are NOT part of the optimization; they always stay in the feature set.
    #
    # Priority:
    #   1) --base-features-yaml (explicit)
    #   2) <strategy_dir>/features_base.yaml (convention)
    #   3) [] (empty, may cause label generation to fail if strategy requires specific features)

    base_features_path = None
    if args.base_features_yaml:
        base_features_path = Path(args.base_features_yaml)
    else:
        # Auto-detect features_base.yaml in strategy directory
        conventional_base = base_dir / "features_base.yaml"
        if conventional_base.exists():
            base_features_path = conventional_base

    if base_features_path and base_features_path.exists():
        base_features = (
            yaml.safe_load(base_features_path.read_text(encoding="utf-8")) or []
        )
        if not isinstance(base_features, list):
            raise ValueError(
                f"base-features-yaml ({base_features_path}) must be a YAML list"
            )
        print(
            f"   📋 Using base_features from {base_features_path} "
            f"({len(base_features)} features required by label/backtest)"
        )
    else:
        # No base features specified - start from empty
        # This may cause issues if label generator requires specific features!
        base_features = []
        print(
            f"   ⚠️ No base_features specified (starting from scratch). "
            f"If label generation fails, create {base_dir / 'features_base.yaml'}"
        )

    blacklist = [s.strip() for s in str(args.feature_blacklist).split(",") if s.strip()]
    base_features, groups = _apply_feature_blacklist(
        base_features=list(base_features),
        groups=dict(groups),
        blacklist=blacklist,
    )

    invert_candidates = None
    if args.invert_candidates_yaml:
        invert_candidates = _load_invert_candidates(Path(args.invert_candidates_yaml))

    t0 = time.time()
    algo = str(getattr(args, "search_algo", "greedy")).strip().lower()
    if algo == "halving":
        stages = _parse_int_list(str(getattr(args, "halving_stages", "1,3,5")))
        result = successive_halving_search(
            cfg=cfg,
            base_features=base_features,
            groups=groups,
            max_steps=int(args.max_steps),
            objective=str(args.objective),
            min_trades=int(args.min_trades),
            stages=stages,
            top_fraction=float(getattr(args, "halving_top_fraction", 0.25)),
            min_survivors=int(getattr(args, "halving_min_survivors", 5)),
        )
    elif algo == "pipeline":
        stages = _parse_int_list(str(getattr(args, "halving_stages", "1,3,5")))
        result = pipeline_sh_beam_sffs(
            cfg=cfg,
            base_features=base_features,
            groups=groups,
            max_steps=int(args.max_steps),
            objective=str(args.objective),
            min_trades=int(args.min_trades),
            stages=stages,
            top_fraction=float(getattr(args, "halving_top_fraction", 0.25)),
            min_survivors=int(getattr(args, "halving_min_survivors", 5)),
            target_survivors=int(getattr(args, "pipeline_survivors", 30)),
            beam_width=int(getattr(args, "beam_width", 3)),
            sffs_max_backward_steps=int(getattr(args, "sffs_max_backward_per_step", 2)),
            invert_candidates=invert_candidates,
        )
    elif algo == "beam":
        result = beam_search(
            cfg=cfg,
            base_features=base_features,
            groups=groups,
            max_steps=int(args.max_steps),
            objective=str(args.objective),
            min_trades=int(args.min_trades),
            beam_width=int(getattr(args, "beam_width", 3)),
        )
    elif algo == "sffs":
        result = sffs_search(
            cfg=cfg,
            base_features=base_features,
            groups=groups,
            max_steps=int(args.max_steps),
            objective=str(args.objective),
            min_trades=int(args.min_trades),
            max_backward_per_step=int(getattr(args, "sffs_max_backward_per_step", 2)),
        )
    else:
        result = greedy_forward_search(
            cfg=cfg,
            base_features=base_features,
            groups=groups,
            max_steps=int(args.max_steps),
            objective=str(args.objective),
            min_trades=int(args.min_trades),
        )
    result["elapsed_sec"] = round(time.time() - t0, 2)

    (out_dir / "feature_group_search_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Write a compact CSV of history (one row per step)
    hist_rows = []
    for h in result.get("history", []):
        # Some algorithms (beam) may not record per-step score/summary in the same shape.
        row = {
            "step": h.get("step"),
            "added_group": h.get("added_group"),
            "score": h.get("score"),
        }
        summ = h.get("summary") or {}
        for k in [
            "Sharpe_mean",
            "return%_mean",
            "DD%_mean",
            "trades_mean",
            "corr_mean",
            "CV_mean",
        ]:
            if k in summ:
                row[k] = summ[k]
        hist_rows.append(row)
    pd.DataFrame(hist_rows).to_csv(
        out_dir / "feature_group_search_history.csv", index=False
    )

    # Write candidate table (one row per evaluated candidate group per step)
    cand_rows = []
    for step_obj in result.get("candidates_history", []) or []:
        step_n = step_obj.get("step")
        for c in step_obj.get("candidates", []) or []:
            summ = c.get("summary") or {}
            row = {
                "step": step_n,
                "candidate_group": c.get("candidate_group"),
                "score": c.get("score"),
            }
            for k in [
                "Sharpe_mean",
                "return%_mean",
                "DD%_mean",
                "trades_mean",
                "corr_mean",
                "CV_mean",
            ]:
                if k in summ:
                    row[k] = summ[k]
            cand_rows.append(row)
    if cand_rows:
        pd.DataFrame(cand_rows).to_csv(
            out_dir / "feature_group_search_candidates.csv", index=False
        )

    # Optional YAML writeback (features_suggested.yaml)
    if args.writeback_yaml:
        meta = {
            "base_strategy_dir": str(cfg.base_strategy_dir),
            "objective": result.get("objective"),
            "min_trades": result.get("min_trades"),
            "seeds": result.get("seeds"),
            "selected_groups": result.get("selected_groups", []),
            "stop_reason": result.get("stop_reason"),
            "base_features": result.get("base_features", []),
            "final_features": result.get("final_features", []),
            "groups_source": resolved_groups_source,
            "groups_yaml_auto": groups_yaml_auto,
            "pool_b_yaml": str(args.pool_b_yaml) if args.pool_b_yaml else None,
            "pool_b_merged_singletons": bool(args.pool_b_yaml),
            "feature_blacklist": blacklist,
        }
        inv_for_writeback = result.get("final_invert_features") or invert_candidates
        writeback_info = _writeback_features_yaml(
            base_strategy_dir=cfg.base_strategy_dir,
            out_path=Path(args.writeback_yaml),
            requested_features=list(result.get("final_features") or []),
            meta=meta,
            invert_candidates=inv_for_writeback,
        )
        result["writeback"] = writeback_info
        (out_dir / "feature_group_search_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Minimal HTML
    html = []
    html.append("<html><head><meta charset='utf-8'><title>Feature Group Search</title>")
    html.append(
        "<style>body{font-family:Arial,Helvetica,sans-serif;padding:16px} table{border-collapse:collapse} td,th{border:1px solid #ddd;padding:6px} th{background:#f5f5f5}</style>"
    )
    html.append("</head><body>")
    html.append("<h2>Feature Group Search (Greedy Forward Selection)</h2>")
    html.append(
        f"<p><b>Base</b>: {cfg.base_strategy_dir} &nbsp; <b>Symbol</b>: {cfg.symbol} &nbsp; <b>Timeframe</b>: {cfg.timeframe}<br/>"
    )
    html.append(
        f"<b>Window</b>: {cfg.start_date} → {cfg.end_date} &nbsp; <b>Test-size</b>: {cfg.test_size} &nbsp; <b>Seeds</b>: {', '.join(map(str,cfg.seeds))}<br/>"
    )
    html.append(
        f"<b>Objective</b>: {result.get('objective')} &nbsp; <b>Min trades</b>: {result.get('min_trades')}</p>"
    )

    # Artifact quick links (same directory)
    html.append("<h3>Artifacts</h3>")
    html.append(
        "<ul>"
        "<li><b>result</b>: <a href='feature_group_search_result.json'><code>feature_group_search_result.json</code></a></li>"
        "<li><b>history</b>: <a href='feature_group_search_history.csv'><code>feature_group_search_history.csv</code></a></li>"
        "<li><b>candidates</b>: <a href='feature_group_search_candidates.csv'><code>feature_group_search_candidates.csv</code></a></li>"
        "<li><b>report</b>: <a href='feature_group_search_report.html'><code>feature_group_search_report.html</code></a></li>"
        "<li><b>why (drilldown)</b>: <a href='feature_group_search_why.html'><code>feature_group_search_why.html</code></a></li>"
        "</ul>"
    )

    # Baseline + stop reason (make it obvious)
    html.append("<h3>Baseline &amp; stop</h3>")
    baseline = result.get("baseline") or {}
    baseline_score = baseline.get("score")
    stop_reason = result.get("stop_reason")
    rejected = result.get("rejected_groups") or []
    html.append(
        "<p>"
        f"<b>baseline_score</b>: {baseline_score} &nbsp; "
        f"<b>stop_reason</b>: {stop_reason} &nbsp; "
        f"<b>rejected_groups</b>: {len(rejected)}"
        "</p>"
    )
    baseline_summary = baseline.get("summary")
    if isinstance(baseline_summary, dict) and baseline_summary:
        keep_keys = [
            "Sharpe_mean",
            "return%_mean",
            "DD%_mean",
            "trades_mean",
            "CV_mean",
            "corr_mean",
        ]
        row = {k: baseline_summary.get(k) for k in keep_keys if k in baseline_summary}
        row = {"baseline_" + k: v for k, v in row.items()}
        if row:
            html.append(pd.DataFrame([row]).to_html(index=False, escape=False))
    if rejected:
        html.append(
            "<details><summary><b>Rejected groups</b> (click to expand)</summary>"
        )
        html.append("<pre>" + "\n".join([str(x) for x in rejected]) + "</pre>")
        html.append("</details>")
    html.append("<h3>Selected groups</h3>")
    html.append("<pre>" + "\n".join(result.get("selected_groups", [])) + "</pre>")
    html.append("<h3>History</h3>")
    if hist_rows:
        html.append(pd.DataFrame(hist_rows).to_html(index=False, escape=False))
    if cand_rows:
        html.append("<h3>All candidate scores (per step)</h3>")
        html.append(
            "<p>Tip: use the <b>why (drilldown)</b> HTML for step-by-step explanations.</p>"
        )
        html.append(pd.DataFrame(cand_rows).to_html(index=False, escape=False))
    html.append("<h3>Final requested_features</h3>")
    html.append("<pre>" + "\n".join(result.get("final_features", [])) + "</pre>")
    if result.get("writeback"):
        html.append("<h3>Writeback</h3>")
        html.append(
            "<pre>"
            + json.dumps(result.get("writeback"), ensure_ascii=False, indent=2)
            + "</pre>"
        )
    html.append("</body></html>")
    (out_dir / "feature_group_search_report.html").write_text(
        "\n".join(html), encoding="utf-8"
    )

    # Candidate drilldown HTML ("why selected / why rejected")
    why = []
    why.append(
        "<html><head><meta charset='utf-8'><title>Feature Group Search - Why</title>"
    )
    why.append(
        "<style>body{font-family:Arial,Helvetica,sans-serif;padding:16px} table{border-collapse:collapse} td,th{border:1px solid #ddd;padding:6px} th{background:#f5f5f5} code{background:#f5f5f5;padding:1px 3px;border-radius:3px}</style>"
    )
    why.append("</head><body>")
    why.append("<h2>Why selected vs rejected (per step)</h2>")
    why.append(
        "<p><a href='feature_group_search_report.html'>← back to main report</a></p>"
    )

    baseline = result.get("baseline") or {}
    why.append("<h3>Baseline</h3>")
    why.append(f"<p><b>baseline_score</b>: {baseline.get('score')}</p>")
    bsum = baseline.get("summary") or {}
    if isinstance(bsum, dict) and bsum:
        keep = [
            "Sharpe_mean",
            "return%_mean",
            "DD%_mean",
            "trades_mean",
            "CV_mean",
            "corr_mean",
        ]
        why.append(
            pd.DataFrame([{k: bsum.get(k) for k in keep if k in bsum}]).to_html(
                index=False, escape=False
            )
        )

    hist_by_step = {h.get("step"): h for h in (result.get("history") or [])}
    cand_hist = result.get("candidates_history") or []
    for step_obj in cand_hist:
        step_n = step_obj.get("step")
        chosen = (hist_by_step.get(step_n) or {}).get("added_group")
        why.append(f"<h3 id='step_{step_n}'>Step {step_n}</h3>")
        why.append(
            f"<p><b>chosen</b>: {chosen or '(none)'} &nbsp; <b>stop_reason</b>: {result.get('stop_reason') if chosen is None else ''}</p>"
        )
        cands = step_obj.get("candidates") or []
        rows = []
        for c in cands:
            summ = c.get("summary") or {}
            rows.append(
                {
                    "step": step_n,
                    "candidate_group": c.get("candidate_group"),
                    "valid": c.get("valid"),
                    "reject_reason": c.get("reject_reason"),
                    "score": c.get("score"),
                    "trades_mean": summ.get("trades_mean"),
                    "Sharpe_mean": summ.get("Sharpe_mean"),
                    "return%_mean": summ.get("return%_mean"),
                    "DD%_mean": summ.get("DD%_mean"),
                    "corr_mean": summ.get("corr_mean"),
                    "CV_mean": summ.get("CV_mean"),
                    "picked": c.get("candidate_group") == chosen,
                }
            )
        if rows:
            df = pd.DataFrame(rows)
            # Sort: picked first, then valid desc by score, then invalid last
            if "score" in df.columns:
                df["_score_sort"] = pd.to_numeric(df["score"], errors="coerce")
                df = df.sort_values(
                    by=["picked", "valid", "_score_sort"],
                    ascending=[False, False, False],
                ).drop(columns=["_score_sort"])
            why.append(df.to_html(index=False, escape=False))

    why.append("</body></html>")
    (out_dir / "feature_group_search_why.html").write_text(
        "\n".join(why), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
