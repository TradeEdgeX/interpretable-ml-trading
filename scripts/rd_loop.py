#!/usr/bin/env python3
"""R&D loop driver: mlbot research scan → variant_grid → decision doc skeleton.

Reads a hypothesis YAML (see ``config/experiments/rd_loop_example.yaml``) and runs
steps in order, persisting progress to ``<output_dir>/rd_loop_state.json`` so a
failed run can resume with ``--resume``.

Does not modify live yaml.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_hypothesis(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("hypothesis yaml must be a mapping")
    return data


def _state_path(output_dir: Path) -> Path:
    return output_dir / "rd_loop_state.json"


def _load_state(output_dir: Path) -> Dict[str, Any]:
    p = _state_path(output_dir)
    if not p.exists():
        return {"completed_steps": [], "steps": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def _save_state(output_dir: Path, state: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _state_path(output_dir).write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _run_cmd(cmd: List[str], *, cwd: Path) -> int:
    print("\n>>>", " ".join(cmd))
    return int(subprocess.run(cmd, cwd=str(cwd)).returncode)


def _mlbot_cmd() -> List[str]:
    exe = shutil.which("mlbot")
    if exe:
        return [exe]
    return [sys.executable, "-m", "cli.main"]


def _append_common_scan_args(
    cmd: List[str], scan: Dict[str, Any], cfg: Dict[str, Any]
) -> None:
    cmd += ["--features-parquet", str(scan["features_parquet"])]
    strategy = scan.get("strategy") or cfg.get("strategy")
    if strategy:
        cmd += ["--strategy", str(strategy)]
    if scan.get("layer"):
        cmd += ["--layer", str(scan["layer"])]
    if scan.get("label"):
        cmd += ["--label", str(scan["label"])]
    for f in scan.get("filter") or []:
        cmd += ["--filter", str(f)]
    if scan.get("subset"):
        cmd += ["--subset", str(scan["subset"])]
    if scan.get("calendar_window"):
        cmd += ["--calendar-window", str(scan["calendar_window"])]
    if scan.get("subject"):
        cmd += ["--subject", str(scan["subject"])]


def _resolve_scan_parquet(scan: Dict[str, Any]) -> Path:
    pq = Path(str(scan["features_parquet"]))
    if not pq.is_absolute():
        pq = (PROJECT_ROOT / pq).resolve()
    return pq


def _resolve_scan_out_dir(
    scan: Dict[str, Any], output_dir: Path, *, default_name: str
) -> Path:
    out_rel = scan.get("out") or default_name
    out = Path(out_rel)
    if not out.is_absolute():
        out = (output_dir / out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _run_entry_plateau_scan(
    scan: Dict[str, Any], output_dir: Path, cfg: Dict[str, Any]
) -> int:
    from scripts.research.entry_plateau_scan import run_entry_plateau_batch

    strategy = str(scan.get("strategy") or cfg.get("strategy") or "")
    if not strategy:
        print("ERROR: entry-plateau requires strategy", file=sys.stderr)
        return 3
    pq = _resolve_scan_parquet(scan)
    if not pq.is_file():
        print(f"ERROR: features parquet not found: {pq}", file=sys.stderr)
        return 3
    out = _resolve_scan_out_dir(
        scan, output_dir, default_name="quick_scan/entry_plateau"
    )
    filter_id = scan.get("entry_filter")
    if filter_id is None and isinstance(scan.get("filter"), str):
        filter_id = scan.get("filter")
    snotio_mode = str(scan.get("snotio_mode", "entry_rr"))
    steps = int(scan.get("steps", 15))
    min_trades = int(scan.get("min_trades", 20))
    print(f"\n>>> entry-plateau batch strategy={strategy} parquet={pq} out={out}")
    try:
        run_entry_plateau_batch(
            pq,
            strategy,
            filter_id=str(filter_id) if filter_id else None,
            snotio_mode=snotio_mode,
            steps=steps,
            min_trades=min_trades,
            plateau_window=int(scan.get("plateau_window", 4)),
            research=bool(scan.get("research", False)),
            simple_execution=bool(scan.get("simple_execution", False)),
            require_gate=False,
            out_dir=out,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    return 0


def _run_gate_plateau_scan(
    scan: Dict[str, Any], output_dir: Path, cfg: Dict[str, Any]
) -> int:
    from scripts.research.gate_plateau_scan import run_gate_plateau_batch

    strategy = str(scan.get("strategy") or cfg.get("strategy") or "")
    if not strategy:
        print("ERROR: gate-plateau requires strategy", file=sys.stderr)
        return 3
    pq = _resolve_scan_parquet(scan)
    if not pq.is_file():
        print(f"ERROR: features parquet not found: {pq}", file=sys.stderr)
        return 3
    out = _resolve_scan_out_dir(
        scan, output_dir, default_name="quick_scan/gate_plateau"
    )
    print(f"\n>>> gate-plateau batch strategy={strategy} parquet={pq} out={out}")
    try:
        run_gate_plateau_batch(
            pq,
            strategy,
            out_dir=out,
            label_col=str(scan.get("label_col", "is_good")),
            step=float(scan.get("step", 0.05)),
            rule_id=str(scan["rule_id"]) if scan.get("rule_id") else None,
            gate_path=str(scan["gate_path"]) if scan.get("gate_path") else None,
            write_back_intervals=bool(scan.get("write_back_intervals", False)),
            min_lift=float(scan.get("min_lift", 0.10)),
            skip_locked=bool(scan.get("skip_locked", True)),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    return 0


def _run_locked_prefilter_tune(
    scan: Dict[str, Any], output_dir: Path, cfg: Dict[str, Any]
) -> int:
    from scripts.locked_prefilter_parquet_tune import (
        suggest_locked_prefilter_params_parquet,
    )

    strategy = str(scan.get("strategy") or cfg.get("strategy") or "")
    if not strategy:
        print("ERROR: locked-prefilter-tune requires strategy", file=sys.stderr)
        return 3
    pq = _resolve_scan_parquet(scan)
    if not pq.is_file():
        print(f"ERROR: features parquet not found: {pq}", file=sys.stderr)
        return 3
    prefilter_rel = (
        scan.get("prefilter_path")
        or f"config/strategies/{strategy}/archetypes/prefilter.yaml"
    )
    prefilter_path = Path(str(prefilter_rel))
    if not prefilter_path.is_absolute():
        prefilter_path = (PROJECT_ROOT / prefilter_path).resolve()
    out = _resolve_scan_out_dir(
        scan, output_dir, default_name="quick_scan/locked_prefilter_tune"
    )
    tcfg = scan.get("locked_threshold_tuning") or {}
    prefilter_gates = scan.get("prefilter_gates") or {}
    print(f"\n>>> locked-prefilter-tune strategy={strategy} parquet={pq} out={out}")
    try:
        params, meta = suggest_locked_prefilter_params_parquet(
            prod_prefilter_path=prefilter_path,
            labeled_parquet_path=pq,
            template=str(scan.get("template", "bindings")),
            tcfg=tcfg,
            prefilter_gates=prefilter_gates,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    payload = {
        "strategy": strategy,
        "params": params,
        "meta": meta,
        "kpi": "prefilter_bindings",
    }
    out_json = out / "locked_prefilter_proposal.json"
    out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_json}")
    return 0


def _build_research_scan_cmd(
    scan: Dict[str, Any], output_dir: Path, cfg: Dict[str, Any]
) -> List[str]:
    mode = str(scan.get("mode", "condition-set"))
    out_rel = scan.get("out") or f"quick_scan/scan_{mode.replace('-', '_')}.md"
    out = Path(out_rel)
    if not out.is_absolute():
        out = (output_dir / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    bucket_by = scan.get("bucket_by")
    if bucket_by:
        cmd = _mlbot_cmd() + [
            "research",
            "segment",
            "--bucket-by",
            str(bucket_by),
            "--mode",
            mode if mode in ("condition-set", "feature-plateau") else "condition-set",
            "--output",
            str(out),
        ]
        _append_common_scan_args(cmd, scan, cfg)
        if mode == "feature-plateau":
            cmd += [
                "--feature",
                str(scan["feature"]),
                "--operator",
                str(scan.get("operator", "<=")),
                "--grid",
                str(scan["grid"]),
            ]
        else:
            for c in scan.get("condition") or []:
                cmd += ["--condition", str(c)]
        return cmd

    if mode == "ic-decay":
        cmd = _mlbot_cmd() + ["research", "ic", "--output", str(out)]
        _append_common_scan_args(cmd, scan, cfg)
        cmd += [
            "--features",
            str(scan["features"]),
            "--horizons",
            str(scan.get("horizons", "1,3,5,10,20")),
        ]
        if scan.get("target"):
            cmd += ["--target", str(scan["target"])]
        if scan.get("baseline_json"):
            cmd += ["--baseline-json", str(scan["baseline_json"])]
        return cmd

    if mode == "snotio-plateau":
        snotio_mode = str(scan.get("snotio_mode", "proxy"))
        cmd = _mlbot_cmd() + [
            "research",
            "plateau",
            "--kpi",
            "snotio",
            "--snotio-mode",
            snotio_mode,
            "--output",
            str(out.with_suffix(".json") if out.suffix == ".md" else out),
        ]
        _append_common_scan_args(cmd, scan, cfg)
        cmd += [
            "--feature",
            str(scan["feature"]),
            "--operator",
            str(scan.get("operator", "<=")),
            "--grid",
            str(scan["grid"]),
        ]
        if scan.get("r_col"):
            cmd += ["--r-col", str(scan["r_col"])]
        return cmd

    if mode == "gate-plateau":
        cmd = _mlbot_cmd() + [
            "research",
            "plateau",
            "--kpi",
            "lift",
            "--layer",
            "gate",
            "--output",
            str(out.with_suffix(".json") if out.suffix == ".md" else out),
        ]
        _append_common_scan_args(cmd, scan, cfg)
        if scan.get("feature"):
            cmd += [
                "--feature",
                str(scan["feature"]),
                "--operator",
                str(scan.get("operator", ">")),
                "--grid",
                str(scan.get("grid", "0,1,0.05")),
            ]
        if scan.get("write_back_intervals"):
            cmd += ["--write-back-intervals"]
        if scan.get("min_lift") is not None:
            cmd += ["--min-lift", str(scan["min_lift"])]
        return cmd

    verb = "research"
    cmd = _mlbot_cmd() + [verb, "scan", mode, "--output", str(out)]
    _append_common_scan_args(cmd, scan, cfg)

    if mode == "feature-plateau":
        cmd += [
            "--feature",
            str(scan["feature"]),
            "--operator",
            str(scan.get("operator", "<=")),
            "--grid",
            str(scan["grid"]),
        ]
    elif mode == "feature-threshold-mean":
        cmd += [
            "--feature",
            str(scan["feature"]),
            "--operator",
            str(scan.get("operator", "<=")),
            "--grid",
            str(scan["grid"]),
            "--target",
            str(scan.get("target", "forward_rr")),
        ]
    elif mode == "condition-set":
        for c in scan.get("condition") or []:
            cmd += ["--condition", str(c)]
    elif mode == "pair-scan":
        cmd += ["--pair-a", str(scan["pair_a"]), "--pair-b", str(scan["pair_b"])]
    return cmd


def _resolve_quick_scan_html_cfg(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return html-report options, or None to skip the step."""
    raw = cfg.get("quick_scan_html")
    scans = cfg.get("quick_layer_scans") or cfg.get("research_scans") or []
    if raw is False:
        return None
    if isinstance(raw, dict):
        if raw.get("enabled") is False:
            return None
        return dict(raw)
    if raw is True or raw is None:
        if not scans:
            return None
        return {}
    return None


def _resolve_bundle_parquet(
    cfg: Dict[str, Any],
    mb: Dict[str, Any],
    output_dir: Path,
    state: Dict[str, Any],
) -> Path:
    raw = mb.get("parquet") or cfg.get("features_parquet")
    if raw is None:
        raise ValueError(
            "monitor_bundle requires parquet or top-level features_parquet"
        )
    if str(raw).strip().lower() == "auto":
        steps = state.get("tree_pipeline_steps") or {}
        last_prepare: Optional[Dict[str, Any]] = None
        for _key, meta in sorted(steps.items()):
            if not isinstance(meta, dict):
                continue
            if meta.get("mode") == "prepare-only" and meta.get("exit_code") == 0:
                last_prepare = meta
        if last_prepare and last_prepare.get("output_root"):
            pq = Path(str(last_prepare["output_root"])) / "features_labeled.parquet"
            return _resolve_project_path(pq)
        raise ValueError(
            "monitor_bundle parquet: auto requires a successful tree_steps prepare-only "
            "with output_root recorded in rd_loop_state.json"
        )
    pq = _resolve_project_path(str(raw), relative_to=output_dir)
    if not pq.is_file():
        pq = _resolve_project_path(str(raw))
    return pq


def _step_monitor_bundle(
    cfg: Dict[str, Any],
    output_dir: Path,
    state: Dict[str, Any],
    *,
    hypothesis_path: Optional[Path] = None,
) -> int:
    mb = cfg.get("monitor_bundle")
    if not mb or not isinstance(mb, dict):
        return 0
    mode = str(mb.get("mode") or "draft").lower()
    if mode == "promote":
        print(
            "ERROR: monitor_bundle.mode=promote is not supported in rd_loop.\n"
            "  Use: mlbot research promote-baseline --experiment-dir <dir>",
            file=sys.stderr,
        )
        return 2
    if mode != "draft":
        print(f"ERROR: unknown monitor_bundle.mode: {mode}", file=sys.stderr)
        return 2

    from scripts.monitoring.export_monitor_bundle import export_monitor_bundle

    strategy = str(mb.get("strategy") or cfg.get("strategy") or "")
    layer = str(mb.get("layer") or "regime")
    if not strategy:
        print("ERROR: monitor_bundle requires strategy", file=sys.stderr)
        return 3

    try:
        pq = _resolve_bundle_parquet(cfg, mb, output_dir, state)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    if not pq.is_file():
        print(f"ERROR: features parquet not found: {pq}", file=sys.stderr)
        return 3

    exp_dir = hypothesis_path.parent if hypothesis_path else output_dir
    out_rel = str(mb.get("out_dir") or "monitor_bundle")
    out_dir = _resolve_project_path(out_rel, relative_to=exp_dir)

    cal = mb.get("calibration") if isinstance(mb.get("calibration"), dict) else {}
    try:
        result = export_monitor_bundle(
            strategy=strategy,
            layer=layer,
            parquet=pq,
            out_dir=out_dir,
            calibration=cal,
            run_smoke=mb.get("run_smoke", True) is not False,
        )
    except Exception as e:
        print(f"ERROR: monitor_bundle export failed: {e}", file=sys.stderr)
        return 3

    state.setdefault("monitor_bundle", {})
    state["monitor_bundle"] = {
        "bundle_path": str(result["bundle_path"]),
        "out_dir": str(out_dir),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    print(f"monitor_bundle draft → {result['bundle_path']}")
    smoke = result.get("smoke_report") or {}
    if smoke.get("drift_exit", 0) != 0 or smoke.get("watchdog_exit", 0) != 0:
        print(
            f"WARNING: smoke drift={smoke.get('drift_exit')} watchdog={smoke.get('watchdog_exit')}",
            file=sys.stderr,
        )
    return 0


def _step_quick_scan_html(cfg: Dict[str, Any], output_dir: Path) -> int:
    html_cfg = _resolve_quick_scan_html_cfg(cfg)
    if html_cfg is None:
        return 0
    scan_rel = str(html_cfg.get("scan_dir") or "quick_scan")
    scan_dir = Path(scan_rel)
    if not scan_dir.is_absolute():
        scan_dir = (output_dir / scan_dir).resolve()
    if not scan_dir.is_dir():
        print(f"skip quick_scan_html: missing {scan_dir}")
        return 0
    artifacts = [
        p
        for p in scan_dir.iterdir()
        if p.suffix in (".md", ".json") and p.is_file() and p.name != "report.html"
    ]
    if not artifacts:
        print(f"skip quick_scan_html: no .md/.json under {scan_dir}")
        return 0

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.research.quick_scan_html import (
        build_report_from_config,
        load_manifest,
        manifest_section_meta,
        resolve_report_config,
        section_order_from_scans,
        write_manifest,
    )

    out_rel = html_cfg.get("out") or "quick_scan/report.html"
    out_path = Path(str(out_rel))
    if not out_path.is_absolute():
        out_path = (output_dir / out_path).resolve()
    report_cfg = resolve_report_config(
        hypothesis=cfg, html_block=html_cfg, scan_dir=scan_dir
    )
    scans = cfg.get("quick_layer_scans") or cfg.get("research_scans") or []
    order = report_cfg.section_order or section_order_from_scans(scans)
    if order:
        write_manifest(
            scan_dir,
            section_order=order,
            scans=scans,
            hypothesis_topic=cfg.get("topic"),
        )
        report_cfg.section_order = order
    blob = load_manifest(scan_dir)
    meta = manifest_section_meta(blob) if blob else {}
    print(f"\n>>> quick_scan_html scan_dir={scan_dir} out={out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        build_report_from_config(scan_dir, report_cfg, section_meta=meta),
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    return 0


def _step_research_scan(cfg: Dict[str, Any], output_dir: Path) -> int:
    scans = cfg.get("quick_layer_scans") or cfg.get("research_scans") or []
    if not isinstance(scans, list):
        raise ValueError("quick_layer_scans / research_scans must be a list")
    worst = 0
    for i, scan in enumerate(scans):
        if not isinstance(scan, dict):
            continue
        mode = str(scan.get("mode", "condition-set"))
        if mode == "entry-plateau":
            rc = _run_entry_plateau_scan(scan, output_dir, cfg)
        elif mode == "gate-plateau":
            if scan.get("feature"):
                cmd = _build_research_scan_cmd(scan, output_dir, cfg)
                rc = _run_cmd(cmd, cwd=PROJECT_ROOT)
            else:
                rc = _run_gate_plateau_scan(scan, output_dir, cfg)
        elif mode == "locked-prefilter-tune":
            rc = _run_locked_prefilter_tune(scan, output_dir, cfg)
        else:
            cmd = _build_research_scan_cmd(scan, output_dir, cfg)
            rc = _run_cmd(cmd, cwd=PROJECT_ROOT)
        worst = max(worst, rc)
    return worst


def _resolve_project_path(
    value: str | Path,
    *,
    relative_to: Path | None = None,
) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    base = relative_to or PROJECT_ROOT
    return (base / p).resolve()


def _tree_step_enabled(step: Dict[str, Any]) -> bool:
    return step.get("enabled", True) is not False


def _merge_tree_defaults(step: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(step)
    holdout = cfg.get("holdout") or {}
    if holdout:
        merged.setdefault("holdout_start", holdout.get("start"))
        merged.setdefault("holdout_end", holdout.get("end"))
        merged.setdefault("holdout_start_date", holdout.get("start"))
        merged.setdefault("holdout_end_date", holdout.get("end"))
    if merged.get("start") and not merged.get("holdout_start"):
        merged["holdout_start"] = merged["start"]
    if merged.get("end") and not merged.get("holdout_end"):
        merged["holdout_end"] = merged["end"]
    merged.setdefault("feature_store_layer", cfg.get("feature_store_layer"))
    merged.setdefault("timeframe", cfg.get("timeframe", "120T"))
    merged.setdefault("data_path", cfg.get("data_path", "data/parquet_data"))
    if not merged.get("config") and cfg.get("strategy"):
        merged.setdefault(
            "config", f"config/strategies/tree_strategies/{cfg['strategy']}"
        )
    return merged


def _build_tree_train_cmd(step: Dict[str, Any], cfg: Dict[str, Any]) -> List[str]:
    step = _merge_tree_defaults(step, cfg)
    config = str(step["config"])
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_strategy_pipeline.py"),
        "--config",
        config,
        "--data-path",
        str(step.get("data_path", "data/parquet_data")),
        "--symbol",
        str(step["symbols"]),
        "--timeframe",
        str(step.get("timeframe", "120T")),
        "--seed",
        str(step.get("seed", 42)),
        "--output-root",
        str(_resolve_project_path(step["output_root"])),
        "--start-date",
        str(step["start_date"]),
        "--end-date",
        str(step["end_date"]),
        "--feature-store-dir",
        str(step.get("feature_store_dir", "feature_store")),
    ]
    if step.get("holdout_start_date"):
        cmd += ["--holdout-start-date", str(step["holdout_start_date"])]
    if step.get("holdout_end_date"):
        cmd += ["--holdout-end-date", str(step["holdout_end_date"])]
    if step.get("feature_store_layer"):
        cmd += ["--feature-store-layer", str(step["feature_store_layer"])]
    if step.get("features"):
        cmd += ["--features", str(_resolve_project_path(step["features"]))]
    if step.get("prepare_only"):
        cmd.append("--prepare-only")
    if step.get("labels"):
        labels = step["labels"]
        cmd += ["--labels", str(_resolve_project_path(labels))]
    if step.get("deterministic", True):
        cmd.append("--deterministic")
    return cmd


def _build_tree_ic_prune_cmd(
    step: Dict[str, Any], output_dir: Path, cfg: Dict[str, Any]
) -> List[str]:
    from src.research.stat_kernels.ic_screen_config import (
        ic_prune_params_to_argv,
        resolve_ic_prune_params,
    )

    step = _merge_tree_defaults(step, cfg)
    out_rel = step.get("out") or "ic_prune"
    out_dir = _resolve_project_path(out_rel, relative_to=output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq = _resolve_project_path(step["features_parquet"], relative_to=output_dir)
    if not pq.is_file():
        pq = _resolve_project_path(step["features_parquet"])

    strategy = step.get("strategy") or cfg.get("strategy")
    config_dir = step.get("config")
    params = resolve_ic_prune_params(
        strategy=strategy,
        config_dir=config_dir,
        overrides=step,
        project_root=PROJECT_ROOT,
    )
    summary = params.pop("_ic_screen_summary", None)
    params.pop("_strategy_config_dir", None)
    if summary:
        print(f"ic_screen: {summary}")

    for path_key in (
        "write_features_yaml",
        "write_model_features_yaml",
        "intersect_features_yaml",
    ):
        val = params.get(path_key)
        if val and val is not False:
            params[path_key] = str(_resolve_project_path(val))
    if step.get("emit_monotone_constraints"):
        params["emit_monotone_constraints"] = str(
            _resolve_project_path(
                step["emit_monotone_constraints"], relative_to=output_dir
            )
        )

    cmd = _mlbot_cmd() + [
        "research",
        "ic-prune",
        "--features-parquet",
        str(pq),
        "--out-dir",
        str(out_dir),
    ]
    if strategy:
        cmd += ["--strategy", str(strategy)]
    if config_dir:
        cmd += ["--config-dir", str(_resolve_project_path(config_dir))]
    cmd += ic_prune_params_to_argv(params)
    return cmd


def _run_filter_predictions_step(
    step: Dict[str, Any], output_dir: Path, cfg: Dict[str, Any]
) -> int:
    import pandas as pd

    step = _merge_tree_defaults(step, cfg)
    preds = _resolve_project_path(step["predictions"])
    if not preds.is_file():
        print(f"ERROR: predictions not found: {preds}", file=sys.stderr)
        return 3
    out = _resolve_project_path(
        step.get("out", "filtered/predictions.parquet"), relative_to=output_dir
    )
    df = pd.read_parquet(preds)
    split = step.get("split")
    if split and "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == str(split).lower()].copy()
    symbols = step.get("symbols")
    if symbols:
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(",") if s.strip()]
        sym_col = "_symbol" if "_symbol" in df.columns else "symbol"
        if sym_col not in df.columns:
            print(f"ERROR: no symbol column in {preds}", file=sys.stderr)
            return 3
        df = df[df[sym_col].isin(symbols)].copy()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out} rows={len(df)}")
    return 0


def _run_tree_step(step: Dict[str, Any], output_dir: Path, cfg: Dict[str, Any]) -> int:
    if not _tree_step_enabled(step):
        return 0
    mode = str(step.get("mode", "")).lower()
    step = _merge_tree_defaults(step, cfg)
    print(f"\n>>> tree step mode={mode}")

    if mode in ("prepare-only", "train"):
        if mode == "prepare-only":
            step = {**step, "prepare_only": True}
        for key in ("symbols", "start_date", "end_date", "output_root", "config"):
            if not step.get(key):
                print(f"ERROR: tree {mode} missing {key}", file=sys.stderr)
                return 3
        cmd = _build_tree_train_cmd(step, cfg)
        return _run_cmd(cmd, cwd=PROJECT_ROOT)

    if mode == "ic-prune":
        if not step.get("features_parquet"):
            print("ERROR: ic-prune requires features_parquet", file=sys.stderr)
            return 3
        cmd = _build_tree_ic_prune_cmd(step, output_dir, cfg)
        return _run_cmd(cmd, cwd=PROJECT_ROOT)

    if mode == "tau-scan":
        from scripts.research.tree_holdout_tau_rr_scan import run_tau_scan

        if not step.get("config"):
            print("ERROR: tau-scan requires config", file=sys.stderr)
            return 3
        if not step.get("predictions") and not step.get("artifact_dir"):
            print(
                "ERROR: tau-scan requires predictions or artifact_dir", file=sys.stderr
            )
            return 3
        out_rel = step.get("out") or "tau_scan"
        out_dir = _resolve_project_path(out_rel, relative_to=output_dir)
        predictions = step.get("predictions")
        if predictions:
            pred_path = _resolve_project_path(predictions, relative_to=output_dir)
            if not pred_path.is_file():
                pred_path = _resolve_project_path(predictions)
            predictions = str(pred_path)
        try:
            run_tau_scan(
                config=step["config"],
                output_dir=out_dir,
                predictions=predictions,
                artifact_dir=step.get("artifact_dir"),
                start_date=step.get("start_date"),
                end_date=step.get("end_date"),
                symbols=step.get("symbols", "BTCUSDT,ETHUSDT"),
                timeframe=str(step.get("timeframe", "120T")),
                data_path=str(step.get("data_path", "data/parquet_data")),
                feature_store_layer=step.get("feature_store_layer"),
                fixed_quantile=step.get("fixed_quantile"),
                long_entry_threshold=step.get("long_entry_threshold"),
                short_entry_threshold=step.get("short_entry_threshold"),
                segment_label=str(step.get("segment_label", "holdout")),
                quantile_grid=str(
                    step.get("quantile_grid", "0.05,0.08,0.10,0.12,0.15,0.20,0.25,0.30")
                ),
                pred_grid=step.get("pred_grid"),
                per_symbol=bool(step.get("per_symbol", True)),
                filter_split=step.get("filter_split", "holdout"),
                regime_gate=step.get("regime_gate"),
            )
        except (ValueError, KeyError, OSError) as exc:
            print(f"ERROR: tau-scan failed: {exc}", file=sys.stderr)
            return 3
        return 0

    if mode == "filter-predictions":
        return _run_filter_predictions_step(step, output_dir, cfg)

    if mode == "shell":
        cmd = step.get("cmd") or step.get("command")
        if not cmd:
            print("ERROR: shell step requires cmd", file=sys.stderr)
            return 3
        return _run_cmd(["bash", "-lc", str(cmd)], cwd=PROJECT_ROOT)

    print(f"ERROR: unknown tree step mode: {mode}", file=sys.stderr)
    return 3


def _step_tree_pipeline(
    cfg: Dict[str, Any], output_dir: Path, state: Dict[str, Any]
) -> int:
    steps = cfg.get("tree_steps") or []
    if not isinstance(steps, list) or not steps:
        return 0
    completed = set(state.get("tree_pipeline_completed") or [])
    worst = 0
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        if i in completed:
            print(f"skip tree step {i}: {step.get('mode', '?')}")
            continue
        print(f"\n=== tree_pipeline [{i}] mode={step.get('mode')} ===")
        rc = _run_tree_step(step, output_dir, cfg)
        state.setdefault("tree_pipeline_completed", [])
        state["tree_pipeline_steps"] = state.get("tree_pipeline_steps") or {}
        state["tree_pipeline_steps"][str(i)] = {
            "mode": step.get("mode"),
            "exit_code": rc,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        if step.get("mode") == "prepare-only" and step.get("output_root"):
            state["tree_pipeline_steps"][str(i)]["output_root"] = str(
                _resolve_project_path(step["output_root"])
            )
        _save_state(output_dir, state)
        worst = max(worst, rc)
        if rc != 0:
            return rc
        completed.add(i)
        state["tree_pipeline_completed"] = sorted(completed)
        _save_state(output_dir, state)
    return worst


def _step_variant_grid(cfg: Dict[str, Any], output_dir: Path) -> int:
    grid = cfg.get("variant_grid")
    if not grid:
        return 0
    grid_path = Path(str(grid))
    if not grid_path.is_absolute():
        grid_path = (PROJECT_ROOT / grid_path).resolve()
    cmd = [
        sys.executable,
        "-m",
        "scripts.event_backtest",
        "--variant-grid",
        str(grid_path),
    ]
    extra = cfg.get("variant_grid_extra_argv") or []
    if isinstance(extra, list):
        cmd += [str(x) for x in extra]
    return _run_cmd(cmd, cwd=PROJECT_ROOT)


def _step_decision_doc(cfg: Dict[str, Any], output_dir: Path) -> int:
    dec = cfg.get("decision_doc") or {}
    if not dec:
        return 0
    index = dec.get("experiment_index")
    if not index:
        strategy = str(cfg.get("strategy") or "tpc")
        index = (
            PROJECT_ROOT
            / "results"
            / strategy
            / "experiments"
            / "EXPERIMENT_INDEX.json"
        )
    index_path = Path(str(index))
    if not index_path.is_absolute():
        index_path = (PROJECT_ROOT / index_path).resolve()
    topic = str(dec.get("topic") or cfg.get("topic") or "rd_loop")
    out = dec.get("out")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "_new_decision_doc.py"),
        "--experiment-index",
        str(index_path),
        "--topic",
        topic,
    ]
    if dec.get("topic_template"):
        cmd += ["--topic-template", str(dec["topic_template"])]
    if out:
        out_path = Path(str(out))
        if not out_path.is_absolute():
            out_path = (output_dir / out_path).resolve()
        cmd += ["--out", str(out_path)]
    return _run_cmd(cmd, cwd=PROJECT_ROOT)


def run_loop(
    hypothesis_path: Path,
    *,
    output_dir: Optional[Path] = None,
    resume: bool = False,
) -> int:
    cfg = _load_hypothesis(hypothesis_path)
    topic = str(cfg.get("topic") or hypothesis_path.stem)
    out = output_dir or Path(str(cfg.get("output_dir") or f"results/rd_loop/{topic}"))
    if not out.is_absolute():
        out = (PROJECT_ROOT / out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    state = _load_state(out) if resume else {"completed_steps": [], "steps": {}}
    completed = set(state.get("completed_steps") or [])

    steps = [
        ("research_scan", lambda: _step_research_scan(cfg, out)),
        (
            "monitor_bundle",
            lambda: _step_monitor_bundle(
                cfg, out, state, hypothesis_path=hypothesis_path
            ),
        ),
        ("quick_scan_html", lambda: _step_quick_scan_html(cfg, out)),
        ("tree_pipeline", lambda: _step_tree_pipeline(cfg, out, state)),
        ("variant_grid", lambda: _step_variant_grid(cfg, out)),
        ("decision_doc", lambda: _step_decision_doc(cfg, out)),
    ]

    worst = 0
    for name, fn in steps:
        if name in completed:
            print(f"skip (already done): {name}")
            continue
        print(f"\n=== rd_loop step: {name} ===")
        rc = fn()
        state["steps"][name] = {
            "exit_code": rc,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        if rc == 0:
            completed.add(name)
            state["completed_steps"] = sorted(completed)
        _save_state(out, state)
        worst = max(worst, rc)
        if rc != 0:
            print(f"rd_loop stopped at step {name} (exit {rc})", file=sys.stderr)
            return rc
    print(f"\nrd_loop complete → {out}")
    return worst


def main() -> int:
    p = argparse.ArgumentParser(
        description="R&D loop driver (mlbot research scan → grid → decision doc)"
    )
    p.add_argument(
        "--hypothesis-yaml",
        required=True,
        help="Hypothesis config (quick_layer_scans / research_scans + variant_grid).",
    )
    p.add_argument("--output-dir", default=None)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip steps already marked complete in rd_loop_state.json.",
    )
    args = p.parse_args()

    hyp = Path(args.hypothesis_yaml)
    if not hyp.is_absolute():
        hyp = (PROJECT_ROOT / hyp).resolve()
    if not hyp.exists():
        print(f"ERROR: not found: {hyp}", file=sys.stderr)
        return 3
    out = Path(args.output_dir) if args.output_dir else None
    if out and not out.is_absolute():
        out = (PROJECT_ROOT / out).resolve()
    return run_loop(hyp, output_dir=out, resume=args.resume)


if __name__ == "__main__":
    sys.exit(main())
