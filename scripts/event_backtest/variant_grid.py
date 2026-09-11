"""Run multiple event_backtest variants from a YAML grid; update EXPERIMENT_INDEX.

Engine dispatcher
-----------------

Each ``run:`` entry (or the grid-level fields) may specify ``engine``:

- ``event_backtest`` (default): drives ``scripts.event_backtest`` — used for
  B-system strategies (BPC/TPC/ME/SRB).
- ``chop_grid``: drives ``scripts.chop_grid_backtest`` — used for C-system
  chop grid R&D.
- ``trend_scalp``: drives ``scripts.diagnose_dual_add_trend`` — used for C-system
  trend scalp R&D.
- ``multileg_joint``: runs ``scripts.backtest_multileg_timeline`` per canonical
  segment (LiveEngine chop + trend, shared account).

All engines write per-run output under the configured ``output_dir``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

from scripts.event_backtest.market_segment import (
    expand_segment_matrix,
    load_market_segments,
    resolve_segment_run,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

_ENGINE_EVENT = "event_backtest"
_ENGINE_CHOP_GRID = "chop_grid"
_ENGINE_TREND_SCALP = "trend_scalp"
_ENGINE_MULTILEG_JOINT = "multileg_joint"
_VALID_ENGINES = (
    _ENGINE_EVENT,
    _ENGINE_CHOP_GRID,
    _ENGINE_TREND_SCALP,
    _ENGINE_MULTILEG_JOINT,
)


def _load_grid(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("variant grid must be a mapping")
    return data


def _resolve_engine(run: Dict[str, Any], grid: Dict[str, Any]) -> str:
    eng = str(run.get("engine") or grid.get("engine") or _ENGINE_EVENT)
    if eng not in _VALID_ENGINES:
        raise ValueError(f"unknown engine {eng!r}; choose from {_VALID_ENGINES}")
    return eng


def _build_event_backtest_cmd(
    *,
    run: Dict[str, Any],
    grid: Dict[str, Any],
    out_path: Path,
    extra_argv: List[str],
) -> List[str]:
    strategy = str(run.get("strategy") or grid.get("strategy") or "tpc")
    strategies_root = str(
        run.get("strategies_root") or grid.get("strategies_root") or "config/strategies"
    )
    trades_csv = out_path / f"event_trades_{strategy}.csv"
    symbols = run.get("symbols") or grid.get("symbols")
    sym_arg = (
        ",".join(symbols)
        if isinstance(symbols, list)
        else "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,HYPEUSDT"
    )
    cmd = [
        sys.executable,
        "-m",
        "scripts.event_backtest",
        "--strategy",
        strategy,
        "--symbols",
        sym_arg,
        "--start-date",
        str(run["start_date"]),
        "--end-date",
        str(run["end_date"]),
        "--strategies-root",
        strategies_root,
        "--trades-csv",
        str(trades_csv),
        "--capital-report",
        str(out_path),
        "--no-kill-switch",
    ]
    data_path = run.get("data_path") or grid.get("data_path")
    if data_path:
        cmd += ["--data-path", str(data_path)]
    if run.get("trading_map") or grid.get("trading_map"):
        map_out = out_path / f"trading_map_{strategy}_event.html"
        cmd += ["--trading-map", str(map_out)]
    if run.get("fast") or grid.get("fast"):
        cmd += ["--fast"]
    constitution_yaml = run.get("constitution_yaml") or grid.get("constitution_yaml")
    if constitution_yaml:
        const_path = Path(str(constitution_yaml))
        if not const_path.is_absolute():
            const_path = (_REPO_ROOT / const_path).resolve()
        cmd += ["--constitution-yaml", str(const_path)]
    seg_id = run.get("segment")
    if seg_id:
        cmd += ["--seg-id", str(seg_id)]
    seg_path = run.get("market_segment_path") or grid.get("market_segment_path")
    if seg_path:
        cmd += ["--market-segment-path", str(seg_path)]
    inject_scores = run.get("inject_scores") or grid.get("inject_scores")
    if inject_scores:
        inj_path = Path(str(inject_scores))
        if not inj_path.is_absolute():
            inj_path = (_REPO_ROOT / inj_path).resolve()
        cmd += ["--inject-add-ml-scores", str(inj_path)]
    for src in (grid.get("extra_argv"), run.get("extra_argv"), extra_argv):
        if src:
            cmd += [str(x) for x in src]
    return cmd


def _build_chop_grid_cmd(
    *,
    run: Dict[str, Any],
    grid: Dict[str, Any],
    out_path: Path,
    extra_argv: List[str],
) -> List[str]:
    config = run.get("config") or grid.get("config")
    if not config:
        raise ValueError(
            "chop_grid engine requires 'config' (path to grid_backtest YAML)"
        )
    symbols = run.get("symbols") or grid.get("symbols")
    sym_arg = (
        ",".join(symbols)
        if isinstance(symbols, list)
        else "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,HYPEUSDT"
    )
    cmd = [
        sys.executable,
        "-m",
        "scripts.chop_grid_backtest",
        "--config",
        str(config),
        "--symbols",
        sym_arg,
        "--start",
        str(run["start_date"]),
        "--end",
        str(run["end_date"]),
        "--out-dir",
        str(out_path),
        "--no-maps",
    ]
    timeframe = run.get("timeframe") or grid.get("timeframe")
    if timeframe:
        cmd += ["--timeframe", str(timeframe)]
    etf = run.get("execution_timeframe") or grid.get("execution_timeframe")
    if etf:
        cmd += ["--execution-timeframe", str(etf)]
    ic = run.get("initial_capital") or grid.get("initial_capital")
    if ic:
        cmd += ["--initial-capital", str(ic)]
    run_extra = run.get("extra_argv") or []
    cmd += [str(x) for x in run_extra]
    cmd += extra_argv
    return cmd


def _build_trend_scalp_cmd(
    *,
    run: Dict[str, Any],
    grid: Dict[str, Any],
    out_path: Path,
    extra_argv: List[str],
) -> List[str]:
    config = run.get("config") or grid.get("config")
    if not config:
        raise ValueError("trend_scalp engine requires 'config'")
    symbols = run.get("symbols") or grid.get("symbols")
    sym_arg = ",".join(symbols) if isinstance(symbols, list) else str(symbols)
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "diagnose_dual_add_trend.py"),
        "--config",
        str(config),
        "--symbols",
        sym_arg,
        "--start",
        str(run["start_date"]),
        "--end",
        str(run["end_date"]),
        "--out-dir",
        str(out_path),
    ]
    tf = run.get("timeframe") or grid.get("timeframe") or "2h"
    cmd += ["--timeframe", str(tf)]
    etf = run.get("execution_timeframe") or grid.get("execution_timeframe")
    if etf:
        cmd += ["--execution-timeframe", str(etf)]
    ic = run.get("initial_capital") or grid.get("initial_capital")
    if ic:
        cmd += ["--initial-capital", str(ic)]
    cmd += extra_argv
    return cmd


def _build_multileg_timeline_cmd(
    *,
    seg_id: str,
    seg: Dict[str, Any],
    grid: Dict[str, Any],
    out_path: Path,
    extra_argv: List[str],
    preload_path: Path | None,
    save_preload: bool,
    run: Dict[str, Any] | None = None,
) -> List[str]:
    run = run or {}
    chop_cfg = run.get("chop_config") or grid.get("chop_config") or grid.get("config")
    trend_cfg = grid.get("trend_config") or grid.get("config")
    if not chop_cfg or not trend_cfg:
        raise ValueError("multileg_joint requires chop_config and trend_config")

    symbols = grid.get(
        "symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    )
    sym_arg = ",".join(symbols) if isinstance(symbols, list) else str(symbols)
    equity = float(grid.get("equity") or grid.get("initial_capital") or 10000)
    constitution_yaml = str(
        grid.get(
            "constitution_yaml",
            "live/highcap/config/constitution/constitution.yaml",
        )
    )
    data_dir = str(grid.get("data_dir") or "data/parquet_data")

    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "backtest_multileg_timeline.py"),
        "--start",
        str(seg["start_date"]),
        "--end",
        str(seg["end_date"]),
        "--symbols",
        sym_arg,
        "--data-dir",
        data_dir,
        "--equity",
        str(equity),
        "--chop-config",
        str(chop_cfg),
        "--trend-config",
        str(trend_cfg),
        "--constitution-yaml",
        constitution_yaml,
        "--summary-json",
        str(out_path / "summary.json"),
    ]
    if save_preload and preload_path is not None:
        cmd += ["--save-preload", str(preload_path)]
    elif preload_path is not None and preload_path.exists():
        cmd += ["--load-preload", str(preload_path)]

    if grid.get("trend_time_filter"):
        cmd.append("--trend-time-filter")
    cmd += extra_argv
    return cmd


def _run_multileg_joint_pipeline(
    grid: Dict[str, Any],
    extra_argv: List[str],
) -> int:
    segments = load_market_segments(
        grid.get("market_segment_path", "config/market_segment.yaml")
    )
    seg_ids = grid.get("segments") or list(segments)
    if not seg_ids:
        print("ERROR: multileg_joint requires segments list", file=sys.stderr)
        return 3

    backend_argv = []
    skip = False
    for a in extra_argv:
        if skip:
            skip = False
            continue
        if a in ("--segments",):
            skip = True
            continue
        backend_argv.append(a)

    out_root = Path(str(grid.get("output_dir", "results/multileg_joint")))
    if not out_root.is_absolute():
        out_root = (_REPO_ROOT / out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    preload_path = out_root / "preload.pkl"
    joint_summaries: List[Dict[str, Any]] = []
    worst_rc = 0

    for i, seg_id in enumerate(seg_ids):
        seg = segments.get(seg_id)
        if not seg:
            print(f"WARNING: unknown segment {seg_id!r}", file=sys.stderr)
            continue

        seg_out = out_root / "timeline" / seg_id
        seg_out.mkdir(parents=True, exist_ok=True)
        cmd = _build_multileg_timeline_cmd(
            seg_id=seg_id,
            seg=seg,
            grid=grid,
            out_path=seg_out,
            extra_argv=backend_argv,
            preload_path=preload_path,
            save_preload=(i == 0 and not preload_path.exists()),
        )
        print(f"\n=== multileg_joint[timeline]: {seg_id} ===")
        print(" ".join(cmd))
        rc = subprocess.run(cmd, cwd=str(_REPO_ROOT)).returncode
        worst_rc = max(worst_rc, rc)

        summary_path = seg_out / "summary.json"
        if summary_path.exists():
            row = json.loads(summary_path.read_text(encoding="utf-8"))
            row["segment_id"] = seg_id
            joint_summaries.append(row)

    joint_out = out_root / "joint"
    joint_out.mkdir(parents=True, exist_ok=True)
    payload = {
        "engine": "backtest_multileg_timeline",
        "experiment_id": grid.get("experiment_id"),
        "segments": joint_summaries,
    }
    joint_path = joint_out / "summary.json"
    joint_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nJoint timeline results: {joint_path}")
    for row in joint_summaries:
        print(
            f"  {row.get('segment_id')}: "
            f"{row.get('return_pct', 0):+.2f}%  "
            f"maxDD={row.get('max_drawdown_pct', 0):.2f}%  "
            f"halted={row.get('halted')}"
        )
    return worst_rc


def _run_multileg_joint_segment_matrix(
    grid: Dict[str, Any],
    extra_argv: List[str],
) -> int:
    """multileg_joint × segment_matrix: per-run chop_config and output_dir."""
    runs = _normalize_runs(grid)
    if not runs:
        print("ERROR: segment_matrix produced no runs", file=sys.stderr)
        return 3

    backend_argv: List[str] = []
    skip = False
    for a in extra_argv:
        if skip:
            skip = False
            continue
        if a in ("--segments",):
            skip = True
            continue
        backend_argv.append(a)

    segments = load_market_segments(
        grid.get("market_segment_path", "config/market_segment.yaml")
    )
    joint_summaries: List[Dict[str, Any]] = []
    worst_rc = 0
    preload_by_variant: Dict[str, Path] = {}

    for i, run in enumerate(runs):
        seg_id = str(run.get("segment") or "")
        seg = segments.get(seg_id)
        if not seg:
            print(f"WARNING: unknown segment {seg_id!r}", file=sys.stderr)
            continue

        out_dir = (
            run.get("output_dir") or grid.get("output_dir") or "results/multileg_joint"
        )
        out_path = Path(str(out_dir))
        if not out_path.is_absolute():
            out_path = (_REPO_ROOT / out_path).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        variant_key = str(
            run.get("chop_config") or grid.get("chop_config") or "default"
        )
        if variant_key not in preload_by_variant:
            preload_by_variant[variant_key] = out_path.parent / "preload.pkl"
        preload_path = preload_by_variant[variant_key]
        save_preload = not preload_path.exists()

        cmd = _build_multileg_timeline_cmd(
            seg_id=seg_id,
            seg=seg,
            grid=grid,
            run=run,
            out_path=out_path,
            extra_argv=backend_argv,
            preload_path=preload_path,
            save_preload=save_preload,
        )
        variant = str(run.get("variant") or f"{seg_id}_{i}")
        print(f"\n=== multileg_joint[matrix]: {variant} ===")
        print(" ".join(cmd))
        rc = subprocess.run(cmd, cwd=str(_REPO_ROOT)).returncode
        worst_rc = max(worst_rc, rc)

        summary_path = out_path / "summary.json"
        if summary_path.exists():
            row = json.loads(summary_path.read_text(encoding="utf-8"))
            row["variant"] = variant
            row["segment_id"] = seg_id
            joint_summaries.append(row)

    exp_id = grid.get("experiment_id") or "multileg_joint_matrix"
    joint_root = grid.get("output_dir") or "results/multileg_joint"
    joint_out = Path(str(joint_root))
    if not joint_out.is_absolute():
        joint_out = (_REPO_ROOT / joint_out).resolve()
    joint_out.mkdir(parents=True, exist_ok=True)
    joint_path = joint_out / "joint_summary.json"
    joint_path.write_text(
        json.dumps(
            {
                "engine": "backtest_multileg_timeline",
                "experiment_id": exp_id,
                "metrics_schema": "timeline_compound_v1",
                "runs": joint_summaries,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nJoint matrix results: {joint_path}")
    for row in joint_summaries:
        print(
            f"  {row.get('variant')}: "
            f"ret={row.get('return_pct', 0):+.2f}%  "
            f"pnl={row.get('realized_pnl_usdt', 0):+.0f}  "
            f"R={row.get('total_r', 0):+.1f}  "
            f"maxDD={row.get('max_drawdown_pct', 0):.2f}%  "
            f"Sharpe={row.get('sharpe_daily', 0):.2f}"
        )
    return worst_rc


def _run_one(
    *,
    run: Dict[str, Any],
    grid: Dict[str, Any],
    extra_argv: List[str],
) -> int:
    engine = _resolve_engine(run, grid)
    strategy = str(run.get("strategy") or grid.get("strategy") or "tpc")
    variant = str(run["variant"])
    out_dir = run.get("output_dir") or (f"results/{strategy}/experiments/{variant}")
    out_path = Path(out_dir)
    if not out_path.is_absolute():
        out_path = (_REPO_ROOT / out_path).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    if engine == _ENGINE_EVENT:
        cmd = _build_event_backtest_cmd(
            run=run, grid=grid, out_path=out_path, extra_argv=extra_argv
        )
    elif engine == _ENGINE_TREND_SCALP:
        cmd = _build_trend_scalp_cmd(
            run=run, grid=grid, out_path=out_path, extra_argv=extra_argv
        )
    else:
        cmd = _build_chop_grid_cmd(
            run=run, grid=grid, out_path=out_path, extra_argv=extra_argv
        )

    seg = run.get("segment")
    seg_note = f" segment={seg!r}" if seg else ""
    print(
        f"\n=== variant_grid[{engine}]: {variant}{seg_note} "
        f"({run.get('start_date')} → {run.get('end_date')}) ==="
    )
    print(" ".join(cmd))
    rc = subprocess.run(cmd, cwd=str(_REPO_ROOT)).returncode
    return int(rc)


def _write_index(
    *,
    grid_path: Path,
    grid: Dict[str, Any],
    runs_meta: List[Dict[str, Any]],
) -> Path:
    strategy = str(grid.get("strategy") or "tpc")
    index_dir = _REPO_ROOT / "results" / strategy / "experiments"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "EXPERIMENT_INDEX.json"
    payload: Dict[str, Any] = {
        "experiment_id": grid.get("experiment_id")
        or f"variant_grid_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        "decision_doc": grid.get("decision_doc"),
        "promoted_variant": grid.get("promoted_variant"),
        "symbols": grid.get("symbols"),
        "grid_yaml": str(grid_path),
        "runs": runs_meta,
    }
    if index_path.exists():
        try:
            prev = json.loads(index_path.read_text(encoding="utf-8"))
            prev_runs = prev.get("runs") or []
            if isinstance(prev_runs, list):
                seen = {r.get("variant") for r in runs_meta if isinstance(r, dict)}
                for pr in prev_runs:
                    if isinstance(pr, dict) and pr.get("variant") not in seen:
                        payload["runs"].append(pr)
        except Exception:
            pass
    index_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return index_path


def _normalize_runs(grid: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = (
        expand_segment_matrix(grid)
        if grid.get("segment_matrix")
        else (grid.get("runs") or [])
    )
    if not raw:
        return []
    seg_path = grid.get("market_segment_path")
    segments = (
        load_market_segments(seg_path)
        if seg_path or any(r.get("segment") for r in raw)
        else None
    )
    if segments is None and any(isinstance(r, dict) and r.get("segment") for r in raw):
        segments = load_market_segments()
    out: List[Dict[str, Any]] = []
    for run in raw:
        if not isinstance(run, dict):
            continue
        out.append(resolve_segment_run(run, grid=grid, segments=segments))
    return out


def run_variant_grid(
    grid_path: Path,
    *,
    extra_argv: List[str] | None = None,
) -> int:
    if not grid_path.is_absolute():
        grid_path = (_REPO_ROOT / grid_path).resolve()
    grid = _load_grid(grid_path)

    engine = _resolve_engine(grid, grid)
    if engine == _ENGINE_MULTILEG_JOINT:
        if grid.get("segment_matrix"):
            return _run_multileg_joint_segment_matrix(grid, list(extra_argv or []))
        return _run_multileg_joint_pipeline(grid, list(extra_argv or []))

    runs = _normalize_runs(grid)
    if not runs:
        print("ERROR: variant grid has no runs", file=sys.stderr)
        return 3

    from src.research.harness_registry import event_backtest_reject_reason

    eb_families = [
        str(run.get("strategy") or grid.get("strategy") or "")
        for run in runs
        if _resolve_engine(run, grid) == _ENGINE_EVENT
    ]
    wrong = event_backtest_reject_reason(eb_families)
    if wrong:
        print(wrong, file=sys.stderr)
        return 2

    extra = list(extra_argv or [])
    runs_meta: List[Dict[str, Any]] = []
    worst_rc = 0
    for run in runs:
        rc = _run_one(run=run, grid=grid, extra_argv=extra)
        worst_rc = max(worst_rc, rc)
        strategy = str(run.get("strategy") or grid.get("strategy") or "tpc")
        variant = str(run["variant"])
        out_dir = run.get("output_dir") or f"results/{strategy}/experiments/{variant}"
        runs_meta.append(
            {
                "variant": variant,
                "segment": run.get("segment"),
                "segment_label": run.get("segment_label"),
                "engine": _resolve_engine(run, grid),
                "period": f"{run['start_date']}/{run['end_date']}",
                "strategies_root": run.get("strategies_root")
                or grid.get("strategies_root"),
                "config": run.get("config") or grid.get("config"),
                "dir": str(out_dir),
                "exit_code": rc,
            }
        )

    index_path = _write_index(grid_path=grid_path, grid=grid, runs_meta=runs_meta)
    print(f"\nWrote {index_path}")
    return worst_rc
