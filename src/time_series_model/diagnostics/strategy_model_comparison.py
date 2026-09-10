"""
Generic strategy model comparison (strategy-agnostic).

This tool compares multiple strategy config directories by *running the unified*
`scripts/train_strategy_pipeline.py` for each config under identical evaluation
settings (symbol/timeframe/test split/seed/date crop), then summarizing:

- CV metrics (if present)
- Backtest metrics (sharpe, trades, drawdown, return)
- Diagnostics snapshot (label/pred distribution, entries/exits)

It is intentionally not SR-reversal-specific (unlike sr_reversal_model_comparison.py).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_strategy_list(raw: str) -> List[str]:
    items = [s.strip() for s in (raw or "").split(",") if s.strip()]
    if not items:
        raise ValueError("--strategy-config must contain at least 1 strategy directory")

    # Auto-complete: if path doesn't exist, try prepending "config/strategies/"
    resolved = []
    for item in items:
        p = Path(item)
        if p.exists():
            resolved.append(str(p.resolve()))
        else:
            # Try with config/strategies/ prefix
            alt = Path("config/strategies") / item
            if alt.exists():
                resolved.append(str(alt.resolve()))
            else:
                # Keep original (will fail later with clearer error)
                resolved.append(item)
    return resolved


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_get(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _find_results_json(strategy_out_dir: Path) -> Path:
    # Default convention
    p = strategy_out_dir / "results.json"
    if p.exists():
        return p
    # Fallback: pick the newest *results*.json
    cands = sorted(
        strategy_out_dir.glob("**/*results*.json"),
        key=lambda x: x.stat().st_mtime if x.exists() else 0,
        reverse=True,
    )
    if cands:
        return cands[0]
    raise FileNotFoundError(f"No results JSON found under {strategy_out_dir}")


@dataclass(frozen=True)
class OneRunSummary:
    strategy: str
    results_path: str
    cv_score: Optional[float]
    sharpe: Optional[float]
    total_return_pct: Optional[float]
    max_drawdown_pct: Optional[float]
    total_trades: Optional[int]
    label_summary: Optional[Dict[str, Any]]
    pred_summary: Optional[Dict[str, Any]]
    entries_exits: Optional[Dict[str, Any]]


def _summarize_results(
    strategy_name: str, results: Dict[str, Any], results_path: Path
) -> OneRunSummary:
    # Objective-ish score: prefer CV sharpe mean, else backtest sharpe.
    cv_score = None
    cv_block = results.get("cv")
    if isinstance(cv_block, dict):
        cv_score = cv_block.get("Sharpe_mean", None)

    backtest = (
        results.get("backtest") if isinstance(results.get("backtest"), dict) else {}
    )
    sharpe = backtest.get("sharpe", None)
    total_return_pct = backtest.get("total_return_pct", None)
    max_drawdown_pct = backtest.get("max_drawdown_pct", None)
    total_trades = backtest.get("total_trades", None)

    diag = (
        results.get("diagnostics")
        if isinstance(results.get("diagnostics"), dict)
        else {}
    )
    label_summary = _safe_get(diag, ["labels"], None)
    pred_summary = _safe_get(diag, ["predictions"], None)
    entries_exits = _safe_get(backtest, ["diagnostics", "entries_exits"], None)

    return OneRunSummary(
        strategy=strategy_name,
        results_path=str(results_path),
        cv_score=float(cv_score) if cv_score is not None else None,
        sharpe=float(sharpe) if sharpe is not None else None,
        total_return_pct=(
            float(total_return_pct) if total_return_pct is not None else None
        ),
        max_drawdown_pct=(
            float(max_drawdown_pct) if max_drawdown_pct is not None else None
        ),
        total_trades=int(total_trades) if total_trades is not None else None,
        label_summary=label_summary,
        pred_summary=pred_summary,
        entries_exits=entries_exits,
    )


def _render_report_html(summaries: List[OneRunSummary], out_html: Path) -> None:
    rows = []
    for s in summaries:
        rows.append(
            "<tr>"
            f"<td><code>{s.strategy}</code></td>"
            f"<td>{'' if s.cv_score is None else f'{s.cv_score:.4f}'}</td>"
            f"<td>{'' if s.sharpe is None else f'{s.sharpe:.4f}'}</td>"
            f"<td>{'' if s.total_return_pct is None else f'{s.total_return_pct:.2f}'}</td>"
            f"<td>{'' if s.max_drawdown_pct is None else f'{s.max_drawdown_pct:.2f}'}</td>"
            f"<td>{'' if s.total_trades is None else str(s.total_trades)}</td>"
            f"<td><code>{s.results_path}</code></td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Strategy Model Comparison</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ background: #f6f6f6; text-align: left; }}
    code {{ background: #f3f3f3; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h2>Strategy Model Comparison</h2>
  <p>Each row is a full run of <code>scripts/train_strategy_pipeline.py</code> under identical settings.</p>
  <table>
    <thead>
      <tr>
        <th>strategy</th>
        <th>cv_score (Sharpe_mean)</th>
        <th>backtest_sharpe</th>
        <th>return%</th>
        <th>dd%</th>
        <th>trades</th>
        <th>results.json</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generic strategy model comparison (strategy-agnostic)"
    )
    p.add_argument(
        "--strategy-config",
        type=str,
        required=True,
        help="Comma-separated list of strategy config directories",
    )
    p.add_argument(
        "--symbol", type=str, required=True, help="Symbol (or comma-separated symbols)"
    )
    p.add_argument("--timeframe", type=str, default="240T")
    p.add_argument("--data-path", type=str, default="data/parquet_data")
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument(
        "--start-date", type=str, default=None, help="Optional date crop (YYYY-MM-DD)"
    )
    p.add_argument(
        "--end-date", type=str, default=None, help="Optional date crop (YYYY-MM-DD)"
    )
    p.add_argument("--output-dir", type=str, default="results/model_comparison_generic")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    configs = _parse_strategy_list(args.strategy_config)

    out_dir = Path(args.output_dir)
    _ensure_dir(out_dir)
    runs_root = out_dir / "runs"
    _ensure_dir(runs_root)

    env = os.environ.copy()
    if args.start_date:
        env["TRAIN_START_DATE"] = str(args.start_date)
    if args.end_date:
        env["TRAIN_END_DATE"] = str(args.end_date)

    summaries: List[OneRunSummary] = []
    for cfg in configs:
        cfg_path = Path(cfg)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Strategy config not found: {cfg}")

        cmd = [
            sys.executable,
            "scripts/train_strategy_pipeline.py",
            "--config",
            str(cfg),
            "--symbol",
            str(args.symbol),
            "--data-path",
            str(args.data_path),
            "--timeframe",
            str(args.timeframe),
            "--test-size",
            str(args.test_size),
            "--seed",
            str(args.seed),
            "--output-root",
            str(runs_root),
        ]
        if bool(args.deterministic):
            cmd.append("--deterministic")

        print("\n" + "=" * 80)
        print(f"▶ Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=env)

        # train_strategy_pipeline writes to output_root / strategy_dir_name
        strategy_name = cfg_path.name
        strategy_out_dir = runs_root / strategy_name
        results_path = _find_results_json(strategy_out_dir)
        results = json.loads(results_path.read_text(encoding="utf-8"))
        summaries.append(_summarize_results(strategy_name, results, results_path))

    out_json = out_dir / "model_comparison_results.json"
    out_json.write_text(
        json.dumps(
            {
                "strategies": [s.strategy for s in summaries],
                "summaries": [s.__dict__ for s in summaries],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    out_html = out_dir / "model_comparison_report.html"
    _render_report_html(summaries, out_html)

    print("\n✅ Saved:")
    print(f"  - {out_json}")
    print(f"  - {out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
