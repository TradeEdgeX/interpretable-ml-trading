from __future__ import annotations

import json
import os
import html as _html
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass(frozen=True)
class KpiRow:
    severity: str  # hard_fail|warn
    name: str
    value: Any
    min: float | None
    max: float | None
    status: str  # PASS|FAIL|WARN|SKIP|MISSING
    note: str | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _read_yaml(p: Path) -> Dict[str, Any]:
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _to_float_or_none(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        return v
    except Exception:
        return None


def _get_metric(metrics: Dict[str, Any], key: str) -> float | None:
    """
    Get numeric metric by key from a potentially nested dict.
    Supports:
      - "a__b__c" path (double-underscore nested access)
      - flat keys
    """
    if not isinstance(metrics, dict):
        return None
    if key in metrics:
        return _to_float_or_none(metrics.get(key))
    if "__" in key:
        cur: Any = metrics
        for part in key.split("__"):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur.get(part)
        return _to_float_or_none(cur)
    return None


def _eval_gate(
    metrics: Dict[str, Any], gate: Dict[str, Any]
) -> Tuple[bool, List[KpiRow]]:
    hard = gate.get("hard_fail") or {}
    warn = gate.get("warn") or {}
    rows: List[KpiRow] = []

    def eval_block(block: Dict[str, Any], severity: str) -> None:
        for metric_name, rule in (block or {}).items():
            opt = False
            lo = None
            hi = None
            if isinstance(rule, (int, float)):
                lo = float(rule)
            elif isinstance(rule, list) and len(rule) == 2:
                lo = None if rule[0] is None else float(rule[0])
                hi = None if rule[1] is None else float(rule[1])
            elif isinstance(rule, dict):
                opt = bool(
                    rule.get("optional", False) or rule.get("skip_if_missing", False)
                )
                lo = None if rule.get("min", None) is None else float(rule.get("min"))
                hi = None if rule.get("max", None) is None else float(rule.get("max"))
            else:
                rows.append(
                    KpiRow(
                        severity=severity,
                        name=str(metric_name),
                        value=None,
                        min=None,
                        max=None,
                        status="FAIL" if severity == "hard_fail" else "WARN",
                        note="invalid gate rule",
                    )
                )
                continue

            v = _get_metric(metrics, str(metric_name))
            if v is None:
                if opt:
                    rows.append(
                        KpiRow(
                            severity=severity,
                            name=str(metric_name),
                            value=None,
                            min=lo,
                            max=hi,
                            status="SKIP",
                            note="optional missing",
                        )
                    )
                    continue
                rows.append(
                    KpiRow(
                        severity=severity,
                        name=str(metric_name),
                        value=None,
                        min=lo,
                        max=hi,
                        status="MISSING" if severity == "hard_fail" else "WARN",
                        note="missing metric",
                    )
                )
                continue

            ok = True
            if lo is not None and v < lo:
                ok = False
            if hi is not None and v > hi:
                ok = False

            if ok:
                rows.append(
                    KpiRow(
                        severity=severity,
                        name=str(metric_name),
                        value=v,
                        min=lo,
                        max=hi,
                        status="PASS",
                    )
                )
            else:
                rows.append(
                    KpiRow(
                        severity=severity,
                        name=str(metric_name),
                        value=v,
                        min=lo,
                        max=hi,
                        status="FAIL" if severity == "hard_fail" else "WARN",
                    )
                )

    eval_block(hard, "hard_fail")
    eval_block(warn, "warn")
    ok = all(
        r.status != "FAIL" and r.status != "MISSING"
        for r in rows
        if r.severity == "hard_fail"
    )
    return ok, rows


def _fmt(x: Any) -> str:
    if x is None:
        return "null"
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def _append_md_section(journal_path: Path, title: str, md_body: str) -> None:
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    if not journal_path.exists():
        journal_path.write_text("# KPI Journal\n\n", encoding="utf-8")
    with journal_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {title}\n\n")
        f.write(md_body)
        if not md_body.endswith("\n"):
            f.write("\n")


def _status_badge_html(status: str) -> str:
    s = str(status).upper()
    cls = {
        "PASS": "pass",
        "FAIL": "fail",
        "WARN": "warn",
        "SKIP": "skip",
        "MISSING": "missing",
    }.get(s, "neutral")
    return f'<span class="badge {cls}">{_html.escape(s)}</span>'


def _ok_badge_html(ok: bool) -> str:
    return _status_badge_html("PASS" if ok else "FAIL")


def _html_shell(title: str, body: str) -> str:
    # Single-file HTML with light styling (no external deps).
    return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>{_html.escape(title)}</title>
    <style>
      :root {{
        --bg: #0b1220;
        --panel: #111a2b;
        --panel2: #0f172a;
        --text: #e5e7eb;
        --muted: #9ca3af;
        --border: rgba(255,255,255,0.08);
        --pass: #22c55e;
        --fail: #ef4444;
        --warn: #f59e0b;
        --skip: #94a3b8;
        --missing: #fb7185;
      }}
      body {{
        margin: 0;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
        background: radial-gradient(1200px 700px at 20% 0%, #0b1b3a 0%, var(--bg) 55%);
        color: var(--text);
      }}
      .wrap {{ max-width: 1100px; margin: 0 auto; padding: 28px 18px 60px; }}
      h1 {{ font-size: 20px; margin: 0 0 6px; }}
      h2 {{ font-size: 16px; margin: 24px 0 10px; }}
      h3 {{ font-size: 14px; margin: 14px 0 8px; color: var(--text); }}
      .muted {{ color: var(--muted); }}
      .topbar {{
        position: sticky; top: 0;
        backdrop-filter: blur(10px);
        background: rgba(11,18,32,0.72);
        border-bottom: 1px solid var(--border);
        z-index: 10;
      }}
      .topbar .wrap {{ padding: 14px 18px; }}
      .card {{
        background: linear-gradient(180deg, rgba(17,26,43,0.95) 0%, rgba(15,23,42,0.95) 100%);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 14px 14px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.25);
      }}
      .grid {{ display: grid; gap: 12px; grid-template-columns: repeat(12, 1fr); }}
      .span-12 {{ grid-column: span 12; }}
      .span-6 {{ grid-column: span 6; }}
      .span-4 {{ grid-column: span 4; }}
      @media (max-width: 900px) {{
        .span-6, .span-4 {{ grid-column: span 12; }}
      }}
      .row {{ display:flex; gap: 12px; flex-wrap: wrap; align-items: center; }}
      code, .mono {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        background: rgba(255,255,255,0.06);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 2px 6px;
      }}
      a {{ color: #93c5fd; text-decoration: none; }}
      a:hover {{ text-decoration: underline; }}
      .badge {{
        display:inline-flex; align-items:center; justify-content:center;
        padding: 3px 8px; border-radius: 999px;
        font-size: 12px; font-weight: 700;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.04);
      }}
      .badge.pass {{ color: var(--pass); border-color: rgba(34,197,94,0.35); background: rgba(34,197,94,0.10); }}
      .badge.fail {{ color: var(--fail); border-color: rgba(239,68,68,0.35); background: rgba(239,68,68,0.10); }}
      .badge.warn {{ color: var(--warn); border-color: rgba(245,158,11,0.35); background: rgba(245,158,11,0.10); }}
      .badge.skip {{ color: var(--skip); border-color: rgba(148,163,184,0.35); background: rgba(148,163,184,0.10); }}
      .badge.missing {{ color: var(--missing); border-color: rgba(251,113,133,0.35); background: rgba(251,113,133,0.10); }}
      table {{
        width: 100%;
        border-collapse: collapse;
        overflow: hidden;
        border-radius: 12px;
        border: 1px solid var(--border);
      }}
      th, td {{
        padding: 10px 10px;
        border-bottom: 1px solid var(--border);
        font-size: 13px;
      }}
      th {{ text-align: left; color: var(--muted); font-weight: 700; background: rgba(255,255,255,0.03); }}
      tr:last-child td {{ border-bottom: 0; }}
      .right {{ text-align: right; }}
      .kpi-title {{ display:flex; justify-content: space-between; align-items:center; gap: 10px; }}
      .divider {{ height: 1px; background: var(--border); margin: 18px 0; }}
      .journal-section {{ margin-top: 18px; padding-top: 18px; border-top: 1px dashed var(--border); }}
      details.explain {{ display: inline-block; }}
      details.explain summary {{ cursor: pointer; list-style: none; }}
      details.explain summary::-webkit-details-marker {{ display: none; }}
      details.explain .exp-box {{
        margin-top: 8px;
        padding: 10px 10px;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.03);
        color: var(--muted);
        max-width: 720px;
      }}
    </style>
  </head>
  <body>
    <div class="topbar"><div class="wrap">
      <div class="row">
        <div><b>{_html.escape(title)}</b></div>
        <div class="muted">single-pane KPI dashboard</div>
      </div>
    </div></div>
    <div class="wrap">
      {body}
    </div>
  </body>
</html>
"""


def _append_html_section(journal_html_path: Path, title: str, html_body: str) -> None:
    """
    Append a section to a single-file HTML journal (append-only).
    """
    journal_html_path.parent.mkdir(parents=True, exist_ok=True)
    if not journal_html_path.exists():
        # Create shell with empty body; we'll append sections by inserting before closing tags.
        journal_html_path.write_text(
            _html_shell(
                title="KPI Journal",
                body="",
            ),
            encoding="utf-8",
        )
    raw = journal_html_path.read_text(encoding="utf-8")
    marker = "</div>\n  </body>\n</html>"
    if marker not in raw:
        # fallback: overwrite as shell
        raw = _html_shell(title="KPI Journal", body="")
    section = f"""
<div class="journal-section">
  <h2>{_html.escape(title)}</h2>
  {html_body}
</div>
"""
    out = raw.replace(marker, section + "\n" + marker)
    journal_html_path.write_text(out, encoding="utf-8")


def _metric_explain(name: str) -> str | None:
    """
    Human explanation for KPI fields (for UI tooltips/details).
    Keep short; link the mental model to docs/archive/architecture/who_owns_sharpe.md.
    """
    n = str(name)
    m: Dict[str, str] = {
        # --- A-layer (model) ---
        "dir_auc": "方向头 AUC（dir_y vs pred_dir_prob）。0.5=随机；>0.5 有信息；<0.5 可能反向或数据/标签对齐问题。",
        "mfe_atr_spearman": "MFE 头的 Spearman（预测 vs 真实，ATR 归一化）。衡量“能否排序出更大 MFE 的路径”。",
        "mae_atr_spearman": "MAE 头的 Spearman（预测 vs 真实）。衡量“风险侧路径强度”的排序能力。",
        "t_to_mfe_spearman": "τ 头（t_to_mfe）Spearman。衡量“多久到达 MFE”的排序能力。",
        "roll_icir__dir": "滚动 ICIR（dir）。稳定性指标：均值/方差比；越高越稳。",
        "trade__dir_auc": "在 Router 定义的 trade 子集（MEAN/TREND）上计算的 dir_auc。用于判断“只在可交易子集上有信息吗”。",
        "trade__rate": "Router trade 子集占比（MEAN/TREND 的样本比例）。不是盈利指标，只是覆盖率/密度。",
        # --- Router/system (counterfactual) ---
        "router_diag__trade_rate": "Router 最终交易密度（counterfactual test 区间）。过高=噪声/成本敏感；过低=空仓/无样本。",
        "rule_avg_mode_entropy": "Rule Router 模式熵（NO/MEAN/TREND）。太低=塌缩；太高=乱切换。",
        "rule_avg_max_dd": "Rule 系统回撤（WARN 级）。属于 system 层，Router 不应以 Sharpe 为 KPI，但可以作为风险警告。",
        "rule_pcm_avg_max_dd": "Gate/PCM 后的系统回撤。用于 gate 层或系统层的风控硬门槛。",
        "rule_pcm_avg_total_return": "Gate/PCM 后的累计收益。属于系统层，默认 WARN。",
        "rule_pcm_avg_mode_entropy": "Gate/PCM 后的模式熵（NO/MEAN/TREND）。用于防止塌缩或随机。",
        "rule_pcm_avg_switch_rate": "Gate/PCM 后的切换率。过高=不稳；过低=可能塌缩。",
        # --- Router KPIs per doc (mismatch/stability) ---
        "router_kpi__mismatch": "Router KPI: mismatch（按文档命名）。这里定义为 1 - acc_vs_rule_mode（shadow test 区间）。越低越好。",
        "router_kpi__acc_vs_rule_mode": "shadow 中 pred policy 与 rule mode 的一致率（越高越一致）。mismatch=1-acc。",
        "router_kpi__switch_rate_pred": "Router KPI: stability。pred policy 的切换率（越低越稳；但过低也可能塌缩）。",
        "router_kpi__mode_entropy_pred": "Router KPI: stability。pred policy 的模式熵（太低=塌缩；太高=随机）。",
        "router_diag__trade_win_rate": "Router 交易胜率（trade 子集）。是执行质量的粗 proxy，不等于最终 Sharpe。",
        "router_diag__trade_avg_ret": "Router 单笔平均收益（trade 子集）。粗 proxy，受成本/滑点影响。",
        # --- Plateau ---
        "plateau_frac_ge_95pct": "Plateau 稳健性：接近最优（best−5%·|best|）的候选比例。越大表示阈值不敏感、更可控。",
        "best__robust_score": "Plateau best 的鲁棒分数（多窗口+bootstrap）。分数可为负；关键是相对与 plateau_frac。",
        # --- Portfolio / Allocation ---
        "rule_pcm_sharpe_mean": "Gate/PCM 后系统 Sharpe（system 层 KPI）。",
        "rule_pcm_ann_return_mean": "Gate/PCM 后年化收益（system 层）。",
        "rule_pcm_ann_vol_mean": "Gate/PCM 后年化波动（system 层）。",
        "pa__avg_weight__GLOBAL_CASH": "组合层现金权重均值（资产配置 sanity check）。",
        "pa__avg_weight__GLOBAL_TREND": "组合层趋势策略权重均值（资产配置诊断）。",
        "pa__avg_weight__GLOBAL_MEAN": "组合层均值回复权重均值（资产配置诊断）。",
        "pa__avg_weight__DEFENSIVE_MEAN": "组合层防御均值权重均值（资产配置诊断）。",
        "pa__trend_zero_rate": "趋势策略空仓比率（过高=趋势腿缺失）。",
    }
    return m.get(n)


def _rows_table_html(rows: List[KpiRow]) -> str:
    tr = []
    for r in rows:
        exp = _metric_explain(str(r.name))
        if exp:
            metric_cell = (
                "<details class='explain'>"
                f"<summary><span class='mono'>{_html.escape(str(r.name))}</span></summary>"
                f"<div class='exp-box'>{_html.escape(exp)}</div>"
                "</details>"
            )
        else:
            metric_cell = f"<span class='mono'>{_html.escape(str(r.name))}</span>"
        tr.append(
            "<tr>"
            f"<td class='mono'>{_html.escape(str(r.severity))}</td>"
            f"<td>{metric_cell}</td>"
            f"<td class='right mono'>{_html.escape(_fmt(r.value))}</td>"
            f"<td class='right mono'>{_html.escape(_fmt(r.min))}</td>"
            f"<td class='right mono'>{_html.escape(_fmt(r.max))}</td>"
            f"<td>{_status_badge_html(r.status)}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr>"
        "<th>severity</th><th>metric</th><th class='right'>value</th>"
        "<th class='right'>min</th><th class='right'>max</th><th>status</th>"
        "</tr></thead>"
        "<tbody>" + "".join(tr) + "</tbody></table>"
    )


def _flatten_plateau_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    # Convert nested "best" fields to best__* keys so they can be gated with kpi_gate-style rules.
    out: Dict[str, Any] = {}
    if not isinstance(summary, dict):
        return out
    for k, v in summary.items():
        if k == "best" and isinstance(v, dict):
            for bk, bv in v.items():
                out[f"best__{bk}"] = bv
        else:
            out[k] = v
    return out


def find_run_root(start_dir: Path) -> Path | None:
    """
    Walk up a few levels to find a nnmultihead run root.
    Heuristic: presence of logs_3action.parquet OR router_thresholds_baseline.json OR e2e/ directory.
    """
    cur = start_dir.resolve()
    for _ in range(6):
        if (
            (cur / "logs_3action.parquet").exists()
            or (cur / "router_thresholds_baseline.json").exists()
            or (cur / "e2e").exists()
        ):
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def write_kpi_journal(
    *, run_dir: str | Path, stage: str, extra: Dict[str, Any] | None = None
) -> Path:
    """
    Append KPI status for a stage into <run_dir>/kpi_journal.md and write latest snapshot JSON/MD.

    Stages supported (best-effort):
      - train: reads latest */metrics.json under run_dir (or run_dir itself)
      - pipeline: reads run_dir/e2e/counterfactual/metrics.json
      - threshold_plateau: reads run_dir/threshold_plateau/summary.json
    """
    run_root = Path(run_dir).resolve()
    journal_path = run_root / "kpi_journal.md"
    journal_html_path = run_root / "kpi_journal.html"

    # Resolve gate YAML paths (defaults, overridable via env)
    primitives_gate = Path(
        os.getenv(
            "MLBOT_KPI_PRIMITIVES_YAML",
            "config/kpi_gates/nnmh_primitives_model.yaml",
        )
    ).resolve()
    safety_gate = Path(
        os.getenv("MLBOT_KPI_SAFETY_YAML", "config/kpi_gates/nnmh_safety_layer.yaml")
    ).resolve()
    gate_layer_gate = Path(
        os.getenv("MLBOT_KPI_GATE_LAYER_YAML", "config/kpi_gates/nnmh_gate_layer.yaml")
    ).resolve()
    execution_gate = Path(
        os.getenv(
            "MLBOT_KPI_EXECUTION_YAML",
            "config/kpi_gates/nnmh_execution_layer.yaml",
        )
    ).resolve()
    portfolio_gate = Path(
        os.getenv(
            "MLBOT_KPI_PORTFOLIO_YAML",
            "config/kpi_gates/nnmh_portfolio_allocation.yaml",
        )
    ).resolve()

    guardrails = [
        {
            "layer": "Execution Intelligence (Primitives)",
            "rule": "Calibrate confidence/SL/TP/holding-time; do not use Sharpe here.",
        },
        {
            "layer": "Safety",
            "rule": "Survival first: hard-fail if tail risk / max DD exceeds safety limits.",
        },
        {
            "layer": "System (Gate/Portfolio/Execution)",
            "rule": "Sharpe/DD/cost/slippage are optimized here after execution-intelligence pass.",
        },
    ]

    snap: Dict[str, Any] = {
        "kind": "kpi_snapshot_v1",
        "run_dir": str(run_root),
        "stage": str(stage),
        "created_at": _utc_now_iso(),
        "extra": extra or {},
        "guardrails": guardrails,
        "layers": {},
    }

    md = []
    md.append(f"- run_dir: `{run_root}`\n")
    md.append(f"- stage: **{stage}**\n")
    md.append(f"- created_at: `{snap['created_at']}`\n")
    md.append("\n### Workflow Guardrails\n")
    for g in guardrails:
        md.append(f"- **{g['layer']}**: {g['rule']}\n")

    # HTML snapshot body (pretty dashboard)
    html_parts: List[str] = []
    html_parts.append("<div class='card span-12'>")
    html_parts.append("<div class='kpi-title'>")
    html_parts.append(
        f"<div><h1>KPI Snapshot</h1><div class='muted'>stage: <b>{_html.escape(str(stage))}</b></div></div>"
    )
    html_parts.append(
        f"<div class='mono'>created_at: {_html.escape(str(snap['created_at']))}</div>"
    )
    html_parts.append("</div>")
    html_parts.append("<div class='card span-12'>")
    html_parts.append("<div class='kpi-title'>")
    html_parts.append(
        "<div><h2>Workflow Guardrails</h2><div class='muted'>Fixed evaluation principles for this run</div></div>"
    )
    html_parts.append("</div>")
    html_parts.append("<div class='divider'></div>")
    html_parts.append("<ul>")
    for g in guardrails:
        html_parts.append(
            f"<li><b>{_html.escape(g['layer'])}</b>: {_html.escape(g['rule'])}</li>"
        )
    html_parts.append("</ul>")
    html_parts.append("</div>")
    html_parts.append("<div class='divider'></div>")
    html_parts.append(
        f"<div class='row'><div class='mono'>run_dir</div><div class='mono'>{_html.escape(str(run_root))}</div></div>"
    )
    html_parts.append("</div>")

    # --- Layer: Execution Intelligence (Primitives) ---
    if stage in {"pipeline", "all"}:
        cf_p = run_root / "e2e" / "counterfactual" / "metrics.json"
        if cf_p.exists() and primitives_gate.exists():
            metrics = _read_json(cf_p)
            gate = _read_yaml(primitives_gate)
            ok, rows = _eval_gate(metrics, gate)
            snap["layers"]["primitives_model"] = {
                "metrics_json": str(cf_p),
                "gate_yaml": str(primitives_gate),
                "ok": bool(ok),
                "rows": [r.__dict__ for r in rows],
            }
            md.append("\n### Execution Intelligence (Primitives)\n")
            md.append(f"- metrics: `{cf_p}`\n")
            md.append(f"- gate: `{primitives_gate}`\n")
            md.append(f"- ok: **{ok}**\n\n")
            md.append("| severity | metric | value | min | max | status |\n")
            md.append("|---|---|---:|---:|---:|---|\n")
            for r in rows:
                md.append(
                    f"| {r.severity} | `{r.name}` | {_fmt(r.value)} | {_fmt(r.min)} | {_fmt(r.max)} | **{r.status}** |\n"
                )
            html_parts.append("<div class='grid'>")
            html_parts.append("<div class='card span-12'>")
            html_parts.append("<div class='kpi-title'>")
            html_parts.append(
                "<div><h2>Execution Intelligence (Primitives)</h2><div class='muted'>Sizing/SL/TP/holding-time control (no Sharpe here)</div></div>"
            )
            html_parts.append(f"<div>{_ok_badge_html(ok)}</div>")
            html_parts.append("</div>")
            html_parts.append("<div class='row'>")
            html_parts.append(
                f"<div>metrics: <a class='mono' href='{_html.escape(str(cf_p))}'>{_html.escape(str(cf_p))}</a></div>"
            )
            html_parts.append(
                f"<div>gate: <a class='mono' href='{_html.escape(str(primitives_gate))}'>{_html.escape(str(primitives_gate))}</a></div>"
            )
            html_parts.append("</div>")
            html_parts.append("<div class='divider'></div>")
            html_parts.append(_rows_table_html(rows))
            html_parts.append("</div></div>")
        else:
            md.append(
                "\n### Execution Intelligence (Primitives)\n- (missing e2e/counterfactual/metrics.json or gate yaml)\n"
            )
            html_parts.append(
                "<div class='card span-12'><h2>Execution Intelligence (Primitives)</h2><div class='muted'>(missing e2e/counterfactual/metrics.json or gate yaml)</div></div>"
            )

    # --- Layer: Gate (PCM / rules) ---
    if stage in {"pipeline", "all"}:
        cf_p = run_root / "e2e" / "counterfactual" / "metrics.json"
        # --- Layer: Safety ---
        if cf_p.exists() and safety_gate.exists():
            metrics_safety = _read_json(cf_p)
            gate = _read_yaml(safety_gate)
            ok, rows = _eval_gate(metrics_safety, gate)
            snap["layers"]["safety_layer"] = {
                "metrics_json": str(cf_p),
                "gate_yaml": str(safety_gate),
                "ok": bool(ok),
                "rows": [r.__dict__ for r in rows],
            }
            md.append("\n### Safety Layer\n")
            md.append(f"- metrics: `{cf_p}`\n")
            md.append(f"- gate: `{safety_gate}`\n")
            md.append(f"- ok: **{ok}**\n\n")
            md.append("| severity | metric | value | min | max | status |\n")
            md.append("|---|---|---:|---:|---:|---|\n")
            for r in rows:
                md.append(
                    f"| {r.severity} | `{r.name}` | {_fmt(r.value)} | {_fmt(r.min)} | {_fmt(r.max)} | **{r.status}** |\n"
                )
            html_parts.append("<div class='grid'>")
            html_parts.append("<div class='card span-12'>")
            html_parts.append("<div class='kpi-title'>")
            html_parts.append(
                "<div><h2>Safety Layer</h2><div class='muted'>生存优先：最大回撤 / 尾部风险</div></div>"
            )
            html_parts.append(f"<div>{_ok_badge_html(ok)}</div>")
            html_parts.append("</div>")
            html_parts.append("<div class='row'>")
            html_parts.append(
                f"<div>metrics: <a class='mono' href='{_html.escape(str(cf_p))}'>{_html.escape(str(cf_p))}</a></div>"
            )
            html_parts.append(
                f"<div>gate: <a class='mono' href='{_html.escape(str(safety_gate))}'>{_html.escape(str(safety_gate))}</a></div>"
            )
            html_parts.append("</div>")
            html_parts.append("<div class='divider'></div>")
            html_parts.append(_rows_table_html(rows))
            html_parts.append("</div></div>")
        else:
            md.append(
                "\n### Safety Layer\n- (missing e2e/counterfactual/metrics.json or gate yaml)\n"
            )
            html_parts.append(
                "<div class='card span-12'><h2>Safety Layer</h2><div class='muted'>(missing e2e/counterfactual/metrics.json or gate yaml)</div></div>"
            )

        if cf_p.exists() and gate_layer_gate.exists():
            metrics_gate = _read_json(cf_p)
            gate = _read_yaml(gate_layer_gate)
            ok, rows = _eval_gate(metrics_gate, gate)
            snap["layers"]["gate_layer"] = {
                "metrics_json": str(cf_p),
                "gate_yaml": str(gate_layer_gate),
                "ok": bool(ok),
                "rows": [r.__dict__ for r in rows],
            }
            md.append("\n### Gate Layer (PCM)\n")
            md.append(f"- metrics: `{cf_p}`\n")
            md.append(f"- gate: `{gate_layer_gate}`\n")
            md.append(f"- ok: **{ok}**\n\n")
            md.append("| severity | metric | value | min | max | status |\n")
            md.append("|---|---|---:|---:|---:|---|\n")
            for r in rows:
                md.append(
                    f"| {r.severity} | `{r.name}` | {_fmt(r.value)} | {_fmt(r.min)} | {_fmt(r.max)} | **{r.status}** |\n"
                )
            html_parts.append("<div class='grid'>")
            html_parts.append("<div class='card span-12'>")
            html_parts.append("<div class='kpi-title'>")
            html_parts.append(
                "<div><h2>Gate Layer (PCM)</h2><div class='muted'>风险控制/过滤层（不以 Sharpe 为核心 KPI）</div></div>"
            )
            html_parts.append(f"<div>{_ok_badge_html(ok)}</div>")
            html_parts.append("</div>")
            html_parts.append("<div class='row'>")
            html_parts.append(
                f"<div>metrics: <a class='mono' href='{_html.escape(str(cf_p))}'>{_html.escape(str(cf_p))}</a></div>"
            )
            html_parts.append(
                f"<div>gate: <a class='mono' href='{_html.escape(str(gate_layer_gate))}'>{_html.escape(str(gate_layer_gate))}</a></div>"
            )
            html_parts.append("</div>")
            html_parts.append("<div class='divider'></div>")
            html_parts.append(_rows_table_html(rows))
            html_parts.append("</div></div>")
        else:
            md.append(
                "\n### Gate Layer (PCM)\n- (missing e2e/counterfactual/metrics.json or gate yaml)\n"
            )
            html_parts.append(
                "<div class='card span-12'><h2>Gate Layer (PCM)</h2><div class='muted'>(missing e2e/counterfactual/metrics.json or gate yaml)</div></div>"
            )

        # --- Layer: Execution ---
        if cf_p.exists() and execution_gate.exists():
            metrics_exec = _read_json(cf_p)
            gate = _read_yaml(execution_gate)
            ok, rows = _eval_gate(metrics_exec, gate)
            snap["layers"]["execution_layer"] = {
                "metrics_json": str(cf_p),
                "gate_yaml": str(execution_gate),
                "ok": bool(ok),
                "rows": [r.__dict__ for r in rows],
            }
            md.append("\n### Execution Layer\n")
            md.append(f"- metrics: `{cf_p}`\n")
            md.append(f"- gate: `{execution_gate}`\n")
            md.append(f"- ok: **{ok}**\n\n")
            md.append("| severity | metric | value | min | max | status |\n")
            md.append("|---|---|---:|---:|---:|---|\n")
            for r in rows:
                md.append(
                    f"| {r.severity} | `{r.name}` | {_fmt(r.value)} | {_fmt(r.min)} | {_fmt(r.max)} | **{r.status}** |\n"
                )
            html_parts.append("<div class='grid'>")
            html_parts.append("<div class='card span-12'>")
            html_parts.append("<div class='kpi-title'>")
            html_parts.append(
                "<div><h2>Execution Layer</h2><div class='muted'>成交质量/成本敏感性（当前使用 trade 诊断 proxy）</div></div>"
            )
            html_parts.append(f"<div>{_ok_badge_html(ok)}</div>")
            html_parts.append("</div>")
            html_parts.append("<div class='row'>")
            html_parts.append(
                f"<div>metrics: <a class='mono' href='{_html.escape(str(cf_p))}'>{_html.escape(str(cf_p))}</a></div>"
            )
            html_parts.append(
                f"<div>gate: <a class='mono' href='{_html.escape(str(execution_gate))}'>{_html.escape(str(execution_gate))}</a></div>"
            )
            html_parts.append("</div>")
            html_parts.append("<div class='divider'></div>")
            html_parts.append(_rows_table_html(rows))
            html_parts.append("</div></div>")
        else:
            md.append(
                "\n### Execution Layer\n- (missing e2e/counterfactual/metrics.json or gate yaml)\n"
            )
            html_parts.append(
                "<div class='card span-12'><h2>Execution Layer</h2><div class='muted'>(missing e2e/counterfactual/metrics.json or gate yaml)</div></div>"
            )

        # --- Layer: Portfolio / Allocation ---
        if cf_p.exists() and portfolio_gate.exists():
            metrics_port = _read_json(cf_p)
            gate = _read_yaml(portfolio_gate)
            ok, rows = _eval_gate(metrics_port, gate)
            snap["layers"]["portfolio_allocation"] = {
                "metrics_json": str(cf_p),
                "gate_yaml": str(portfolio_gate),
                "ok": bool(ok),
                "rows": [r.__dict__ for r in rows],
            }
            md.append("\n### Portfolio / Allocation\n")
            md.append(f"- metrics: `{cf_p}`\n")
            md.append(f"- gate: `{portfolio_gate}`\n")
            md.append(f"- ok: **{ok}**\n\n")
            md.append("| severity | metric | value | min | max | status |\n")
            md.append("|---|---|---:|---:|---:|---|\n")
            for r in rows:
                md.append(
                    f"| {r.severity} | `{r.name}` | {_fmt(r.value)} | {_fmt(r.min)} | {_fmt(r.max)} | **{r.status}** |\n"
                )
            html_parts.append("<div class='grid'>")
            html_parts.append("<div class='card span-12'>")
            html_parts.append("<div class='kpi-title'>")
            html_parts.append(
                "<div><h2>Portfolio / Allocation</h2><div class='muted'>系统层 KPI（Sharpe/DD/配置健康）</div></div>"
            )
            html_parts.append(f"<div>{_ok_badge_html(ok)}</div>")
            html_parts.append("</div>")
            html_parts.append("<div class='row'>")
            html_parts.append(
                f"<div>metrics: <a class='mono' href='{_html.escape(str(cf_p))}'>{_html.escape(str(cf_p))}</a></div>"
            )
            html_parts.append(
                f"<div>gate: <a class='mono' href='{_html.escape(str(portfolio_gate))}'>{_html.escape(str(portfolio_gate))}</a></div>"
            )
            html_parts.append("</div>")
            html_parts.append("<div class='divider'></div>")
            html_parts.append(_rows_table_html(rows))
            html_parts.append("</div></div>")
        else:
            md.append(
                "\n### Portfolio / Allocation\n- (missing e2e/counterfactual/metrics.json or gate yaml)\n"
            )
            html_parts.append(
                "<div class='card span-12'><h2>Portfolio / Allocation</h2><div class='muted'>(missing e2e/counterfactual/metrics.json or gate yaml)</div></div>"
            )

    # --- Layer: Plateau tuning robustness ---
    # Write journal + latest snapshot
    title = f"{stage} @ {snap['created_at']}"
    _append_md_section(journal_path, title, "".join(md))
    _append_html_section(journal_html_path, title, "".join(html_parts))
    (run_root / "kpi_latest.json").write_text(
        json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_root / "kpi_latest.md").write_text("".join(md), encoding="utf-8")
    (run_root / "kpi_latest.html").write_text(
        _html_shell(title="KPI Latest", body="".join(html_parts)),
        encoding="utf-8",
    )
    return journal_path
