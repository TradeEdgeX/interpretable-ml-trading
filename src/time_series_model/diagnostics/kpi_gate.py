from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


@dataclass(frozen=True)
class KpiGateResult:
    ok: bool
    hard_failures: List[str]
    warnings: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "hard_failures": list(self.hard_failures),
            "warnings": list(self.warnings),
        }


def _load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _get_metric(metrics: Dict[str, Any], key: str) -> float | None:
    """
    Support dotted keys: e.g. "summary.rule_sharpe".
    """
    cur: Any = metrics
    for part in str(key).split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur.get(part)
    try:
        return float(cur)
    except Exception:
        return None


def _check_range(
    name: str, value: float | None, lo: float | None, hi: float | None
) -> str | None:
    if value is None:
        return f"{name}: missing"
    if lo is not None and float(value) < float(lo):
        return f"{name}: {value} < {lo}"
    if hi is not None and float(value) > float(hi):
        return f"{name}: {value} > {hi}"
    return None


def check_kpi_gate(*, metrics: Dict[str, Any], gate: Dict[str, Any]) -> KpiGateResult:
    hard = gate.get("hard_fail") or {}
    warn = gate.get("warn") or {}

    hard_failures: List[str] = []
    warnings: List[str] = []

    def eval_block(block: Dict[str, Any], out: List[str]) -> None:
        for metric_name, rule in (block or {}).items():
            if isinstance(rule, (int, float)):
                # shorthand: min threshold
                v = _get_metric(metrics, metric_name)
                msg = _check_range(metric_name, v, float(rule), None)
                if msg:
                    out.append(msg)
                continue

            if isinstance(rule, list) and len(rule) == 2:
                lo, hi = rule
                lo_f = None if lo is None else float(lo)
                hi_f = None if hi is None else float(hi)
                v = _get_metric(metrics, metric_name)
                msg = _check_range(metric_name, v, lo_f, hi_f)
                if msg:
                    out.append(msg)
                continue

            if isinstance(rule, dict):
                # Optional/skip-if-missing support so gates can evolve without breaking older reports.
                skip_if_missing = bool(
                    rule.get("optional", False) or rule.get("skip_if_missing", False)
                )
                lo = rule.get("min", None)
                hi = rule.get("max", None)
                lo_f = None if lo is None else float(lo)
                hi_f = None if hi is None else float(hi)
                v = _get_metric(metrics, metric_name)
                if v is None and skip_if_missing:
                    continue
                msg = _check_range(metric_name, v, lo_f, hi_f)
                if msg:
                    out.append(msg)
                continue

            out.append(
                f"{metric_name}: invalid gate rule (expected number, [min,max], or {{min,max}})"
            )

    eval_block(hard, hard_failures)
    eval_block(warn, warnings)
    ok = len(hard_failures) == 0
    return KpiGateResult(ok=ok, hard_failures=hard_failures, warnings=warnings)


def run_kpi_gate(
    *,
    metrics_json: str | Path,
    gate_yaml: str | Path,
    out_json: str | Path | None = None,
) -> Tuple[int, KpiGateResult]:
    metrics = _load_json(metrics_json)
    gate = _load_yaml(gate_yaml)
    res = check_kpi_gate(metrics=metrics, gate=gate)

    if out_json is not None:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(
            json.dumps(res.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return (0 if res.ok else 2), res
