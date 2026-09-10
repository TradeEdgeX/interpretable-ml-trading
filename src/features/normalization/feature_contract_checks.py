"""
Feature contract checks (normalization + semantic safety).

This module is intentionally lightweight:
- No ta-lib imports
- Only reads config/feature_dependencies.yaml

It is designed to be used by:
- `mlbot diagnose feature-contract`
- CI gates
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.features.normalization.feature_contract import (
    collect_feature_normalization_meta,
    validate_feature_dependencies_normalization,
)

_PROTECTED_SCALE_COLS: Dict[str, str] = {
    # column_name -> expected producer method
    "atr": "price_unit",
    "market_cap_usd": "usd",
}


def _check_protected_scale_consumers(
    deps: Dict[str, Any],
    *,
    by_col: Dict[str, Dict[str, Any]],
) -> List[str]:
    """
    Second-stage safety check:
    If a feature consumes a protected scale column, it must explicitly declare
    that it expects that input to have the same method.

    Why:
    - Prevent silent bugs where a scale column's semantics change (e.g., atr -> atr/close)
      but downstream math still assumes price-unit ATR.
    """
    features = (deps or {}).get("features", {}) or {}
    errs: List[str] = []

    for feat_name, info in features.items():
        req_cols = info.get("required_columns") or []
        used = [c for c in req_cols if c in _PROTECTED_SCALE_COLS]
        if not used:
            continue

        cp = info.get("compute_params") or {}
        in_map = cp.get("input_normalization_map") or {}

        for c in used:
            expected_method = _PROTECTED_SCALE_COLS[c]
            # Also ensure producer method matches expectation (global invariant)
            prod = by_col.get(c, {})
            prod_method = str(prod.get("method", ""))
            if prod_method and prod_method != expected_method:
                errs.append(
                    f"protected_scale_producer_mismatch: col={c} expected={expected_method} got={prod_method}"
                )

            got = in_map.get(c)
            if got is None:
                errs.append(
                    f"protected_scale_input_not_declared: feature={feat_name} requires {c} but compute_params.input_normalization_map missing"
                )
            else:
                if str(got) != str(expected_method):
                    errs.append(
                        f"protected_scale_input_wrong: feature={feat_name} col={c} expected={expected_method} got={got}"
                    )

    return errs


def _load_yaml(path: str) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _index_meta(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    # column -> meta (best-effort; columns are expected unique across outputs)
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        out[str(r["column"])] = r
    return out


def _check_expected_column_methods(
    by_col: Dict[str, Dict[str, Any]],
    expected: Dict[str, str],
) -> List[str]:
    errs: List[str] = []
    for col, method in expected.items():
        if col not in by_col:
            errs.append(f"missing_expected_column: {col} (expected method={method})")
            continue
        got = str(by_col[col].get("method"))
        if got != str(method):
            errs.append(f"wrong_method: {col} expected={method} got={got}")
    return errs


def run_feature_contract_checks(
    *,
    feature_deps_path: str = "config/feature_dependencies.yaml",
    mode: str = "error",  # error|warn
) -> Tuple[bool, Dict[str, Any]]:
    """
    Returns (ok, report).
    """
    deps = _load_yaml(feature_deps_path)

    # 1) Base contract: no missing methods (this is the core contract)
    base_report = validate_feature_dependencies_normalization(deps, mode="error")

    rows = collect_feature_normalization_meta(deps, only_features=None)
    by_col = _index_meta(rows)

    # 2) Scale-column expectations (semantic safety)
    expected_methods = dict(_PROTECTED_SCALE_COLS)
    method_errors = _check_expected_column_methods(by_col, expected_methods)

    # 3) Protected scale consumers must explicitly declare input semantics.
    consumer_errors = _check_protected_scale_consumers(deps, by_col=by_col)

    # 4) Order-flow raw-source leakage (best-effort)
    #
    # We cannot fully reason about base data columns here, but we can at least ensure
    # that feature outputs are not left as `raw` unintentionally.
    raw = [r for r in rows if r.get("method") == "raw"]
    raw_errs = [f"raw_output_column: {r['feature']}:{r['column']}" for r in raw[:50]]

    errors: List[str] = []
    errors.extend(method_errors)
    errors.extend(consumer_errors)
    errors.extend(raw_errs)

    report = {
        "ok": len(errors) == 0 and bool(base_report.get("ok", False)),
        "feature_deps": str(feature_deps_path),
        "base_contract": base_report,
        "expected_method_errors": method_errors,
        "protected_scale_consumer_errors": consumer_errors,
        "raw_output_columns_count": len(raw),
        "errors": errors,
    }

    if (not report["ok"]) and str(mode).lower() == "error":
        return False, report
    return True, report


def main() -> int:
    ap = argparse.ArgumentParser(description="Feature contract checks (normalization + semantic safety).")
    ap.add_argument("--feature-deps", default="config/feature_dependencies.yaml")
    ap.add_argument("--mode", default="error", choices=["error", "warn"])
    ap.add_argument("--out-json", default=None, help="Optional output JSON path.")
    args = ap.parse_args()

    ok, report = run_feature_contract_checks(
        feature_deps_path=str(args.feature_deps),
        mode=str(args.mode),
    )

    if args.out_json:
        p = Path(args.out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())


