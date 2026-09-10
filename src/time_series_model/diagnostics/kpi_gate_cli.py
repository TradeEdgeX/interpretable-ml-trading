from __future__ import annotations

import argparse
import sys

from .kpi_gate import run_kpi_gate


def main() -> None:
    ap = argparse.ArgumentParser(description="KPI gate checker (hard-fail for CI).")
    ap.add_argument("--metrics-json", required=True)
    ap.add_argument("--gate-yaml", required=True)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    rc, res = run_kpi_gate(
        metrics_json=args.metrics_json,
        gate_yaml=args.gate_yaml,
        out_json=args.out_json,
    )
    print("kpi_gate:", res.as_dict())
    raise SystemExit(int(rc))


if __name__ == "__main__":
    main()
