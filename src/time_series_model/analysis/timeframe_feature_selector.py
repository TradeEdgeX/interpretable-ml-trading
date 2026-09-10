#!/usr/bin/env python3
"""
Utility to convert timeframe-forward correlation outputs into reusable training configurations.

Given a `timeframe_forward_details.csv` produced by `timeframe_forward_correlation.py`,
this script filters the strongest features, groups them by (timeframe, forward horizon),
and emits both machine-friendly artefacts (YAML/JSON) and human-readable summaries.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


DEFAULT_PVALUE_THRESHOLD = 1e-5


@dataclass
class SelectorConfig:
    details_csv: Path
    output_dir: Path
    pearson_threshold: float
    pvalue_threshold: Optional[float]
    min_samples: int
    top_features_per_symbol: int
    top_features_per_group: int
    config_filename: str
    summary_filename: str
    json_filename: str
    min_symbols_per_group: int


def parse_args() -> SelectorConfig:
    parser = argparse.ArgumentParser(
        description="Build high-IC feature/timeframe configurations from correlation details."
    )
    parser.add_argument(
        "--details-csv",
        type=Path,
        required=True,
        help="Path to timeframe_forward_details.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/timeframe_configs"),
        help="Directory for generated configs and summaries.",
    )
    parser.add_argument(
        "--pearson-threshold",
        type=float,
        default=0.25,
        help="Minimum absolute Pearson correlation to keep a feature.",
    )
    parser.add_argument(
        "--pvalue-threshold",
        type=float,
        default=DEFAULT_PVALUE_THRESHOLD,
        help="Maximum Pearson p-value allowed (set negative to skip filter).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=500,
        help="Minimum sample count required per (symbol, timeframe, horizon).",
    )
    parser.add_argument(
        "--top-features-per-symbol",
        type=int,
        default=5,
        help="How many features to retain per (symbol, timeframe, forward) combination.",
    )
    parser.add_argument(
        "--top-features-per-group",
        type=int,
        default=10,
        help="How many aggregated features to keep for each (timeframe, forward) group.",
    )
    parser.add_argument(
        "--config-filename",
        type=str,
        default="strategy_groups.yaml",
        help="Output YAML file name.",
    )
    parser.add_argument(
        "--summary-filename",
        type=str,
        default="strategy_summary.md",
        help="Output Markdown summary filename.",
    )
    parser.add_argument(
        "--json-filename",
        type=str,
        default="strategy_groups.json",
        help="Optional JSON manifest mirroring the YAML structure.",
    )
    parser.add_argument(
        "--min-symbols-per-group",
        type=int,
        default=1,
        help="Minimum number of unique symbols required to keep a (timeframe, horizon) group.",
    )

    args = parser.parse_args()
    pvalue_threshold = args.pvalue_threshold if args.pvalue_threshold >= 0 else None

    return SelectorConfig(
        details_csv=args.details_csv,
        output_dir=args.output_dir,
        pearson_threshold=args.pearson_threshold,
        pvalue_threshold=pvalue_threshold,
        min_samples=args.min_samples,
        top_features_per_symbol=args.top_features_per_symbol,
        top_features_per_group=args.top_features_per_group,
        config_filename=args.config_filename,
        summary_filename=args.summary_filename,
        json_filename=args.json_filename,
        min_symbols_per_group=args.min_symbols_per_group,
    )


def load_and_filter(config: SelectorConfig) -> pd.DataFrame:
    df = pd.read_csv(config.details_csv)
    required_columns = {
        "symbol",
        "timeframe",
        "forward_bars",
        "feature",
        "pearson_corr",
        "pearson_p",
        "spearman_corr",
        "spearman_p",
        "samples",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Details CSV missing expected columns: {missing}")

    df["forward_bars"] = df["forward_bars"].astype(int)
    df["samples"] = df["samples"].astype(int)
    df["abs_pearson"] = df["pearson_corr"].abs()

    mask = df["abs_pearson"] >= config.pearson_threshold
    mask &= df["samples"] >= config.min_samples
    if config.pvalue_threshold is not None:
        mask &= df["pearson_p"] <= config.pvalue_threshold

    filtered = df[mask].copy()
    filtered.sort_values(
        ["symbol", "timeframe", "forward_bars", "abs_pearson"],
        ascending=[True, True, True, False],
        inplace=True,
    )

    if filtered.empty:
        raise ValueError(
            "No records passed the filtering criteria. "
            "Consider lowering the pearson or sample thresholds."
        )

    top_per_symbol = (
        filtered.groupby(["symbol", "timeframe", "forward_bars"])
        .head(config.top_features_per_symbol)
        .reset_index(drop=True)
    )
    return top_per_symbol


def group_by_timeframe(top_df: pd.DataFrame, config: SelectorConfig) -> List[Dict]:
    groups: List[Dict] = []

    grouped = top_df.groupby(["timeframe", "forward_bars"])
    for (timeframe, forward), group in grouped:
        symbols = sorted(group["symbol"].unique())
        if len(symbols) < config.min_symbols_per_group:
            continue

        features_ranked = (
            group.groupby("feature")["abs_pearson"]
            .mean()
            .sort_values(ascending=False)
            .head(config.top_features_per_group)
        )

        feature_by_symbol: Dict[str, List[str]] = {}
        for symbol, sub in group.groupby("symbol"):
            feature_by_symbol[symbol] = (
                sub.sort_values("abs_pearson", ascending=False)["feature"]
                .head(config.top_features_per_symbol)
                .tolist()
            )
        feature_by_symbol = {
            symbol: feature_by_symbol[symbol]
            for symbol in sorted(feature_by_symbol.keys())
        }

        group_entry = {
            "name": f"{timeframe.lower()}_{forward}b",
            "timeframe": timeframe,
            "forward_bars": int(forward),
            "symbols": symbols,
            "shared_features": features_ranked.index.tolist(),
            "features_by_symbol": feature_by_symbol,
            "label_expression": f"Ref($close, -{forward}) / $close - 1",
            "metrics": {
                "mean_abs_pearson": round(group["abs_pearson"].mean(), 4),
                "max_abs_pearson": round(group["abs_pearson"].max(), 4),
                "min_abs_pearson": round(group["abs_pearson"].min(), 4),
                "mean_samples": int(group["samples"].mean()),
            },
        }
        groups.append(group_entry)

    groups.sort(
        key=lambda g: (
            g["timeframe"],
            g["forward_bars"],
            -g["metrics"]["mean_abs_pearson"],
        )
    )
    return groups


def render_yaml(groups: List[Dict], min_abs: float) -> str:
    lines: List[str] = []
    lines.append("groups:")
    for group in groups:
        lines.append(f"  - name: \"{group['name']}\"")
        lines.append(f"    timeframe: \"{group['timeframe']}\"")
        lines.append(f"    forward_bars: {group['forward_bars']}")
        symbols = ", ".join(f'"{sym}"' for sym in group["symbols"])
        lines.append(f"    symbols: [{symbols}]")
        shared = ", ".join(f'"{feat}"' for feat in group["shared_features"])
        lines.append(f"    shared_features: [{shared}]")
        lines.append("    features_by_symbol:")
        for symbol in sorted(group["features_by_symbol"].keys()):
            feats = group["features_by_symbol"][symbol]
            feats_str = ", ".join(f'"{feat}"' for feat in feats)
            lines.append(f"      {symbol}: [{feats_str}]")
        metrics = group["metrics"]
        lines.append(f"    label_expr: \"{group['label_expression']}\"")
        lines.append(f"    min_abs_pearson: {min_abs}")
        lines.append("    metrics:")
        lines.append(f"      mean_abs_pearson: {metrics['mean_abs_pearson']}")
        lines.append(f"      max_abs_pearson: {metrics['max_abs_pearson']}")
        lines.append(f"      min_abs_pearson: {metrics['min_abs_pearson']}")
        lines.append(f"      mean_samples: {metrics['mean_samples']}")
    return "\n".join(lines) + "\n"


def write_summary_markdown(
    output_path: Path,
    config: SelectorConfig,
    groups: List[Dict],
    raw_df: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as md:
        md.write("# Strategy Group Summary\n\n")
        md.write(f"- Source CSV: `{config.details_csv}`\n")
        md.write(f"- Pearson threshold: {config.pearson_threshold}\n")
        md.write(f"- Min samples: {config.min_samples}\n")
        if config.pvalue_threshold is not None:
            md.write(f"- P-value threshold: {config.pvalue_threshold}\n")
        md.write(f"- Total qualifying records: {len(raw_df)}\n\n")

        if not groups:
            md.write("No groups generated with current filters.\n")
            return

        for group in groups:
            md.write(f"## Group `{group['name']}`\n\n")
            md.write(
                f"- Timeframe: `{group['timeframe']}`  |  Forward Bars: `{group['forward_bars']}`\n"
            )
            md.write(f"- Symbols: {', '.join(group['symbols'])}\n")
            shared = ", ".join(group["shared_features"])
            md.write(
                f"- Shared features (top {len(group['shared_features'])}): {shared}\n"
            )
            metrics = group["metrics"]
            md.write(
                f"- mean |ρ|: {metrics['mean_abs_pearson']:.4f}, "
                f"max |ρ|: {metrics['max_abs_pearson']:.4f}, "
                f"mean samples: {metrics['mean_samples']}\n"
            )
            md.write("- Features by symbol:\n")
            for symbol in sorted(group["features_by_symbol"].keys()):
                feats = group["features_by_symbol"][symbol]
                md.write(f"  - {symbol}: {', '.join(feats)}\n")
            md.write("\n")

        md.write("## Next Steps\n\n")
        md.write(
            "- Train separate models per group to avoid feature leakage across regimes.\n"
        )
        md.write(
            "- Use the generated YAML as a training manifest to align data loaders.\n"
        )
        md.write(
            "- Monitor group-wise IC/IR; pause a group if its metrics degrade materially.\n"
        )


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    filtered = load_and_filter(config)
    groups = group_by_timeframe(filtered, config)

    yaml_text = render_yaml(groups, config.pearson_threshold)
    yaml_path = config.output_dir / config.config_filename
    yaml_path.write_text(yaml_text, encoding="utf-8")

    json_path = config.output_dir / config.json_filename
    json_payload = {"groups": groups, "pearson_threshold": config.pearson_threshold}
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    summary_path = config.output_dir / config.summary_filename
    write_summary_markdown(summary_path, config, groups, filtered)

    print(f"Saved YAML config to {yaml_path}")
    print(f"Saved JSON manifest to {json_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
