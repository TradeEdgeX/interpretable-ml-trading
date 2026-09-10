#!/usr/bin/env python3
"""
Generate additional plateau visualizations (heatmaps & scatter) for SR rule optimization
and inject them into the optimization HTML report.
"""
from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_heatmap(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    title: str,
    output_path: Path,
) -> None:
    """Create heatmap by averaging value_col over parameter grid."""
    pivot = (
        df.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc=np.mean)
        .sort_index()
        .sort_index(axis=1)
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    cax = ax.imshow(pivot.values, cmap="YlGnBu", origin="lower", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{v:g}" for v in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{v:g}" for v in pivot.index])
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.set_title(title)
    fig.colorbar(cax, ax=ax, label=value_col)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _build_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    size_col: str,
    title: str,
    output_path: Path,
) -> None:
    """Create scatter plot highlighting high-performing plateau region."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sc = ax.scatter(
        df[x_col],
        df[y_col],
        c=df[color_col],
        s=40 + 60 * (df[size_col] / df[size_col].max()),
        cmap="viridis",
        alpha=0.7,
        edgecolor="k",
    )
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label=color_col)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _inject_section(html_path: Path, section_html: str) -> None:
    """Insert plateau section into HTML between marker comments."""
    MARKER_START = "<!-- PLATEAU_CHARTS_START -->"
    MARKER_END = "<!-- PLATEAU_CHARTS_END -->"

    html = html_path.read_text(encoding="utf-8")
    if MARKER_START in html and MARKER_END in html:
        prefix, _rest = html.split(MARKER_START, 1)
        _, suffix = _rest.split(MARKER_END, 1)
        html = prefix + suffix

    # insert before closing container/body
    insert_index = html.rfind("</div>")
    if insert_index == -1:
        insert_index = len(html)

    new_html = (
        html[:insert_index]
        + f"\n{MARKER_START}\n{section_html}\n{MARKER_END}\n"
        + html[insert_index:]
    )
    html_path.write_text(new_html, encoding="utf-8")


def build_section_html(image_info: List[Tuple[str, Path]]) -> str:
    blocks = []
    for title, path in image_info:
        blocks.append(
            f"""
        <div class="comparison-item" style="flex:1; margin:10px;">
            <h3 style="text-align:center;">{title}</h3>
            <img src="{path.as_posix()}" alt="{title}" style="width:100%; max-width:420px; border:1px solid #ddd; border-radius:6px; box-shadow:0 2px 4px rgba(0,0,0,0.1);" />
        </div>
        """
        )

    section = f"""
    <h2>🏞️ Plateau Heatmaps & Scatter</h2>
    <p>Additional visualizations of the rule-based parameter landscape (averaged Total&nbsp;R). Darker colors indicate higher Total&nbsp;R. Scatter highlights the top 20% configurations.</p>
    <div class="comparison-box" style="display:flex; flex-wrap:wrap; gap:10px;">
        {''.join(blocks)}
    </div>
    """
    return section


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate plateau charts for rule optimization report"
    )
    parser.add_argument(
        "--results-csv", required=True, help="Path to optimization_results.csv"
    )
    parser.add_argument(
        "--report-html", required=True, help="Path to optimization_report.html"
    )
    args = parser.parse_args()

    csv_path = Path(args.results_csv).resolve()
    report_path = Path(args.report_html).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"Report HTML not found: {report_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Optimization results CSV is empty")

    output_dir = report_path.parent / "plots"
    _ensure_dir(output_dir)

    # Heatmaps
    heatmaps = []
    hm_specs = [
        ("stop_loss_r", "take_profit_r", "Total R (avg)", "heatmap_sl_tp.png"),
        ("sr_strength_min", "sqs_min", "Total R (avg)", "heatmap_sr_sqs.png"),
        (
            "max_holding_bars",
            "touch_distance_atr",
            "Total R (avg)",
            "heatmap_hold_touch.png",
        ),
    ]
    for x_col, y_col, title_suffix, filename in hm_specs:
        if x_col in df.columns and y_col in df.columns:
            out_path = output_dir / filename
            _build_heatmap(
                df,
                x_col=x_col,
                y_col=y_col,
                value_col="total_r",
                title=f"{title_suffix}: {y_col} vs {x_col}",
                output_path=out_path,
            )
            heatmaps.append((f"{y_col} vs {x_col}", Path("plots") / filename))

    # Scatter for top 20%
    top_df = df[df["total_r"] >= df["total_r"].quantile(0.8)].copy()
    scatter_path = output_dir / "scatter_sl_tp_top20.png"
    if not top_df.empty:
        _build_scatter(
            top_df,
            x_col="stop_loss_r",
            y_col="take_profit_r",
            color_col="total_r",
            size_col="n_trades",
            title="Top 20% configs (Total R & trades)",
            output_path=scatter_path,
        )
        heatmaps.append(
            (
                "Top 20% Stop Loss vs Take Profit",
                Path("plots") / "scatter_sl_tp_top20.png",
            )
        )

    if not heatmaps:
        print("⚠️ No valid plots generated; required columns may be missing.")
        return

    section_html = build_section_html(heatmaps)
    _inject_section(report_path, section_html)
    print(f"✅ Generated {len(heatmaps)} plots and updated report: {report_path}")


if __name__ == "__main__":
    main()
