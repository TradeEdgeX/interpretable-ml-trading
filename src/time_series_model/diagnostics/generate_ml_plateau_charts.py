#!/usr/bin/env python3
from __future__ import annotations

"""
Generate plateau heatmaps/scatter plots for ML / ML+Vol parameter sweep
and attach them to the model comparison HTML report.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ML plateau charts")
    parser.add_argument(
        "--results-csv", required=True, help="Path to ml_param_sweep.csv"
    )
    parser.add_argument("--report-html", required=True, help="Comparison report HTML")
    parser.add_argument(
        "--methods",
        default="ml_model,ml_volatility",
        help="Comma-separated method names to include",
    )
    parser.add_argument(
        "--plots-subdir",
        default="ml_plots",
        help="Subdirectory under report folder for generated plots",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_heatmap(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    title: str,
    output_path: Path,
) -> None:
    pivot = (
        df.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc=np.mean)
        .sort_index()
        .sort_index(axis=1)
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    cax = ax.imshow(pivot.values, cmap="YlOrBr", origin="lower", aspect="auto")
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


def build_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    size_col: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sc = ax.scatter(
        df[x_col],
        df[y_col],
        c=df[color_col],
        s=40 + 60 * (df[size_col] / (df[size_col].max() or 1)),
        cmap="viridis",
        alpha=0.75,
        edgecolor="k",
    )
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label=color_col)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_section(title: str, image_info: List[Tuple[str, Path]]) -> str:
    blocks = []
    for img_title, rel_path in image_info:
        blocks.append(
            f"""
        <div class="comparison-item" style="flex:1; margin:10px;">
            <h3 style="text-align:center;">{img_title}</h3>
            <img src="{rel_path.as_posix()}" alt="{img_title}" style="width:100%; max-width:420px; border:1px solid #ddd; border-radius:6px; box-shadow:0 2px 4px rgba(0,0,0,0.1);" />
        </div>
        """
        )

    return f"""
    <h2>🏞️ {title}</h2>
    <p>Heatmaps show average Total&nbsp;R across parameter grids; scatter highlights the top 20% configurations by Total&nbsp;R.</p>
    <div class="comparison-box" style="display:flex; flex-wrap:wrap; gap:10px;">
        {''.join(blocks)}
    </div>
    """


def inject_section(report_path: Path, section_html: str) -> None:
    MARKER = "<!-- ML_PLATEAU_CHARTS_START -->"
    END_MARKER = "<!-- ML_PLATEAU_CHARTS_END -->"
    html = report_path.read_text(encoding="utf-8")
    if MARKER in html and END_MARKER in html:
        prefix, _rest = html.split(MARKER, 1)
        _, suffix = _rest.split(END_MARKER, 1)
        html = prefix + suffix
    insert_index = html.rfind("</div>")
    if insert_index == -1:
        insert_index = len(html)
    new_html = (
        html[:insert_index]
        + f"\n{MARKER}\n{section_html}\n{END_MARKER}\n"
        + html[insert_index:]
    )
    report_path.write_text(new_html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.results_csv).resolve()
    report_path = Path(args.report_html).resolve()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not report_path.exists():
        raise FileNotFoundError(report_path)

    df = pd.read_csv(csv_path)
    if "method" not in df.columns:
        raise ValueError("results CSV must contain a 'method' column")

    plots_dir = report_path.parent / args.plots_subdir
    ensure_dir(plots_dir)

    sections = []
    for method in methods:
        method_df = df[df["method"] == method]
        if method_df.empty:
            continue
        rel_images: List[Tuple[str, Path]] = []
        heatmap_specs = [
            ("stop_loss_r", "take_profit_r", "Total R", f"{method}_heatmap_sl_tp.png"),
            (
                "threshold",
                "stop_loss_r",
                "Total R",
                f"{method}_heatmap_threshold_sl.png",
            ),
        ]
        for x_col, y_col, val_name, filename in heatmap_specs:
            if x_col in method_df.columns and y_col in method_df.columns:
                out_path = plots_dir / filename
                build_heatmap(
                    method_df,
                    x_col=x_col,
                    y_col=y_col,
                    value_col="total_r",
                    title=f"{method}: {y_col} vs {x_col}",
                    output_path=out_path,
                )
                rel_images.append(
                    (
                        f"{method} – {y_col} vs {x_col}",
                        Path(args.plots_subdir) / filename,
                    )
                )

        top_df = method_df[method_df["total_r"] >= method_df["total_r"].quantile(0.8)]
        if not top_df.empty:
            scatter_path = plots_dir / f"{method}_scatter_top20.png"
            build_scatter(
                top_df,
                x_col="threshold",
                y_col="take_profit_r",
                color_col="total_r",
                size_col="n_trades",
                title=f"{method}: Top 20% threshold vs TP",
                output_path=scatter_path,
            )
            rel_images.append(
                (
                    f"{method} – Top 20% Threshold vs TP",
                    Path(args.plots_subdir) / scatter_path.name,
                )
            )

        if rel_images:
            sections.append(build_section(f"{method} Plateau Heatmaps", rel_images))

    if not sections:
        print("⚠️ No sections generated (check methods/results).")
        return

    inject_section(report_path, "\n".join(sections))
    print(f"✅ Injected ML plateau charts into {report_path}")


if __name__ == "__main__":
    main()
