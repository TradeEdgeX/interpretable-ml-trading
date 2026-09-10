#!/usr/bin/env python3
"""
生成1h和4h时间框架的综合对比报告
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate timeframe comparison report (1h vs 4h)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/model_comparison",
        help="Output directory for the comparison report",
    )
    parser.add_argument(
        "--results-1h",
        type=str,
        default="results/model_comparison/comparison_results.csv",
        help="Path to 1h timeframe results CSV",
    )
    parser.add_argument(
        "--results-4h",
        type=str,
        default="results/model_comparison_240h/comparison_results.csv",
        help="Path to 4h timeframe results CSV",
    )
    return parser.parse_args()


def load_results(csv_path: str) -> pd.DataFrame:
    """Load results from CSV file."""
    df = pd.read_csv(csv_path)
    return df


def generate_html_report(
    df_1h: pd.DataFrame, df_4h: pd.DataFrame, output_path: Path
) -> None:
    """Generate HTML comparison report."""

    # Extract metrics for each model
    def get_metrics(df: pd.DataFrame, model_name: str) -> dict:
        row = df[df["Method"] == model_name].iloc[0]
        return {
            "trades": int(row["Trades"]),
            "win_rate": row["Win Rate"] * 100,
            "breakeven_rate": row["Breakeven Rate"] * 100,
            "total_r": row["Total R"],
            "avg_r": row["Avg R"],
            "sharpe": row["Sharpe Ratio"],
        }

    models = ["Rule-Based", "ML Model", "ML + Volatility Model"]

    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>SR Reversal Timeframe Comparison: 1h vs 4h</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 30px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .positive {
            color: #4CAF50;
            font-weight: bold;
        }
        .negative {
            color: #f44336;
            font-weight: bold;
        }
        .best {
            background-color: #e8f5e9;
            font-weight: bold;
        }
        .timeframe-header {
            background-color: #2196F3;
            color: white;
        }
        .metric-name {
            font-weight: bold;
            background-color: #f9f9f9;
        }
        .comparison-cell {
            text-align: center;
        }
        .better {
            background-color: #c8e6c9;
        }
        .worse {
            background-color: #ffcdd2;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 SR Reversal Timeframe Comparison: 1h vs 4h</h1>
        
        <h2>🎯 Performance Comparison by Timeframe</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Metric</th>
                    <th class="timeframe-header">1h (60T)</th>
                    <th class="timeframe-header">4h (240T)</th>
                    <th>Winner</th>
                </tr>
            </thead>
            <tbody>
"""

    for model in models:
        m1h = get_metrics(df_1h, model)
        m4h = get_metrics(df_4h, model)

        # Determine winner for each metric
        def compare_metric(
            name: str, val1h: float, val4h: float, higher_better: bool = True
        ) -> tuple:
            if higher_better:
                winner = "1h" if val1h > val4h else "4h"
                better_class = "better" if val1h > val4h else "worse"
            else:
                winner = "1h" if val1h < val4h else "4h"
                better_class = "better" if val1h < val4h else "worse"
            return winner, better_class

        metrics_data = [
            ("Trades", m1h["trades"], m4h["trades"], True),
            ("Win Rate", m1h["win_rate"], m4h["win_rate"], True),
            ("Breakeven Rate", m1h["breakeven_rate"], m4h["breakeven_rate"], True),
            ("Total R", m1h["total_r"], m4h["total_r"], True),
            ("Avg R per Trade", m1h["avg_r"], m4h["avg_r"], True),
            ("Sharpe Ratio", m1h["sharpe"], m4h["sharpe"], True),
        ]

        for i, (metric_name, val1h, val4h, higher_better) in enumerate(metrics_data):
            winner, better_class = compare_metric(
                metric_name, val1h, val4h, higher_better
            )

            # Format values
            if "Rate" in metric_name:
                val1h_str = f"{val1h:.2f}%"
                val4h_str = f"{val4h:.2f}%"
            elif "Sharpe" in metric_name:
                val1h_str = f"{val1h:.2f}"
                val4h_str = f"{val4h:.2f}"
            elif "R" in metric_name:
                val1h_str = f"{val1h:.2f}"
                val4h_str = f"{val4h:.2f}"
            else:
                val1h_str = str(val1h)
                val4h_str = str(val4h)

            # Add class for better/worse
            class1h = better_class if winner == "1h" else ""
            class4h = better_class if winner == "4h" else ""

            html += f"""
                <tr>
                    <td class="metric-name">{model if i == 0 else ""}</td>
                    <td>{metric_name}</td>
                    <td class="comparison-cell {class1h}">{val1h_str}</td>
                    <td class="comparison-cell {class4h}">{val4h_str}</td>
                    <td class="comparison-cell"><strong>{winner.upper()}</strong></td>
                </tr>
"""

    html += """
            </tbody>
        </table>
        
        <h2>📈 Key Insights</h2>
        <ul>
            <li><strong>Trade Frequency:</strong> 1h timeframe generates significantly more trades (4-17x more depending on model)</li>
            <li><strong>Win Rate:</strong> 4h timeframe shows higher win rates across all models</li>
            <li><strong>Sharpe Ratio:</strong> 4h timeframe demonstrates superior risk-adjusted returns, especially for ML+Volatility model (2.36 vs 0.33)</li>
            <li><strong>Total R:</strong> Mixed results - Rule-based performs better on 4h, but ML+Volatility shows positive returns on both timeframes</li>
            <li><strong>ML Model Filtering:</strong> Both timeframes show effective signal filtering (1h: 658→141, 4h: 136→39)</li>
        </ul>
        
        <h2>💡 Recommendations</h2>
        <ul>
            <li><strong>For Higher Frequency Trading:</strong> Use 1h timeframe with ML+Volatility model (46% win rate, positive Total R)</li>
            <li><strong>For Better Risk-Adjusted Returns:</strong> Use 4h timeframe with ML+Volatility model (92% win rate, Sharpe 2.36)</li>
            <li><strong>For Rule-Based Strategy:</strong> 4h timeframe is clearly superior (51% win rate vs 21%, positive Total R)</li>
        </ul>
    </div>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"✅ Comparison report saved to {output_path}")


def main() -> None:
    args = parse_args()

    # Load results
    print(f"📊 Loading 1h results from {args.results_1h}...")
    df_1h = load_results(args.results_1h)

    print(f"📊 Loading 4h results from {args.results_4h}...")
    df_4h = load_results(args.results_4h)

    # Generate report
    output_path = Path(args.output_dir) / "timeframe_comparison.html"
    print(f"📝 Generating comparison report...")
    generate_html_report(df_1h, df_4h, output_path)

    print("✅ Done!")


if __name__ == "__main__":
    main()
