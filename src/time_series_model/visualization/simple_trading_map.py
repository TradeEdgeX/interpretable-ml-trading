"""Create simple trading map visualization without emoji issues."""

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta


def create_simple_trading_map():
    """Create simple trading map visualization."""
    print("Creating Trading Map Visualization")
    print("=" * 50)

    # Load trading data
    try:
        trades_df = pd.read_csv("improved_features_vectorbot_trades.csv")
        equity_df = pd.read_csv("improved_features_vectorbot_equity_curve.csv")

        # Convert timestamps
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])
        equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"])

        print(f"Loaded {len(trades_df)} trades and {len(equity_df)} equity points")

    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        return

    # Create comprehensive trading map
    fig = plt.figure(figsize=(20, 16))

    # 1. Main price chart with trades (top-left)
    ax1 = plt.subplot(3, 3, 1)

    # Plot price data
    price_data = equity_df.set_index("timestamp")
    ax1.plot(
        price_data.index,
        price_data["equity"],
        linewidth=2,
        alpha=0.8,
        color="blue",
        label="Equity Curve",
    )
    ax1.axhline(
        y=100000, color="red", linestyle="--", alpha=0.7, label="Initial Capital"
    )

    # Plot trade entries and exits
    for idx, trade in trades_df.iterrows():
        entry_time = trade["entry_time"]
        exit_time = trade["exit_time"]
        entry_price = trade["entry_price"]
        exit_price = trade["exit_price"]
        pnl = trade["pnl"]

        # Color based on P&L
        color = "green" if pnl > 0 else "red"
        alpha = 0.7 if pnl > 0 else 0.5

        # Plot entry point
        ax1.scatter(
            entry_time,
            entry_price,
            color=color,
            s=50,
            alpha=alpha,
            marker="^" if trade["side"] == "long" else "v",
        )

        # Plot exit point
        ax1.scatter(exit_time, exit_price, color=color, s=50, alpha=alpha, marker="o")

        # Draw trade line
        ax1.plot(
            [entry_time, exit_time],
            [entry_price, exit_price],
            color=color,
            alpha=alpha,
            linewidth=1,
        )

    ax1.set_title(
        "Trading Map: Price Chart with Trades", fontsize=14, fontweight="bold"
    )
    ax1.set_ylabel("Price ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Trade distribution by time (top-center)
    ax2 = plt.subplot(3, 3, 2)

    # Create hourly trade distribution
    trades_df["hour"] = trades_df["entry_time"].dt.hour
    hourly_trades = trades_df.groupby("hour").size()

    bars = ax2.bar(
        hourly_trades.index, hourly_trades.values, alpha=0.7, color="skyblue"
    )
    ax2.set_title("Trade Distribution by Hour", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Number of Trades")
    ax2.grid(True, alpha=0.3)

    # 3. P&L distribution (top-right)
    ax3 = plt.subplot(3, 3, 3)

    # Create P&L histogram
    pnl_values = trades_df["pnl"].values
    ax3.hist(pnl_values, bins=20, alpha=0.7, color="lightcoral", edgecolor="black")
    ax3.axvline(x=0, color="red", linestyle="--", alpha=0.8, linewidth=2)
    ax3.set_title("P&L Distribution", fontsize=14, fontweight="bold")
    ax3.set_xlabel("P&L ($)")
    ax3.set_ylabel("Frequency")
    ax3.grid(True, alpha=0.3)

    # Add statistics
    mean_pnl = np.mean(pnl_values)
    ax3.axvline(
        x=mean_pnl,
        color="blue",
        linestyle="-",
        alpha=0.8,
        linewidth=2,
        label=f"Mean: ${mean_pnl:.2f}",
    )
    ax3.legend()

    # 4. Trade duration analysis (middle-left)
    ax4 = plt.subplot(3, 3, 4)

    # Plot trade duration vs P&L
    ax4.scatter(
        trades_df["duration"],
        trades_df["pnl"],
        c=trades_df["pnl"],
        cmap="RdYlGn",
        alpha=0.7,
        s=60,
    )
    ax4.axhline(y=0, color="red", linestyle="--", alpha=0.8)
    ax4.set_title("Trade Duration vs P&L", fontsize=14, fontweight="bold")
    ax4.set_xlabel("Duration (minutes)")
    ax4.set_ylabel("P&L ($)")
    ax4.grid(True, alpha=0.3)

    # 5. Win/Loss by exit reason (middle-center)
    ax5 = plt.subplot(3, 3, 5)

    # Analyze exit reasons
    exit_reasons = trades_df["reason"].value_counts()
    win_loss_by_reason = (
        trades_df.groupby("reason")
        .agg({"pnl": ["count", lambda x: (x > 0).sum(), lambda x: (x < 0).sum()]})
        .round(2)
    )

    # Create exit reason analysis
    exit_reasons_df = pd.DataFrame(
        {
            "Total": exit_reasons,
            "Wins": [
                trades_df[trades_df["reason"] == reason]["pnl"].gt(0).sum()
                for reason in exit_reasons.index
            ],
            "Losses": [
                trades_df[trades_df["reason"] == reason]["pnl"].lt(0).sum()
                for reason in exit_reasons.index
            ],
        }
    )

    # Create stacked bar chart
    x = np.arange(len(exit_reasons_df))
    width = 0.35

    bars1 = ax5.bar(
        x - width / 2,
        exit_reasons_df["Wins"],
        width,
        label="Wins",
        color="green",
        alpha=0.7,
    )
    bars2 = ax5.bar(
        x + width / 2,
        exit_reasons_df["Losses"],
        width,
        label="Losses",
        color="red",
        alpha=0.7,
    )

    ax5.set_title("Win/Loss by Exit Reason", fontsize=14, fontweight="bold")
    ax5.set_xlabel("Exit Reason")
    ax5.set_ylabel("Number of Trades")
    ax5.set_xticks(x)
    ax5.set_xticklabels(exit_reasons_df.index, rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Equity curve with drawdown (middle-right)
    ax6 = plt.subplot(3, 3, 6)

    # Calculate drawdown
    equity_series = equity_df["equity"]
    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak * 100

    # Plot equity and drawdown
    ax6_twin = ax6.twinx()

    ax6.plot(
        equity_df["timestamp"], equity_series, linewidth=2, color="blue", label="Equity"
    )
    ax6_twin.fill_between(
        equity_df["timestamp"], drawdown, 0, alpha=0.3, color="red", label="Drawdown"
    )

    ax6.set_title("Equity Curve with Drawdown", fontsize=14, fontweight="bold")
    ax6.set_ylabel("Equity ($)", color="blue")
    ax6_twin.set_ylabel("Drawdown (%)", color="red")
    ax6.grid(True, alpha=0.3)

    # 7. Trade size analysis (bottom-left)
    ax7 = plt.subplot(3, 3, 7)

    # Plot trade size vs P&L
    ax7.scatter(
        trades_df["size"],
        trades_df["pnl"],
        c=trades_df["pnl"],
        cmap="RdYlGn",
        alpha=0.7,
        s=60,
    )
    ax7.axhline(y=0, color="red", linestyle="--", alpha=0.8)
    ax7.set_title("Trade Size vs P&L", fontsize=14, fontweight="bold")
    ax7.set_xlabel("Trade Size (units)")
    ax7.set_ylabel("P&L ($)")
    ax7.grid(True, alpha=0.3)

    # 8. Price level performance (bottom-center)
    ax8 = plt.subplot(3, 3, 8)

    # Create price level analysis
    price_ranges = [
        (90000, 95000, "Low"),
        (95000, 100000, "Medium-Low"),
        (100000, 105000, "Medium"),
        (105000, 110000, "Medium-High"),
        (110000, 115000, "High"),
    ]

    range_data = []
    for low, high, label in price_ranges:
        range_trades = trades_df[
            (trades_df["entry_price"] >= low) & (trades_df["entry_price"] < high)
        ]
        if len(range_trades) > 0:
            range_data.append(
                {
                    "Range": label,
                    "Count": len(range_trades),
                    "Avg_PnL": range_trades["pnl"].mean(),
                }
            )

    if range_data:
        range_df = pd.DataFrame(range_data)
        bars = ax8.bar(
            range_df["Range"],
            range_df["Avg_PnL"],
            color=["red" if x < 0 else "green" for x in range_df["Avg_PnL"]],
            alpha=0.7,
        )
        ax8.set_title("Average P&L by Price Level", fontsize=14, fontweight="bold")
        ax8.set_xlabel("Price Range")
        ax8.set_ylabel("Average P&L ($)")
        ax8.axhline(y=0, color="black", linestyle="-", alpha=0.5)
        ax8.grid(True, alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax8.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + (1 if height >= 0 else -3),
                f"${height:.1f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
            )

    # 9. Performance metrics summary (bottom-right)
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis("off")

    # Calculate key metrics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df["pnl"] > 0])
    losing_trades = len(trades_df[trades_df["pnl"] < 0])
    win_rate = winning_trades / total_trades * 100

    total_pnl = trades_df["pnl"].sum()
    avg_win = trades_df[trades_df["pnl"] > 0]["pnl"].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df["pnl"] < 0]["pnl"].mean() if losing_trades > 0 else 0

    max_drawdown = drawdown.min()
    best_trade = trades_df["pnl"].max()
    worst_trade = trades_df["pnl"].min()

    # Create metrics text
    metrics_text = f"""
    TRADING PERFORMANCE SUMMARY
    
    Basic Metrics:
    • Total Trades: {total_trades}
    • Win Rate: {win_rate:.1f}%
    • Total P&L: ${total_pnl:.2f}
    
    P&L Analysis:
    • Average Win: ${avg_win:.2f}
    • Average Loss: ${avg_loss:.2f}
    • Best Trade: ${best_trade:.2f}
    • Worst Trade: ${worst_trade:.2f}
    
    Risk Metrics:
    • Max Drawdown: {max_drawdown:.2f}%
    • Profit Factor: {abs(avg_win * winning_trades / (avg_loss * losing_trades)):.2f}
    
    Time Analysis:
    • Avg Duration: {trades_df['duration'].mean():.0f} min
    • Longest Trade: {trades_df['duration'].max():.0f} min
    • Shortest Trade: {trades_df['duration'].min():.0f} min
    """

    ax9.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax9.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig("simple_trading_map.png", dpi=300, bbox_inches="tight")

    print(f"Trading map saved to 'simple_trading_map.png'")

    # Print detailed analysis
    print_detailed_analysis(trades_df, equity_df)


def print_detailed_analysis(trades_df, equity_df):
    """Print detailed trading analysis."""
    print("\nDetailed Trading Analysis:")

    # Time-based analysis
    print(f"\nTime Analysis:")
    trades_df["hour"] = trades_df["entry_time"].dt.hour
    hourly_performance = (
        trades_df.groupby("hour")
        .agg({"pnl": ["count", "sum", "mean"], "duration": "mean"})
        .round(2)
    )

    print(f"   Best trading hours:")
    best_hours = hourly_performance.sort_values(("pnl", "sum"), ascending=False).head(3)
    for hour, data in best_hours.iterrows():
        print(
            f"     {hour:02d}:00 - {data[('pnl', 'count')]} trades, ${data[('pnl', 'sum')]:.2f} P&L"
        )

    # Price level analysis
    print(f"\nPrice Level Analysis:")
    price_ranges = [
        (90000, 95000, "Low"),
        (95000, 100000, "Medium-Low"),
        (100000, 105000, "Medium"),
        (105000, 110000, "Medium-High"),
        (110000, 115000, "High"),
    ]

    for low, high, label in price_ranges:
        range_trades = trades_df[
            (trades_df["entry_price"] >= low) & (trades_df["entry_price"] < high)
        ]
        if len(range_trades) > 0:
            avg_pnl = range_trades["pnl"].mean()
            count = len(range_trades)
            print(
                f"     {label} ({low}-{high}): {count} trades, avg P&L: ${avg_pnl:.2f}"
            )

    # Trade sequence analysis
    print(f"\nTrade Sequence Analysis:")
    consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0

    for pnl in trades_df["pnl"]:
        if pnl > 0:
            consecutive_wins += 1
            consecutive_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
        else:
            consecutive_losses += 1
            consecutive_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

    print(f"     Max consecutive wins: {max_consecutive_wins}")
    print(f"     Max consecutive losses: {max_consecutive_losses}")

    # Risk analysis
    print(f"\nRisk Analysis:")
    daily_returns = equity_df["equity"].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252) * 100
    sharpe_ratio = (
        daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        if daily_returns.std() > 0
        else 0
    )

    print(f"     Annualized volatility: {volatility:.2f}%")
    print(f"     Sharpe ratio: {sharpe_ratio:.2f}")

    # Exit reason analysis
    print(f"\nExit Reason Analysis:")
    exit_analysis = (
        trades_df.groupby("reason")
        .agg({"pnl": ["count", "sum", "mean"], "duration": "mean"})
        .round(2)
    )

    for reason in exit_analysis.index:
        count = exit_analysis.loc[reason, ("pnl", "count")]
        total_pnl = exit_analysis.loc[reason, ("pnl", "sum")]
        avg_pnl = exit_analysis.loc[reason, ("pnl", "mean")]
        avg_duration = exit_analysis.loc[reason, ("duration", "mean")]
        print(
            f"     {reason}: {count} trades, ${total_pnl:.2f} total, ${avg_pnl:.2f} avg, {avg_duration:.0f} min avg"
        )

    print(f"\nAnalysis completed!")


if __name__ == "__main__":
    create_simple_trading_map()
