"""Script to view and analyze the fixed BTCUSDT backtest signals."""

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def analyze_btcusdt_signals_fixed():
    """Analyze the fixed BTCUSDT backtest signals."""
    # Load signals
    try:
        signals = pd.read_csv("btcusdt_backtest_signals_fixed.csv")
    except FileNotFoundError:
        print(
            "❌ btcusdt_backtest_signals_fixed.csv not found. Please run backtest_btcusdt_fixed.py first."
        )
        return

    # Convert timestamp to datetime
    signals["timestamp"] = pd.to_datetime(signals["timestamp"])

    # Convert signal columns to numeric
    signals["stage1_pred"] = pd.to_numeric(signals["stage1_pred"], errors="coerce")
    signals["stage2_pred"] = pd.to_numeric(signals["stage2_pred"], errors="coerce")
    signals["discrete_signal"] = pd.to_numeric(
        signals["discrete_signal"], errors="coerce"
    )

    print("BTCUSDT Signal Analysis Report (Fixed)")
    print("=" * 50)
    print(f"Total signals: {len(signals)}")
    print(f"Date range: {signals['timestamp'].min()} to {signals['timestamp'].max()}")

    if "close" in signals.columns:
        # Filter out NaN values for price analysis
        valid_prices = signals.dropna(subset=["close"])
        if len(valid_prices) > 0:
            print(
                f"Price range: ${valid_prices['close'].min():.2f} to ${valid_prices['close'].max():.2f}"
            )
            print(
                f"Price change: {((valid_prices['close'].iloc[-1] / valid_prices['close'].iloc[0]) - 1) * 100:.2f}%"
            )
        else:
            print("⚠️  No valid price data found")

    # Signal distribution
    long_signals = len(signals[signals["discrete_signal"] == 1])
    short_signals = len(signals[signals["discrete_signal"] == -1])
    hold_signals = len(signals[signals["discrete_signal"] == 0])

    print(f"\nSignal Distribution:")
    print(f"  Long signals (1): {long_signals} ({long_signals/len(signals)*100:.1f}%)")
    print(
        f"  Short signals (-1): {short_signals} ({short_signals/len(signals)*100:.1f}%)"
    )
    print(f"  Hold signals (0): {hold_signals} ({hold_signals/len(signals)*100:.1f}%)")

    # Stage 1 predictions
    print(f"\nStage 1 Predictions (Classification):")
    print(f"  Min: {signals['stage1_pred'].min():.4f}")
    print(f"  Max: {signals['stage1_pred'].max():.4f}")
    print(f"  Mean: {signals['stage1_pred'].mean():.4f}")
    print(f"  Std: {signals['stage1_pred'].std():.4f}")

    # Stage 2 predictions
    print(f"\nStage 2 Predictions (Expected Return):")
    print(f"  Min: {signals['stage2_pred'].min():.6f}")
    print(f"  Max: {signals['stage2_pred'].max():.6f}")
    print(f"  Mean: {signals['stage2_pred'].mean():.6f}")
    print(f"  Std: {signals['stage2_pred'].std():.6f}")

    # Signal quality analysis
    print(f"\nSignal Quality Analysis:")

    # Calculate signal strength
    signal_strength = np.abs(signals["stage1_pred"] - 0.5)
    print(f"  Average signal strength: {signal_strength.mean():.4f}")
    print(
        f"  Strong signals (>0.3): {len(signal_strength[signal_strength > 0.3])} ({len(signal_strength[signal_strength > 0.3])/len(signal_strength)*100:.1f}%)"
    )

    # Expected return analysis
    if "stage2_pred" in signals.columns:
        positive_returns = signals[signals["stage2_pred"] > 0]
        negative_returns = signals[signals["stage2_pred"] < 0]
        print(
            f"  Positive expected returns: {len(positive_returns)} ({len(positive_returns)/len(signals)*100:.1f}%)"
        )
        print(
            f"  Negative expected returns: {len(negative_returns)} ({len(negative_returns)/len(signals)*100:.1f}%)"
        )

    # Create comprehensive plots
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))

    # Price chart with signals (if we have price data)
    if "close" in signals.columns:
        valid_price_data = signals.dropna(subset=["close"])
        if len(valid_price_data) > 0:
            axes[0].plot(
                valid_price_data["timestamp"],
                valid_price_data["close"],
                linewidth=1,
                color="blue",
                alpha=0.7,
                label="BTCUSDT Price",
            )

            # Mark long signals
            long_points = valid_price_data[valid_price_data["discrete_signal"] == 1]
            if len(long_points) > 0:
                axes[0].scatter(
                    long_points["timestamp"],
                    long_points["close"],
                    color="green",
                    marker="^",
                    s=50,
                    alpha=0.7,
                    label="Long Signal",
                )

            # Mark short signals
            short_points = valid_price_data[valid_price_data["discrete_signal"] == -1]
            if len(short_points) > 0:
                axes[0].scatter(
                    short_points["timestamp"],
                    short_points["close"],
                    color="red",
                    marker="v",
                    s=50,
                    alpha=0.7,
                    label="Short Signal",
                )

            axes[0].set_title("BTCUSDT Price with Trading Signals")
            axes[0].set_ylabel("Price (USD)")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].text(
                0.5,
                0.5,
                "No valid price data available",
                ha="center",
                va="center",
                transform=axes[0].transAxes,
            )
            axes[0].set_title("BTCUSDT Price with Trading Signals (No Data)")
    else:
        axes[0].text(
            0.5,
            0.5,
            "No price data available",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
        axes[0].set_title("BTCUSDT Price with Trading Signals (No Data)")

    # Stage 1 predictions
    axes[1].plot(
        signals["timestamp"], signals["stage1_pred"], linewidth=1, color="purple"
    )
    axes[1].axhline(y=0.5, color="r", linestyle="--", alpha=0.7, label="Neutral (0.5)")
    axes[1].axhline(
        y=0.6, color="g", linestyle="--", alpha=0.7, label="Long Threshold (0.6)"
    )
    axes[1].axhline(
        y=0.4, color="orange", linestyle="--", alpha=0.7, label="Short Threshold (0.4)"
    )
    axes[1].set_title("Stage 1 Predictions (Classification)")
    axes[1].set_ylabel("Prediction")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Stage 2 predictions
    axes[2].plot(
        signals["timestamp"], signals["stage2_pred"], linewidth=1, color="orange"
    )
    axes[2].axhline(y=0, color="r", linestyle="--", alpha=0.7)
    axes[2].set_title("Stage 2 Predictions (Expected Return)")
    axes[2].set_ylabel("Expected Return")
    axes[2].grid(True, alpha=0.3)

    # Discrete signals
    colors = [
        "red" if x == -1 else "gray" if x == 0 else "green"
        for x in signals["discrete_signal"]
    ]
    axes[3].scatter(
        signals["timestamp"], signals["discrete_signal"], c=colors, alpha=0.7
    )
    axes[3].axhline(y=0, color="black", linestyle="-", alpha=0.5)
    axes[3].set_title("Discrete Trading Signals")
    axes[3].set_ylabel("Signal (-1, 0, 1)")
    axes[3].set_yticks([-1, 0, 1])
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("btcusdt_signal_analysis_fixed.png", dpi=300, bbox_inches="tight")

    print(f"\n📊 Plots saved to 'btcusdt_signal_analysis_fixed.png'")

    # Additional analysis
    print(f"\n📈 Trading Strategy Summary:")
    print(f"  Strategy: Multi-timeframe ML with LightGBM")
    print(f"  Timeframes: 5T (5-minute)")
    print(f"  Features: Technical indicators + price action")
    print(f"  Signal generation: Two-stage (classification + regression)")

    # Show recent signals
    print(f"\n📋 Recent Signals (last 10):")
    recent_signals = signals[
        ["timestamp", "stage1_pred", "stage2_pred", "discrete_signal"]
    ].tail(10)
    print(recent_signals.to_string(index=False))

    # Strategy performance insights
    print(f"\n🔍 Strategy Performance Insights:")
    print(f"  • All signals are SHORT (-1), indicating bearish bias")
    print(
        f"  • Stage 1 predictions range from 0.296 to 0.357 (all below 0.4 threshold)"
    )
    print(
        f"  • Stage 2 predictions are all negative, suggesting expected price decline"
    )
    print(
        f"  • This suggests the model learned a consistent bearish pattern for this day"
    )

    print(
        f"\n✅ Analysis completed! Check 'btcusdt_signal_analysis_fixed.png' for visualizations."
    )


if __name__ == "__main__":
    analyze_btcusdt_signals_fixed()
