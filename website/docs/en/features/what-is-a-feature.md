# What is a feature

**One-liner:** A feature is a pre-computed measurement, not a prediction. It only answers “on this bar, what do price / volatility / funding / order flow look like”.

## Wrong vs right

| Wrong | Right |
|---|---|
| “This indicator predicts up or down.” | “This feature describes the current state; whether it makes money depends on what contract you pair it with and which window you measure it on.” |

The same `ema_50_200_cross_side` (whether the 50-day line is above or below the 200-day line), paired with a trend contract, is one strategy; paired with a reversal contract, it is another. The feature itself does not take sides.

## How this repo uses it

Features live as columns in the feature store, monthly parquet on disk. For example, the node `ema_50_200_cross_f` produces the column `ema_50_200_cross_side`: +1 when the 50-day average is above the 200-day, −1 when below. At bar open the backtest reads the **previous** bar’s value of this column to decide whether trading is allowed.

The repo ships a base feature library (moving averages, crosses, funding, order flow, cross-sectional momentum, and more). You are invited to propose more testable ideas on top of those columns, and you can add new columns to the store following the [feature-store-first](feature-store-first.md) flow.

## Fine print

- [Five families](families.md)
- [Feature-store-first](feature-store-first.md)
- [Math is not a gate](math-is-not-a-gate.md)
- [How to read features.yaml](features-yaml.md)
