# What is a feature

**One line:** A feature is a precomputed, monthly-on-disk measurement — not an EMA invented inside the backtest loop.

## Wrong vs right

| Wrong | Right |
|---|---|
| Call `compute_ema()` ad hoc whenever the loop needs it. | Register columns, build them into the feature store; if a column is missing, backfill the same layer, then run the court. |

A hand-rolled MA in chat will not align with the store’s clock and cannot be audited.

## In this repo

The golden cross reads a stored column (e.g. `ema_50_200_cross_side`), not a screenshot “daily cross”. Backtests use strict store reads: missing column stops the run — no hot-path compute.

## Deeper docs

- [Features](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/features.en.md)
- [Feature store first](store-first.md)
