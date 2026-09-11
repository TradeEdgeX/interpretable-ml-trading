# A strategy is a contract, not a feeling

**One line:** A strategy must state when trading is allowed, how you exit when wrong, what you risk, and when the trade ends. “Buy the golden cross” alone is not a strategy.

## Wrong vs right

| Wrong | Right |
|---|---|
| “Buy the MA cross; trends should pay.” | “Long only if close is above the 50 and just crossed the 200; exit if it breaks the 50; the sentence dies if any window’s CAGR is negative or range-year drawdown is worse.” |

Feelings cannot be audited or falsified. Contracts can.

## In this repo

The MA-cross practice sentence fills mechanism, regimes, contract, falsification, and deployment. The robot YAML and a human checklist are **the same sentence** — if the backtest rejects it, you do not hand-trade “this time is different” at night.

See [Five boxes](../design/five-boxes.md) · full example in [README.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/README.md).

## Deeper docs

- [Hypothesis template](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/hypothesis_template.en.md)
- [Philosophy](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/philosophy.en.md)
