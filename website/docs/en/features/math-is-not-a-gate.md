# Math is not a gate

**One-liner:** You don’t need stochastic processes or PDEs to use this repo. You need “can read a five-KPI table” and “can write an idea as a contract”.

## Wrong vs right

| Wrong | Right |
|---|---|
| “My math is bad, I can’t do quant.” | “I can read CAGR, drawdown, and win rate, so I can judge whether this sentence loses money in the recent window.” |

The math in this repo mostly hides in two places: how features are computed (`docs/math.md`), and how the backtest fills trades (`docs/framework.md`). You don’t need to derive formulas — you only need the rules “closed bar”, “three windows”, “five KPIs” to read a report.

## When you do need more math

- Designing a new feature → reading a few sections of `docs/math.md` is enough.
- Changing the fill logic (fees, slippage) → you need to read the event backtester in `src/`.
- Just validating a sentence → the template + five KPIs are enough.

## Fine print

- [docs/math.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/math.md)
