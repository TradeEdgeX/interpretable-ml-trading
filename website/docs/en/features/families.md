# Five feature families

**One-liner:** Features in this repo are grouped into five families by “who is being measured”. Knowing which family a sentence belongs to tells you which YAML to look in and which court to use.

## The five families at a glance

| Family | Who is measured | Typical columns | Court |
|---|---|---|---|
| Single-symbol event | One symbol on one bar | MA, cross, ATR, RSI | Event backtest |
| Cross-section | A basket compared every day | Momentum rank, amount rank | Cross-section backtest |
| Funding / order flow | Perp funding, big orders | funding, P99 big orders | Event backtest |
| Calendar / weekday | Time: day of week, day of month | Monday, month-end | Event backtest |
| Cross-symbol lead-lag | One symbol leads another | BTC leads AI alts | Event backtest |

## How to use it

When writing a hypothesis, first ask: is this sentence measuring “one symbol on one bar”, or “a basket compared every day”? The former goes on the event axis, the latter on the cross-section axis. Pick the wrong axis and the backtest script is wrong.

## Fine print

- [framework.md §3 feature families](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/framework.md)
