# Fill the template

**One-liner:** Turning a trading idea into a testable contract starts with filling the five boxes of `docs/hypothesis_template.md`.

## The five boxes

| Box | What to write | Example (MA golden cross) |
|---|---|---|
| Mechanism | Why this sentence makes money | Trend continuation: what went up keeps going up |
| Expected regime | Which window should earn, which should lose | Earns in trend windows, loses in chop windows |
| Contract | Entry / direction / exit written down hard | Go long when close is above the 50-day line and just crossed above the 200-day; exit on break of the 50-day |
| Falsification | Which KPI in which window voids it | Any window with negative CAGR, or chop-window drawdown deeper than trend-window |
| Landing | Which court, which feature store | Event backtest + `features_ma_cross_120T_<hash>` |

## After filling it

Hand the filled template to the AI; it helps you translate it into YAML, run the backtest, and print the table. You read the table, check against the falsification line, and write your own conclusion.

## Fine print

- [docs/hypothesis_template.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/hypothesis_template.md)
- [docs/hypothesis.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/hypothesis.md)
