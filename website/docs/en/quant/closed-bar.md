# Closed bar

**One line:** When you decide at the open, you must not use this bar’s already-finished close and features.

## Wrong vs right

Think of a candle as a two-hour exam:

| Wrong | Right |
|---|---|
| Use the final exam score at the moment the exam starts. | At the open, use only the **previous** exam’s published score. |

In this repo the row index is the bar **open**, but that row’s OHLC and features are only knowable at the bar **close**. Reading the same row at the open is lookahead.

The same script once printed 50–70% CAGR with open-bar reads; after closed-bar, clean windows were about −35% to −43%. Those pretty numbers are **void**.

```text
Wrong: wall-clock = this bar open → read this bar’s completed features
Right: wall-clock = this bar open → read the previous closed bar
```

## In this repo

The feature store is built monthly; backtests at the open read only the prior bar. Golden-cross columns are knowable only after the close — not a hand-marked point on a website chart.

## Deeper docs

- [Lessons · closed bar](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/lessons.md#closed-bar)
- [Math · clock](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/math.en.md)
