# Closed bar

**One-liner:** When you decide at bar open, you cannot use this bar’s already-completed close and features.

## Wrong vs right

Think of a bar as a “two-hour exam”:

| Wrong | Right |
|---|---|
| The exam just started, and you decide using the whole exam’s score. | The exam just started, and you can only use the **previous** exam’s published score. |

In this repo: the feature-store row is indexed by this bar’s **open** time, but this row’s high, low, close, and features are only knowable at this bar’s **close**. Deciding at open while reading the same row = peeking into the future.

The same script, using “same-bar”, once printed 50–70% CAGR; after switching to closed-bar, the clean windows were roughly −35% to −43%. Those pretty numbers are **void**.

```text
Wrong: wall clock = this bar’s open → read this bar’s completed features
Right: wall clock = this bar’s open → read previous bar’s closed features
```

## How this repo uses it

The feature store is computed monthly; the backtest at open only reads the previous bar. The cross column used by the golden cross is also only knowable after close — it is not a point you casually mark on a web daily chart. Labels (future returns) may look ahead, but are forbidden as entry features.

## Fine print

- [Lessons · closed bar](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/lessons.md#closed-bar)
- [Math · closed-bar clock](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/math.md)
