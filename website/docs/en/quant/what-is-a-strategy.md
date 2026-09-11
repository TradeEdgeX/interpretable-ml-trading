# A strategy is a contract, not a feeling

**One-liner:** A strategy must state four things — when trading is allowed, how you exit when wrong, what you risk on this trade, and when it ends. “Buy the golden cross” alone is not a strategy.

## Wrong vs right

| Wrong | Right |
|---|---|
| “Buy the golden cross, feels like trends make money.” | “Only go long when the close is above the 50-day line *and* it just crossed above the 200-day line; exit when it breaks the 50-day line; if any window has negative CAGR, or the chop window draws down deeper than the trend window, this sentence is void.” |

A feeling cannot be re-run or falsified. A contract can: it pins down the boundary between “trade” and “don’t trade”, and pins down “what to do when wrong”, so a backtest can tell you what this sentence looks like in bear, bull, and recent windows.

## How this repo uses it

The MA golden-cross practice sentence fills every box: mechanism, expected regime, contract, falsification, landing. The machine YAML and the hand-written checklist are **the same sentence**. When the backtest does not pass, read the report and write the same-sentence conclusion, instead of making a “this time is different” version that night.

See [Five boxes](../design/five-boxes.md) · repo example in [README_CN.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/README_CN.md).

## Fine print

- [Hypothesis template](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/hypothesis_template.md)
- [Philosophy: charts are contracts](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/philosophy.md)
