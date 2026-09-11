# Feature store first

**One line:** At the open, read only the previous closed bar; layers land as monthly files. Missing columns → backfill the same layer — never compute inside the backtest.

## Wrong vs right

| Wrong | Right |
|---|---|
| Drop columns from `requested_features` to dodge errors; or rename the layer to “add a column”. | Register the node → rebuild with the **same layer** name; keep old columns, compute only gaps. |
| Use labels (forward returns) as entry features. | Labels may look ahead; never as X / entry. |

Node names and rule column names often differ:

```text
node ema_50_200_cross_f  →  column ema_50_200_cross_side (what rules reference)
```

## In this repo

Public `ma_cross` lists requested nodes; `event_backtest` reads the layer strictly. Commands and incremental builds live in repo docs — this site does not copy the encyclopedia.

## Deeper docs

- [Features · add a column](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/features.en.md)
- [Closed bar](../quant/closed-bar.md)
