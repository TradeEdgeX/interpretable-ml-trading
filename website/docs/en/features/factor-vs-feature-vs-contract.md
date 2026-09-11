# Factor · feature · contract

**One line:** Keep three things apart — factor/regime is which side you stand on; features measure; the contract writes entry/exit.

## Wrong vs right

| | What it is | Example |
|---|---|---|
| **Factor / regime** | Which exposure side you stand on (often permission only) | Price above a slow MA → longs allowed |
| **Feature** | A measurement of geometry or flow | Cross sign change, funding z, bar P99 |
| **Contract** | Locked entry / stop / exit rules | Long on cross; exit if price breaks EMA50 |

| Wrong | Right |
|---|---|
| Stack a dozen features into one score and call it stronger alpha. | The store only measures; archetype YAML signs the contract. Stacking scores does not magically become pick alpha. |

## In this repo

MA cross: slow-MA side can be permission; the cross column is the feature; breaking the 50 is the contract exit. The store does not size or stop for you.

## Deeper docs

- [Math · layers](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/math.en.md)
- [YAML contract](../tech/yaml-contract.md)
