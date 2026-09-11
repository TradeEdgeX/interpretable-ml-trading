# Which court

**One line:** The shape of the sentence’s clock picks the court. Capability is not edge.

## How to choose

| Sentence shape | Court | Examples |
|---|---|---|
| Events on one symbol, one timeline | **Event backtest** `event_backtest` | MA cross, funding fade, Monday |
| Score a universe each day; buy a quantile | **Cross-section** `cs_panel` | Mom+amount (rejected) |
| Enter on a date; hold a fixed number of years | **Cohort** | Small-cap hold from entry day |

“Who will be the next tenbagger?” has no closed-bar column — **cannot** be measured as a hypothesis. You can only measure rules that already happened.

Public `mlbot research run` dispatches event backtest only. Cross-section / cohort use their own scripts.

## Wrong vs right

| Wrong | Right |
|---|---|
| The framework can run cross-section → the idea makes money. | Measured cross-section sentences are rejects — capability demos, not strategy tips. |
| Stuff “score the basket daily” into a single-name event backtest. | Use the right court, or rewrite as an event sentence. |

## Deeper docs

- [cs_panel](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/cs_panel.en.md)
- [Tenbagger example](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260911_tenbagger_smallcap.en.md)
