# YAML contract

**One line:** Rules live in YAML; features only measure. The feature store does not sign the contract.

## Three layers (plain)

| Layer | Question |
|---|---|
| **Permission** | Is this geometry allowed now? |
| **Direction** | Long / short / flat |
| **Execution** | Stops, exits; adds off by default |

MA cross: the cross column sets direction; breaking EMA50 is the exit contract — not another ad-hoc indicator.

## Wrong vs right

| Wrong | Right |
|---|---|
| Invent untested take-profits or averaging mid-backtest. | Entry / direction / exit are written in the experiment pack; adding size is another hypothesis. |

## Deeper docs

- [Factor · feature · contract](../features/factor-vs-feature-vs-contract.md)
- [Public `ma_cross`](https://github.com/TradeEdgeX/interpretable-ml-trading/tree/main/config/strategies/ma_cross)
