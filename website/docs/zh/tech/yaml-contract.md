# YAML 合同

**一句话：** 规则写在 YAML；特征只提供测量。特征库不签合同。

## 三层（人话）

| 层 | 问什么 |
|---|---|
| **许可** | 这类几何现在许不许做 |
| **方向** | 多 / 空 / 观望 |
| **执行** | 止损、出场、默认不加仓 |

金叉：交叉列给出方向；跌破 EMA50 是出场合同，不是另一个现场算的指标。

## 错 vs 对

| 错 | 对 |
|---|---|
| 回测时临时加没验过的止盈、摊平。 | 进 / 向 / 出事先写在实验包；加仓是另一句假设。 |

## 还想看细则

- [因子·特征·合同](../features/factor-vs-feature-vs-contract.md)
- [公开练习 `ma_cross`](https://github.com/TradeEdgeX/interpretable-ml-trading/tree/main/config/strategies/ma_cross)
