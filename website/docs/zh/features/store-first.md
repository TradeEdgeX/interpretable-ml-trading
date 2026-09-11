# 特征库优先

**一句话：** 开盘只读上一根已收盘行；层按月落盘。缺列先补同一层，不要在回测里现场算。

## 错 vs 对

| 错 | 对 |
|---|---|
| 为躲报错从 `requested_features` 删列；或换层名「加一列」。 | 登记节点 → 带**原来的层名**增量 build；旧列留下，只算缺的。 |
| 把标签（未来收益）当入场特征。 | 标签可以看未来；禁止当 X / 入场。 |

节点名和规则读的列名常常不是同一个：

```text
节点 ema_50_200_cross_f  →  列 ema_50_200_cross_side（规则里写这个）
```

## 本仓库怎么用

公开练习包 `ma_cross` 点名需要的节点；`event_backtest` 严格读层。命令与增量细节在仓库文档，站点不复制百科。

## 还想看细则

- [特征计算 · 加一列](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/features.md)
- [闭棒](../quant/closed-bar.md)
