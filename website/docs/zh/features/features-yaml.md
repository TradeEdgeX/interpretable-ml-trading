# features.yaml 怎么读

**一句话：** `features.yaml` 是特征库的「配方表」：每个节点声明它产出哪几列、吃什么原料、算什么。

## 一个节点长什么样

```yaml
- name: ema_50_200_cross_f
  outputs: [ema_50_200_cross_side]
  inputs: [close]
  params: {fast: 50, slow: 200}
```

- `name`：节点名，脚本里引用它。
- `outputs`：这个节点产出哪些列，落盘到特征库。
- `inputs`：它吃哪些原始列（这里是收盘价）。
- `params`：参数（快线 50、慢线 200）。

## 怎么找到你要的列

1. 打开 `config/strategies/<archetype>/features.yaml`。
2. 搜列名（比如 `ema_50_200_cross_side`），找到产出它的节点。
3. 看 `inputs` 和 `params`，知道这列是怎么算的。

## 还想看细则

- [docs/features.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/features.md)
- [特征库优先](feature-store-first.md)
