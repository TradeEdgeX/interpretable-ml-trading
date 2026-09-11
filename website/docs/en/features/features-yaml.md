# How to read features.yaml

**One-liner:** `features.yaml` is the feature store’s “recipe table”: each node declares which columns it produces, what raw inputs it eats, and what it computes.

## What a node looks like

```yaml
- name: ema_50_200_cross_f
  outputs: [ema_50_200_cross_side]
  inputs: [close]
  params: {fast: 50, slow: 200}
```

- `name`: the node name, referenced by scripts.
- `outputs`: which columns this node produces, written to the feature store.
- `inputs`: which raw columns it eats (here: close).
- `params`: parameters (fast 50, slow 200).

## How to find the column you want

1. Open `config/strategies/<archetype>/features.yaml`.
2. Search the column name (e.g. `ema_50_200_cross_side`) to find the node that produces it.
3. Read `inputs` and `params` to see how the column is computed.

## Fine print

- [docs/features.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/features.md)
- [Feature-store-first](feature-store-first.md)
