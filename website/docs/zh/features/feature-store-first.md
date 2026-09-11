# 特征库优先

**一句话：** 所有策略 / 回测要用的特征，都必须先从特征库读。库里没有这列，就回去补建库，而不是在回测脚本里现算。

## 为什么

特征库是「按月算好、落盘、可复用」的测量层。如果每个回测脚本都自己现算一遍，同一个特征在不同脚本里可能算出不同的值（参数漂移、闭棒处理不一致），表就没法对齐。锁死一个库，所有策略用同一把尺子。

## 错 vs 对

| 错 | 对 |
|---|---|
| 回测脚本里 `def compute_rsi(...)` 现算一列。 | 在 `features.yaml` 注册这列 → 跑 `build_feature_store_from_config.py` 增量补建 → 回测直接读列。 |

## 本仓库怎么用

1. 在 `config/feature_dependencies.yaml` 注册新特征（如果还没注册）。
2. 跑增量补建：

```bash
PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/strategies/<archetype> \
  --symbols BTCUSDT,BNBUSDT,SOLUSDT \
  --timeframe 120T \
  --root feature_store \
  --layer features_<archetype>_<TF>_<hash> \
  --data-path data/parquet_data
```

构建器会检测每月缺哪些输出列，只跑增量 merge（`+N features`）。

## 还想看细则

- [docs/features.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/features.md)
- [docs/usage.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/usage.md)
