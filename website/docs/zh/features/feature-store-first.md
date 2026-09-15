# 特征库优先

策略和回测要用的特征，都必须先从特征库里读。库里没有这一列，就回去把库补建好，而不是在回测脚本里现场另算。

## 为什么要这样

特征库是「按月算好、存成文件、可以反复用」的测量层。如果每个回测脚本都自己现算一遍，同一个特征在不同脚本里可能算出不同的值（参数漂了、收盘才知道的规则处理得不一样），表就对不齐。锁成一个库，所有策略用同一套标准。

## 哪种做法才对

| 错 | 对 |
|---|---|
| 在回测脚本里临时写一段代码，现场算出一列。 | 先在配方表里登记这一列，再跑补建脚本，回测直接读已经算好的列。 |

## 在这个仓库里怎么用

1. 如果这列还没登记，先写进 `config/feature_dependencies.yaml`。
2. 再跑增量补建：

```bash
PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/strategies/<archetype> \
  --symbols BTCUSDT,BNBUSDT,SOLUSDT \
  --timeframe 120T \
  --root feature_store \
  --layer features_<archetype>_<TF>_<hash> \
  --data-path data/parquet_data
```

构建器会检查每个月缺哪些输出列，只补缺的那些。

## 还想看细则

- [docs/features.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/features.md)
- [docs/usage.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/usage.md)
