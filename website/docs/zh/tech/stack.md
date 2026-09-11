# 数据流与目录

**一句话：** 从原始成交到一张 KPI 表，数据流过四个目录：`data/agg_data/` → `data/parquet_data/` → `feature_store/` → `results/`。

## 四个目录

| 目录 | 装什么 | 谁写 |
|---|---|---|
| `data/agg_data/` | 原始 ZIP（从交易所下载） | `mlbot data download` |
| `data/parquet_data/` | 清洗后的按月 parquet | `mlbot data convert` |
| `feature_store/` | 按月算好的特征列 | `build_feature_store_from_config.py` |
| `results/` | 回测印出的 KPI 表和日志 | `event_backtest.py` 等 |

## 数据流

```text
交易所 ZIP
  → data/agg_data/
  → data/parquet_data/  (mlbot data convert)
  → feature_store/      (build_feature_store_from_config.py)
  → results/            (event_backtest.py)
```

## 配置文件

- `config/strategies/<archetype>/features.yaml`：特征配方表。
- `config/strategies/<archetype>/strategy.yaml`：策略合同。
- `config/feature_dependencies.yaml`：特征依赖注册表。

## 还想看细则

- [docs/usage.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/usage.md)
- [命令地图](commands-map.md)
