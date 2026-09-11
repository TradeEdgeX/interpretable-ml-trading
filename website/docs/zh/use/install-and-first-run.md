# 安装与第一次

**一句话：** 装一次，跑通均线金叉的例子，就知道这个仓库怎么用了。

## 安装

```bash
git clone git@github.com:TradeEdgeX/interpretable-ml-trading.git
cd interpretable-ml-trading
pip install -r requirements.txt
```

## 第一次跑

1. 下载数据：

```bash
mlbot data pipeline --symbols BTCUSDT \
  --start-year 2022 --start-month 1 --end-year 2026 --end-month 8
```

2. 建特征库：

```bash
PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/strategies/ma_cross \
  --symbols BTCUSDT \
  --timeframe 120T \
  --root feature_store \
  --layer features_ma_cross_120T_<hash> \
  --data-path data/parquet_data
```

3. 跑回测：

```bash
PYTHONPATH=src python scripts/event_backtest.py \
  --config config/strategies/ma_cross \
  --symbols BTCUSDT \
  --timeframe 120T
```

4. 看结果：`results/ma_cross/<timestamp>/kpi_table.csv`。

## 还想看细则

- [docs/usage.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/usage.md)
- [命令地图](../tech/commands-map.md)
