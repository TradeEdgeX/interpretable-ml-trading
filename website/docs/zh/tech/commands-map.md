# 命令地图

**一句话：** 常用命令分四组：下载数据、建特征库、跑回测、查结果。

## 下载数据

```bash
# 下载原始 ZIP
mlbot data download --symbols BTCUSDT \
  --start-year 2026 --start-month 6 --end-year 2026 --end-month 6

# 转换成 parquet
mlbot data convert --symbols BTCUSDT

# 或者一步到位
mlbot data pipeline --symbols BTCUSDT
```

## 建特征库

```bash
PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/strategies/ma_cross \
  --symbols BTCUSDT,BNBUSDT,SOLUSDT \
  --timeframe 120T \
  --root feature_store \
  --layer features_ma_cross_120T_<hash> \
  --data-path data/parquet_data
```

## 跑回测

```bash
# 事件回测
PYTHONPATH=src python scripts/event_backtest.py \
  --config config/strategies/ma_cross \
  --symbols BTCUSDT \
  --timeframe 120T

# 横截面回测
PYTHONPATH=src python scripts/cross_section_backtest.py \
  --config config/strategies/cs_mom_amount \
  --timeframe 1d
```

## 查结果

结果在 `results/<strategy>/<timestamp>/` 里，包括：

- `kpi_table.csv`：三段五项 KPI 表。
- `trades.csv`：每一笔成交。
- `equity_curve.csv`：权益曲线。

## 还想看细则

- [docs/usage.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/usage.md)
