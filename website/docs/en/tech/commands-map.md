# Commands map

**One-liner:** Common commands fall into four groups: download data, build the feature store, run backtests, read results.

## Download data

```bash
# Download raw ZIPs
mlbot data download --symbols BTCUSDT \
  --start-year 2026 --start-month 6 --end-year 2026 --end-month 6

# Convert to parquet
mlbot data convert --symbols BTCUSDT

# Or in one shot
mlbot data pipeline --symbols BTCUSDT
```

## Build the feature store

```bash
PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/strategies/ma_cross \
  --symbols BTCUSDT,BNBUSDT,SOLUSDT \
  --timeframe 120T \
  --root feature_store \
  --layer features_ma_cross_120T_<hash> \
  --data-path data/parquet_data
```

## Run backtests

```bash
# Event backtest
PYTHONPATH=src python scripts/event_backtest.py \
  --config config/strategies/ma_cross \
  --symbols BTCUSDT \
  --timeframe 120T

# Cross-section backtest
PYTHONPATH=src python scripts/cross_section_backtest.py \
  --config config/strategies/cs_mom_amount \
  --timeframe 1d
```

## Read results

Results live in `results/<strategy>/<timestamp>/`, including:

- `kpi_table.csv`: the three-window five-KPI table.
- `trades.csv`: every fill.
- `equity_curve.csv`: the equity curve.

## Fine print

- [docs/usage.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/usage.md)
