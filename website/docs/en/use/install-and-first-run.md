# Install and first run

**One-liner:** Install once, run the MA golden-cross example end to end, and you know how this repo works.

## Install

```bash
git clone git@github.com:TradeEdgeX/interpretable-ml-trading.git
cd interpretable-ml-trading
pip install -r requirements.txt
```

## First run

1. Download data:

```bash
mlbot data pipeline --symbols BTCUSDT \
  --start-year 2022 --start-month 1 --end-year 2026 --end-month 8
```

2. Build the feature store:

```bash
PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/strategies/ma_cross \
  --symbols BTCUSDT \
  --timeframe 120T \
  --root feature_store \
  --layer features_ma_cross_120T_<hash> \
  --data-path data/parquet_data
```

3. Run the backtest:

```bash
PYTHONPATH=src python scripts/event_backtest.py \
  --config config/strategies/ma_cross \
  --symbols BTCUSDT \
  --timeframe 120T
```

4. Read the result: `results/ma_cross/<timestamp>/kpi_table.csv`.

## Fine print

- [docs/usage.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/usage.md)
- [Commands map](../tech/commands-map.md)
