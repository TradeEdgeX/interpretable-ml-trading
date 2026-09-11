# Feature-store-first

**One-liner:** Every feature a strategy / backtest uses must be read from the feature store first. If the column is not in the store, go backfill the store — don’t compute it on the fly inside the backtest script.

## Why

The feature store is the “computed monthly, on disk, reusable” measurement layer. If every backtest script computes its own copy, the same feature can come out different across scripts (parameter drift, inconsistent closed-bar handling), and tables stop lining up. Lock one store, and all strategies use the same ruler.

## Wrong vs right

| Wrong | Right |
|---|---|
| `def compute_rsi(...)` inside the backtest script. | Register the column in `features.yaml` → run `build_feature_store_from_config.py` incremental backfill → the backtest reads the column directly. |

## How this repo uses it

1. Register the new feature in `config/feature_dependencies.yaml` (if not already).
2. Run incremental backfill:

```bash
PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/strategies/<archetype> \
  --symbols BTCUSDT,BNBUSDT,SOLUSDT \
  --timeframe 120T \
  --root feature_store \
  --layer features_<archetype>_<TF>_<hash> \
  --data-path data/parquet_data
```

The builder detects which output columns are missing per month and only runs an incremental merge (`+N features`).

## Fine print

- [docs/features.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/features.md)
- [docs/usage.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/usage.md)
