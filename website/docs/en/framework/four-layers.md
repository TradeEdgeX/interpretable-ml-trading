# Four local layers

**One-liner:** From raw trades to a KPI table, data passes through four layers: data → feature store → strategy YAML → court. Each layer has a locked format, and layers only talk to each other through files.

## The four layers at a glance

| Layer | What it does | Locked format |
|---|---|---|
| Data | Download, clean, monthly parquet | `data/parquet_data/<SYMBOL>/<TF>/<YYYY-MM>.parquet` |
| Feature store | Compute all measurement columns monthly | `feature_store/features_<arch>_<TF>_<hash>/<SYMBOL>/<YYYY-MM>.parquet` |
| Strategy YAML | Write the contract as machine-readable rules | `config/strategies/<archetype>/*.yaml` |
| Court | Run the backtest, print the five-KPI-by-window table | `scripts/event_backtest.py` etc. |

## Why layer it

Layers only talk through files, not memory. So:

- Switch machines — as long as the files are there, results reproduce.
- Change a strategy without recomputing features; change features without re-downloading data.
- When the AI edits YAML for you, it can’t accidentally touch data or features.

## How this repo uses it

Most of the time you only touch the “strategy YAML” layer. Data and the feature store are infrastructure; the court is the locked ruler.

## Fine print

- [Stack and paths](../tech/stack.md)
- [docs/framework.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/framework.md)
