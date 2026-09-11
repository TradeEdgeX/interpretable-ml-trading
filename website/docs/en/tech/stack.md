# Stack and paths

**One-liner:** From raw trades to a KPI table, data flows through four directories: `data/agg_data/` → `data/parquet_data/` → `feature_store/` → `results/`.

## The four directories

| Directory | What it holds | Who writes it |
|---|---|---|
| `data/agg_data/` | Raw ZIPs (downloaded from the exchange) | `mlbot data download` |
| `data/parquet_data/` | Cleaned monthly parquet | `mlbot data convert` |
| `feature_store/` | Monthly computed feature columns | `build_feature_store_from_config.py` |
| `results/` | KPI tables and logs printed by backtests | `event_backtest.py` etc. |

## The data flow

```text
Exchange ZIP
  → data/agg_data/
  → data/parquet_data/  (mlbot data convert)
  → feature_store/      (build_feature_store_from_config.py)
  → results/            (event_backtest.py)
```

## Config files

- `config/strategies/<archetype>/features.yaml`: the feature recipe table.
- `config/strategies/<archetype>/strategy.yaml`: the strategy contract.
- `config/feature_dependencies.yaml`: the feature dependency registry.

## Fine print

- [docs/usage.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/usage.md)
- [Commands map](commands-map.md)
