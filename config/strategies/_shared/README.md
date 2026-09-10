# Shared FeatureStore layers (`config/strategies/_shared`)

Two AUTO layers from the same config dir; hash differs by feature manifest.

| Layer | Manifest | Nodes | ~Output cols | Resolve |
|-------|----------|-------|--------------|---------|
| **tree_core** | `features.yaml` | ~95 (BPC/TPC/ME/SRB union) | ~455 | `features_tree_core_120T_c005db49f7` |
| **tree_full** | `features_all.yaml` | 289 (`_f` registry) | ~940 | `features_tree_full_120T_958f665062` |

Regenerate manifest + print current layer id:

```bash
python scripts/generate_all_features_yaml.py --strategy-config config/strategies/_shared

python -c "from src.feature_store.tree_full_layer import resolve_tree_full_layer; print(resolve_tree_full_layer())"
```

## Build tree_full (~940 cols, all strategies)

```bash
LAYER=$(python -c "from src.feature_store.tree_full_layer import resolve_tree_full_layer; print(resolve_tree_full_layer())")

mlbot feature-store build --no-docker \
  -c config/strategies/_shared \
  --features-yaml config/strategies/_shared/features_all.yaml \
  -t 120T \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT \
  --start-date 2024-01-01 --end-date 2026-04-01 \
  --warmup-months 6
# Layer printed at end → e.g. features_tree_full_120T_958f665062
```

Or highcap universe:

```bash
mlbot feature-store build --no-docker \
  -c config/strategies/_shared \
  --features-yaml config/strategies/_shared/features_all.yaml \
  -t 120T \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-groups highcap \
  --start-date 2024-01-01 --end-date 2026-04-01 \
  --warmup-months 6
```

## Use from any strategy (read-only subset)

Each slug keeps its own slim `features.yaml` for **model** columns. Point prepare/train at **tree_full** layer so FeatureStore hits cache for any column in the registry:

```bash
LAYER=features_tree_full_120T_958f665062

# IC / prepare on full registry (parquet gets ~940 cols)
mlbot train final --no-docker --prepare-only \
  -c config/strategies/tree_strategies/short_term_swing \
  --features config/strategies/_shared/features_all.yaml \
  --feature-store-layer "$LAYER" \
  --symbol BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT \
  -t 120T \
  --output-root results/train_final/short_term_swing/prepare_wide_<run_id>

# Train still uses pruned features.yaml (post ic-prune writeback)
mlbot train final --no-docker \
  -c config/strategies/tree_strategies/short_term_swing \
  --feature-store-layer "$LAYER" \
  ...
```

**rd_loop:** set top-level `feature_store_layer: features_tree_full_120T_958f665062`.  
For wide IC prepare, add prepare step `features: config/strategies/_shared/features_all.yaml`.

## Invalidation

Rebuild when `feature_dependencies.yaml`, feature compute code, or `features_all.yaml` changes (layer hash shifts).
