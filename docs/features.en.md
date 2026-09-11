# Feature computation

**中文:** [features.md](features.md)  
Landing: [../README.md](../README.md)

Features are **computed ahead of time and stored by month**. Backtests and rules only read the FeatureStore. They do not `compute_*` on the hot path.  
Closed bar: [lessons.md#closed-bar](lessons.md#closed-bar) · [math.en.md](math.en.md) §1.

---

## What this page is measuring

| Name | What it is | What it is not |
|---|---|---|
| `*_f` | A DAG node in `config/feature_dependencies.yaml` | The column a rule cites |
| Output column | The FeatureStore column (`output_columns`) | Often the node name without `_f` |
| FeatureStore layer | `feature_store/<layer>/<symbol>/<TF>/YYYY-MM.parquet` | A moving average typed into chat |

Example: node `ema_50_200_cross_f` → column `ema_50_200_cross_side`. The rule YAML says `feature: ema_50_200_cross_side`.

```
Trade parquet (bars; order-flow also needs ticks)
        │
        ▼
feature_dependencies.yaml (DAG: deps, required cols, outputs)
        │
        ▼
Monthly FeatureStore (warmup window + this month)
        │
        ▼
Archetype YAML / event_backtest read existing columns only
```

Missing column → register, backfill, then run the ruler. Do not drop a name from `requested_features` to silence a strict-read error.

---

## How to use columns that are already built

### 1. The pack names nodes

`config/strategies/<family>/features.yaml`:

```yaml
feature_pipeline:
  requested_features:
    - atr_f
    - ema_50_f
    - ema_200_value_f
    - ema_50_200_cross_f
```

Public dummy: `config/strategies/ma_cross/`. The golden-cross side reads `ema_50_200_cross_side`.

### 2. Rules read output columns, not node names

```yaml
direction_rules:
  - id: ma_cross_ema50_200_side
    feature: ema_50_200_cross_side
    transform: sign
```

Enter / stop / add / exit live in archetype YAML. The FeatureStore does not sign the ticket.

### 3. Build the store

```bash
mlbot feature-store build \
  --config config/strategies/ma_cross \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 120T \
  --start-date 2022-01-01 --end-date 2026-06-01
```

Pin an existing layer (required when adding a column — see below):

```bash
PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/strategies/ma_cross \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 120T \
  --root feature_store \
  --layer features_ma_cross_120T_<hash> \
  --data-path data/parquet_data
```

### 4. Backtests only read the layer

`event_backtest` / `mlbot research run` use `feature_store_strict=True`. The layer name is `feature_store_layer` in the pack `meta.yaml`.  
Column missing on disk → stop and backfill. Do not add `compute_*` in the backtest.

### 5. List compute functions

```bash
mlbot features list
mlbot features list --category orderflow
mlbot features list --search vpin
mlbot features count
```

The DAG in `config/feature_dependencies.yaml` is authoritative. Registration lives in `src/features/registry.py`.

---

## Order flow

Order-flow nodes use **aggTrade ticks** (`price` / `volume` / `side` plus a time index). They are not a fake delta made from the close.

| Need | Data | Node examples | Output examples |
|---|---|---|---|
| Informed-trade probability | ticks | `vpin_features_f`, `vpin_base_aligned_features_f` | `vpin_*` |
| Buy/sell consensus | ticks | `ofci_f`, `ofci_pct_f` | `ofci`, `ofci_pct` |
| Cumulative volume delta | ticks or derived | `cvd_basic_f`, `cvd_divergence_v2_f` | `cvd`, `cvd_change_*` |
| Single-bar footprint | ticks inside the bar | `footprint_basic_f` | `fp_poc`, `fp_hvn`, `fp_delta_*` |
| Same-side runs | ticks | trade clustering (`utils_order_flow_features`) | `trade_cluster_*` |

Download then convert:

```bash
mlbot data download --symbols BTCUSDT --start-year 2022 --start-month 1 \
  --end-year 2026 --end-month 6
mlbot data convert --symbols BTCUSDT
```

Default tick dir: `data/parquet_data/`. Compute functions load months via `ticks_loader_json` / `ticks_dir`. Bars with no ticks stay NaN; nothing is invented.

Funding and open interest are a separate parquet path (`mlbot data download-funding-rate` / `download-open-interest`). Not order flow, but they also land on disk before the DAG.

Order flow may gate (trade or not). **Do not** use a slow math series as the master veto. Layers: [math.en.md](math.en.md) §2.

---

## Math features

Registered slow math (all under `src/features/time_series/`):

| Family | Node examples | Question | Not the question |
|---|---|---|---|
| WPT | `wpt_price_fluctuation_f` | Detrended fluctuation | Is this a break |
| Hilbert | `hilbert_features_f` | Envelope, CVD/price divergence | Enter or not |
| Hurst | `hurst_price_f`, `hurst_cvd_f` | Trend vs noise path | Direction |
| Spectrum | `spectrum_*_f` | Cycle / energy | Entry geometry |
| DTW | `dtw_features_f` | Distance to a template path | A gate by itself |
| GARCH / EVT | `garch_features_f`, EVT nodes | Vol clustering, tails | Contract geometry |

Hilbert depends on WPT-detrended price / CVD. Hurst rolling windows use `[t-W, t-1]` only.  
**Math does not answer “is this a break”; it answers “how careful now.”** Use it in Execution (noise penalty, smaller size), not as a gate that zeroes historical failures.

Close-only technicals (ATR, RSI, EMA, ROC) sit in the same DAG. They are cheap and do not read ticks. The public dummy uses only that tier.

---

## How to go faster

Feature-level compute is **sequential**. Speed comes from doing less work and slicing by month / symbol, not from shipping a whole frame into a process pool.

| Lever | What it does | When |
|---|---|---|
| Incremental merge | Existing months compute only missing `*_f`; old columns stay | **Default when adding a column** |
| Same `--layer` | Do not rename the layer for a new column | Unless you mean a new dataset |
| Cross-layer donor | Copy a missing month from another layer (same symbol / TF), then fill gaps | On by default; `--no-reuse` turns it off |
| `--workers N` | Parallelize across symbols (one subprocess each) | 2–8; does not change values |
| `FEATURE_MONTHLY_WORKERS` | Parallelize months inside one symbol | Tick / DTW / Hilbert are heavy |
| Monthly cache + warmup | On-disk per month; default `FEATURE_MONTHLY_WARMUP_MONTHS=3` | Avoids all-NaN rolling at month start |
| Close-only first | EMA / ATR / crosses need OHLC only | An order of magnitude faster than VPIN / footprint |
| Avoid `--force-rebuild` | Rebuilds the whole layer | Only if the layer is corrupt and donors must be dropped |

```bash
# Many symbols
mlbot feature-store build --config config/strategies/ma_cross \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT --timeframe 120T --workers 4

# Heavy features: slice by month
FEATURE_MONTHLY_WORKERS=4 FEATURE_MONTHLY_WARMUP_MONTHS=3 \
  PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/strategies/ma_cross --symbols BTCUSDT --timeframe 120T \
  --layer features_ma_cross_120T_<hash>
```

`--workers` and `FEATURE_MONTHLY_WORKERS` only cut work units. They do not change the formula.

---

## Add one column and reuse the ones you already have

Do not rename the layer for a new feature. A new name rebuilds everything.

1. Write the function under `src/features/` and `@register_feature` it if it is new.
2. Add a `*_f` node to `config/feature_dependencies.yaml`: `compute_func`, `dependencies`, `required_columns`, `output_columns`. Dependencies already on the layer are not recomputed.
3. Append the node to that pack’s `features.yaml` `requested_features`.
4. **Build again with the same `--layer`** (do not rename, do not `--force-rebuild`).

The builder looks at each month’s meta:

- All output columns present → skip the month
- A `*_f` output is missing → `merge_mode`, compute only the gap, write back into the same monthly parquet (`+N features`)
- This layer lacks the month, another layer has it → copy the donor, then fill only the gap

The DAG pulls dependencies. Asking only for `hilbert_features_f` still needs `wpt_price_fluctuation` / `wpt_cvd_fluctuation`; if they are already on the layer they are read, not rebuilt.

```bash
# features.yaml just gained ema_50_200_cross_f
PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/strategies/ma_cross \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 120T \
  --root feature_store \
  --layer features_ma_cross_120T_<hash> \
  --data-path data/parquet_data
```

The log should say `Incremental … +1 features (ema_50_200_cross_f)`, not a full-month rebuild.

Close-only columns (crosses, MAs, ATR) backfill quickly. VPIN / footprint / DTW still read ticks or long windows, but **existing OHLC columns are not recomputed**.

### Do not

- Drop a name from `requested_features` to dodge a strict-read error
- Add `_compute_*_column()` on the backtest hot path
- Rename the layer when you did not mean a new dataset
- Treat `--force-rebuild` as “add a column”
- Use a label (`forward_rr*`) as an entry feature

---

## Related

| Page | Content |
|---|---|
| [usage.en.md](usage.en.md) | Download, build, backtest commands |
| [lessons.md#features](lessons.md#features) | FeatureStore-first |
| [math.en.md](math.en.md) | Closed bar; math is not a gate |
| [ARCHITECTURE.en.md](ARCHITECTURE.en.md) | How the ruler meets the backtest |
| `config/feature_dependencies.yaml` | DAG |
| `src/features/` | Compute functions |
| `src/feature_store/` | Monthly read/write |

Landing: [README.md](../README.md)
