# Second example: fade crowded funding

The landing page is still [README.md](../../README.md).  
That page uses a moving-average golden cross to contrast a web AI with this repo.  
This page records a **second full, auditable loop**: sentence → five boxes → data → FeatureStore → YAML → three-segment court → numbers. A person still writes the verdict.

Experiment: [config/experiments/20260910_funding_fade/](../../config/experiments/20260910_funding_fade/)  
Public demo family stays `ma_cross`. **Do not edit** the default pack `config/strategies/ma_cross/` (EMA1200 dead zone + ATR trail — not this sentence).

Chinese original: [20260910_funding_fade_CN.md](20260910_funding_fade_CN.md)

---

## Why this sentence, not a chain dashboard

Nansen / Glassnode / Dune screenshots need keys and cannot be reproduced in this public tree.  
Funding rates **can** be downloaded here (`mlbot data download-funding-rate`). The FeatureStore already registers `funding_rate_features_f` → `funding_rate_zscore_50`.

---

## What you ask a web AI

> Short when funding is very high, long when it is very low, exit when it normalizes. Does that make money?

It usually says: crowded unwind is a classic story, funding stays positive in bulls so shorts get hurt, add open interest or a price filter.  
No bear / bull / last-few-years numbers. No contract for what counts as wrong.

---

## What you say here

> Fade a 50-observation funding z-score at ±1.5; exit when it returns through 0.  
> I expect the crowded side to give back; none of the three windows should blow the book. Measure this.

---

## Five boxes

| Box | This sentence |
|---|---|
| Mechanism | Robust z-score of the last 50 **funding prints** ≥ 1.5 → short; ≤ −1.5 → long (`negate_sign`). |
| Regimes | Extreme crowding should unwind. Bear / bull / recent windows must not blow the book. |
| Contract | Exit when z returns through 0 (long if z≥0, short if z≤0). No adds, no averaging, no trail. |
| Falsifiers | Any window with negative CAGR, or recent MaxDD worse than both trend windows. |
| Landing | Same sentence for the robot and for your hands. |

Those 50 prints are native funding cadence (~8 hours, ~17 days), not 50 two-hour bars.

Lineage: `mlbot research index --trusted --query funding` → 0 trusted hits in the public tree. Then `mlbot research init 20260910_funding_fade --strategy ma_cross`.

---

## Data and FeatureStore

Download (see [usage.en.md](../usage.en.md)):

```bash
mlbot data download --symbols BTCUSDT \
  --start-year 2021 --start-month 6 --end-year 2026 --end-month 6
mlbot data convert --symbols BTCUSDT
mlbot data download-funding-rate --symbols BTCUSDT \
  --start-year 2020 --start-month 1 --end-year 2026 --end-month 6
```

Build a **local** layer (do not write into someone else’s FeatureStore):

```bash
PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/experiments/20260910_funding_fade/strategies/ma_cross \
  --symbols BTCUSDT --timeframe 120T \
  --root feature_store --layer features_funding_fade_120T \
  --data-path data/parquet_data \
  --start-date 2021-06-01 --end-date 2026-06-01 --no-reuse
```

The column already existed. The **exit contract** did not: the engine gained `structural_exit: funding_zscore0`. Do not invent the z-score inside the backtest.

---

## Court (kill switch off)

```bash
mlbot research run 20260910_funding_fade
mlbot research close 20260910_funding_fade
```

A person `--declare`s. Agents must not type `verdict:`.

| Window | CAGR | Calmar | Win rate | MaxDD | Sharpe(R) | Trades |
|---|---|---|---|---|---|---|
| Bear 2022 | +2.7% | 0.93 | 47.7% | −2.9% | 0.08 | 128 |
| Bull 2023–2024 | +3.0% | 0.60 | 50.7% | −4.9% | 0.13 | 75 |
| Recent range → bear | −3.4% | −0.50 | 45.9% | −6.9% | −0.11 | 109 |

Recent CAGR is negative, and recent MaxDD is deeper than both trend windows. Both falsifiers hit.  
**You write the verdict.** Fail: the robot does not run it, and you do not hand-trade it tonight.
