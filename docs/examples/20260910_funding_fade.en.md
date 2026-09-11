# Fade crowded funding

Paper: [config/experiments/20260910_funding_fade/](../../config/experiments/20260910_funding_fade/)

Short when funding is very high, long when it is very low, exit when it normalizes. This repo can download funding (`mlbot data download-funding-rate`) and already has `funding_rate_zscore_50`.

中文：[20260910_funding_fade_CN.md](20260910_funding_fade_CN.md)

---

## Rule

> Fade a 50-print funding z-score at ±1.5; exit when z returns through 0. None of the three windows should blow the book.

| Box | This sentence |
|---|---|
| Mechanism | z ≥ 1.5 → short; z ≤ −1.5 → long (`negate_sign`). |
| Regimes | Crowding should unwind. Bear / bull / recent must not blow the book. |
| Contract | Exit when z crosses 0 (long if z≥0, short if z≤0). No adds, no trail. |
| Falsifiers | Any window with negative CAGR, or recent MaxDD worse than both trend windows. |
| Landing | Same sentence for the robot and for your hands. |

Those 50 prints are funding’s own clock (~8 hours, ~17 days), not 50 two-hour bars.

```bash
mlbot research index --trusted --query funding
mlbot research init 20260910_funding_fade --strategy ma_cross
```

---

## Data

```bash
mlbot data download --symbols BTCUSDT \
  --start-year 2021 --start-month 6 --end-year 2026 --end-month 6
mlbot data convert --symbols BTCUSDT
mlbot data download-funding-rate --symbols BTCUSDT \
  --start-year 2020 --start-month 1 --end-year 2026 --end-month 6

PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/experiments/20260910_funding_fade/strategies/funding_fade \
  --symbols BTCUSDT --timeframe 120T \
  --root feature_store --layer features_funding_fade_120T \
  --data-path data/parquet_data \
  --start-date 2021-06-01 --end-date 2026-06-01 --no-reuse
```

The z-score column already existed. The exit did not: the engine gained `structural_exit: funding_zscore0`. Do not compute the z-score inside the backtest.

---

## Court (kill switch off)

```bash
mlbot research run 20260910_funding_fade
mlbot research close 20260910_funding_fade
```

| Window | CAGR | Calmar | Win rate | MaxDD | Sharpe(R) | Trades |
|---|---|---|---|---|---|---|
| Bear 2022 | +2.7% | 0.93 | 47.7% | −2.9% | 0.08 | 128 |
| Bull 2023–2024 | +3.0% | 0.60 | 50.7% | −4.9% | 0.13 | 75 |
| Recent range → bear | −3.4% | −0.50 | 45.9% | −6.9% | −0.11 | 109 |

Recent CAGR is negative, and recent MaxDD is deeper than both trend windows. Both falsifiers hit.  
Declare with `mlbot research close 20260910_funding_fade --declare …`. Do not type `verdict:`.
