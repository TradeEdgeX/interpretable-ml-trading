# Fade crowded funding

Paper: [config/experiments/20260910_funding_fade/](../../config/experiments/20260910_funding_fade/)  
Class first as **crowding fade / mean-reversion**, not point-selection alpha and not a trend sleeve.  
Harness: **event_backtest** (BTCUSDT · 2h). The table is on the paper; the human still `--declare`. Do not type `verdict:`.

> Short when funding is very high, long when it is very low, exit when it normalizes. Extreme crowding should unwind; none of the three windows should blow the book.

The court asks whether perpetual-funding crowding survives all three windows. If it does, the sentence lives. If any window prints a negative CAGR, or recent MaxDD is deeper than both trend windows, the sentence dies. It is not a starting point for “add an OI filter.”

中文：[20260910_funding_fade_CN.md](20260910_funding_fade_CN.md)

---

## How this sentence walks the repo

You ask a web AI: “Fade crowded funding, flatten when it normalizes — does that make money?”

It usually says this is a classic crowding fade, better with open interest. There are no bear / bull / recent numbers, and no “who loses if the recent window dies.” Change the model, it will suggest another filter.

You say the same sentence here, plus “measure this.” The AI must first put the sentence into the [template](../hypothesis_template.en.md): the paying side is the crowded perpetual book; the object is a z-score on **funding’s own clock** (about every 8 hours, not 50 two-hour bars); class it as crowding fade, not trend alpha; any window with negative CAGR, or recent MaxDD deeper than both trends, kills it. After the template passes and lineage is empty, and only after you ask to measure, it downloads trades and funding, builds FeatureStore, and sits the 2h court.

The z-score column already existed. The missing piece was the exit, so the engine gained a contract — missing column → FeatureStore; missing contract → engine. The program prints three windows: small positives in trend years, recent −3.4% with a deeper hole. Both falsifiers hit. You still `--declare`. If it dies, the robot does not run it and a human does not hand-trade “funding is different this time.” Twisting OI onto this paper is a new sentence.

That is the philosophy: heuristics are priors; the framework falsifies. IC cannot ship. A rejected sentence is also forbidden for hands. The seven sections below are the same walk, unpacked.

---

## Design

```text
Human sentence (fade extreme funding, exit at z = 0)
  → template: sociology / math / stats / ruler / crypto three windows / five boxes
  → validate + lineage (no prior close of this sentence)
  → download ticks and funding, build FeatureStore — only after “measure this”
  → 2h court: fade |z| ≥ 1.5, exit when z crosses 0
  → human --declare
```

Use `mlbot research run` (that path dispatches 2h `event_backtest`). The z-score column already lived in the DAG. The missing piece was the exit, so the engine gained `structural_exit: funding_zscore0`. Missing column → FeatureStore. Missing contract → engine. Do not `compute_*` the z-score in the backtest.

| Box | This sentence |
|---|---|
| Mechanism | Robust z of the last 50 *funding prints* ≥ 1.5 → short; ≤ −1.5 → long (`negate_sign`). |
| Regimes | Crowding should unwind. A one-way year must not crush the fade book. If crowding stops mean-reverting recently, CAGR flips negative. |
| Contract | Exit when z crosses 0 (long if z≥0, short if z≤0). No adds, no scale-in, no trail. Kill switch off. Closed-bar. |
| Falsifier | Any window with negative CAGR, or recent MaxDD deeper than both trend windows. The recent window cannot promote alone. |
| Landing | Robot: `strategies/funding_fade` in this folder. Same sentence for hands. Do not keep twisting OI filters onto this paper. |

| Template box | How this sentence is written |
|---|---|
| Sociology | The paying side is the crowded perpetual book. Funding is a coordination ritual. |
| Math | Robust z on *funding’s own clock*, not 50 two-hour bars. Death is z through 0. |
| Stats | Crowding fade. Three-window five KPIs. Do not blend one CAGR. |
| Ruler | Any window CAGR < 0, or recent MaxDD worse than bear and bull. |
| Data range | `BTCUSDT` · 2h · `bear_2022` / `bull_2023_2024` / `recent_range_to_bear`. |

Clock (signal known at close t; position starts on **t+1**):

| Moment | What happens |
|---|---|
| Entry | Previous closed `funding_rate_zscore_50` absolute value ≥ 1.5 |
| Side | Short if z > 0, long if z < 0 |
| Exit | About 94% of trades exit when z crosses 0; the rest hit a wide initial stop, not a trail |
| Flat days | Earn 0 |

Headline five KPIs only: CAGR / Calmar / win rate / MaxDD / Sharpe. Do not sum R.

---

## Data

```bash
mlbot research validate 20260910_funding_fade
mlbot research index --trusted --query funding
mlbot data download --symbols BTCUSDT \
  --start-year 2021 --start-month 6 --end-year 2026 --end-month 6
mlbot data convert --symbols BTCUSDT
mlbot data download-funding-rate --symbols BTCUSDT \
  --start-year 2020 --start-month 1 --end-year 2026 --end-month 6
```

| Item | This machine |
|---|---|
| Symbol | `BTCUSDT` perpetual |
| Bar | Two-hour (`120T`) |
| Trades | `mlbot data download` + `convert` → `data/parquet_data` |
| Funding | `mlbot data download-funding-rate` → `data/funding_rate/parquet` |
| Funding clock | About one print every 8 hours; 50 prints ≈ 17 calendar days |
| Layer | `features_funding_fade_120T` (61 months; ATR + funding only, so the layer is thin) |
| Spot check | 2023-05: `funding_rate_zscore_50` has no NaNs; about 18% of bars have `|z| ≥ 1.5` |
| Calendar | [`config/market_segment.yaml`](../../config/market_segment.yaml) |
| Kill switch | Off |

| Window | Span | Role |
|---|---|---|
| `bear_2022` | 2022-01-01 → 2023-11-01 | Bear baseline. |
| `bull_2023_2024` | 2023-06-01 → 2025-01-01 | Bull. Crowded longs should unwind. |
| `recent_range_to_bear` | 2025-01-01 → 2026-05-31 | Recent. Cannot promote alone. |
| `recent_6m_oos` | 2025-12-01 → 2026-05-31 | Shorter reference. This sentence does not close on it. |

Funding files start in 2020 so the 2022 z-score is warm. That is not a fourth verdict window.

---

## Features

Read from FeatureStore (`feature_store_strict=True`). Do not compute the z-score in the backtest.

| Column | Construction | Closed-bar use |
|---|---|---|
| `funding_rate` | Binance perpetual funding, aligned onto 2h bars | Input to the node, not the side |
| `funding_rate_zscore_50` | Median / MAD z of the last 50 funding prints | `z[t]` is first tradable on the next bar |
| `atr_f` | Volatility for size and the wide stop | Open reads the previous bar |
| Exit | Engine contract `structural_exit: funding_zscore0` | Longs flatten when z≥0, shorts when z≤0 |

Pack: `config/experiments/20260910_funding_fade/strategies/funding_fade/`

| File | What it locks |
|---|---|
| `features.yaml` | `atr_f`, `funding_rate_features_f` |
| `prefilter.yaml` | Open only if `|funding_rate_zscore_50| ≥ 1.5` |
| `direction.yaml` | Feature `funding_rate_zscore_50`, `negate_sign` |
| `execution.yaml` | `structural_exit: funding_zscore0`; trail and adds off |
| `meta.yaml` | Layer `features_funding_fade_120T` |
| Grid | `funding_fade_grid.yaml` |

```bash
PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/experiments/20260910_funding_fade/strategies/funding_fade \
  --symbols BTCUSDT --timeframe 120T \
  --root feature_store --layer features_funding_fade_120T \
  --data-path data/parquet_data \
  --start-date 2021-06-01 --end-date 2026-06-01 --no-reuse
```

Days that have not warmed the z-score stay flat. No look-ahead fill.

---

## IC

**There is no cross-sectional IC table, and there should not be one.**

Daily Spearman IC is a flashlight for ranking a universe. This sentence is a single-name 0/1 clock. If someone wants `funding_rate_zscore_50` as a continuous score and an overnight IC, that is a new folder. A flashlight cannot change the sentence and cannot close the case.

---

## Validation

```bash
mlbot research run 20260910_funding_fade
mlbot research close 20260910_funding_fade
```

Ruler, written first: any window CAGR **< 0**, or recent MaxDD **deeper** than bear and bull.

### Three windows (BTCUSDT · 2h · kill switch off)

About 94% of exits are z through 0.

| Window | CAGR | Calmar | Win rate | MaxDD | Sharpe(R) | Trades |
|---|---:|---:|---:|---:|---:|---:|
| `bear_2022` | +2.7% | 0.93 | 47.7% | −2.9% | 0.08 | 128 |
| `bull_2023_2024` | +3.0% | 0.60 | 50.7% | −4.9% | 0.13 | 75 |
| `recent_range_to_bear` | **−3.4%** | −0.50 | 45.9% | **−6.9%** | −0.11 | 109 |

| Written falsifier | Hit? |
|---|---|
| Any window CAGR < 0 | Yes. Recent −3.4%. |
| Recent MaxDD deeper than both trend windows | Yes. −6.9% vs −2.9% and −4.9%. |

Positive trend windows cannot keep the recent window alive. The ruler said “none of the three windows should blow the book,” not “green trend years are enough.”

---

## Conclusion

Class: **crowding-fade claim**. Funding z measures crowding, not trend.

- Bear and bull CAGRs are small positives. Sharpe is 0.08 / 0.13 — not book-sized alpha.
- Recent CAGR is negative and MaxDD is deeper: crowding no longer unwinds the way the paper paid for.
- Dropping Top-3 trades is classification only.
- Do not twist “add OI / add funding level” onto this paper.

Declare with `mlbot research close 20260910_funding_fade --declare …`. If it dies, the robot does not run it and a human does not hand-trade “funding is different this time.”

| Path | What it is |
|---|---|
| `results/funding_fade/experiments/20260910_funding_fade/funding_z15/bear_2022` | Bear report |
| `results/funding_fade/experiments/20260910_funding_fade/funding_z15/bull_2023_2024` | Bull report |
| `results/funding_fade/experiments/20260910_funding_fade/funding_z15/recent_range_to_bear` | Recent report |
| [DECISION.md](../../config/experiments/20260910_funding_fade/DECISION.md) | Paper |

---

## How to read the report

1. **Same window first, then CAGR.** Do not explain recent −3.4% with bear +2.7%. Do not blend three windows into one “full-sample is fine.”
2. **Headline is CAGR, not ΣR.** Sharpe(R) is from trade R-multiples, not daily-return Sharpe.
3. **Win rate is not “it caught the bull.”** Recent win rate is 45.9%, close to bear 47.7%, but CAGR already flipped.
4. **Trade count is coverage, not a license to retune.** 109 recent trades do not authorize a second threshold sweep.
5. **~94% of exits are z back to 0.** If most exits were stops, the written contract was not the one that printed the table.
6. **The 50 prints are funding’s clock**, not 50 two-hour bars.
7. **Kill switch off compares edge.** “A fuse would have saved the recent window” is not a close.
8. **A green trend window cannot promote.** The falsifier says “any window.”

The paper is [DECISION.md](../../config/experiments/20260910_funding_fade/DECISION.md).
