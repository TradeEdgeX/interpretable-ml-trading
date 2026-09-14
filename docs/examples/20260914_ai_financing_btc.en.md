# AI financing days vs BTC funding and returns

Paper: [config/experiments/20260914_ai_financing_btc/](../../config/experiments/20260914_ai_financing_btc/)  
Class first as **beta** (risk-on spillover), not point-selection alpha.  
Planned harness: **event_backtest** (BTCUSDT · 2h). **Only Phase 1 daily scan has run. The two-hour five KPIs are still empty. Do not `--declare` yet.**

> After a public mega AI-lab financing announcement (UTC date D closed), long BTCUSDT for five UTC days. None of the three windows should blow the book.

The human said “AI financing and BTC funding and returns are related somehow.” The template turns that into the contract above: locked calendar, enter D+1, hold five days. “A relationship” is not a hypothesis; “long BTC after the print, three windows must not blow” is.

Not [BTC leads AI alts](20260911_btc_lead_ai_alts.en.md). Not [funding fade](20260910_funding_fade.en.md). Not [chip-spend QoQ vs BTC regime](20260914_ai_chip_spend_btc_regime.en.md).

中文：[20260914_ai_financing_btc_CN.md](20260914_ai_financing_btc_CN.md)

---

## How this sentence walks the repo

You ask a web AI: “Isn’t there some relationship between AI financings and BTC funding and returns?”

“A relationship” is not a hypothesis. A web AI will talk risk-on spillover, or tell you to sweep Crunchbase. This repo first writes an **executable contract**: locked public mega rounds, after date D closes, long BTC the next day for five UTC days; none of the three windows should blow the book. The relationship itself does not sit the court.

After the template passes and lineage is empty, Phase 1 runs: daily event windows and IC. Window dummy versus next-day return IC = −0.021 (p = 0.40) — no daily conditional expectation. The bull window loses to just holding BTC. The recent mean is one OpenAI print (+8.5%); the median is negative. AI-basket and BTC funding crowd together (IC 0.098) and that does not forecast return.

**The story pauses here.** The written falsifier is 2h CAGR and MaxDD. That book is empty. A flashlight, bright or dark, cannot `--declare` and cannot flip the sentence into a short. Until a human asks to finish the 2h book, the AI must not pretend the case is closed. Do not hand-trade “this financing is different.”

The seven sections below unpack the locked calendar, how to read IC, and what an empty table means.

---

## Design

```text
Human sentence (financing vs BTC funding and returns)
  → template: after date D closes, long BTC on D+1, hold 5 UTC days
  → validate + lineage
  → Phase 1: locked calendar vs daily return / funding (flashlight, does not close)
  → 2h court not run yet: build layer → mlbot research run → human --declare
```

Phase 1 IC or event-window means **cannot** change the sentence and **cannot** declare. The close waits for the two-hour book.

| Box | This sentence |
|---|---|
| Mechanism | Locked calendar `config/research/ai_financing_events.yaml` (public mega rounds, default ≥ $300M). Long BTC when the first 2h bar of D+1 has `ai_financing_event = 1`. |
| Regimes | Spillover years should pay; years where the print is cash leaving for compute should fail. Three windows must not blow. |
| Contract | Hold 5 UTC days (60×2h); no adds; kill switch off; closed-bar. The announcement day’s path is not entry information. |
| Falsifier | Any window CAGR < 0, or recent MaxDD deeper than both trend windows. Phase 1 IC / window means are not this line. |
| Landing | Robot: this folder’s pack. Hands: long BTC the day after the print, flatten five days later. |

Clock (2h court, not run yet):

| Moment | What happens |
|---|---|
| Calendar | UTC date D is the print; usable only after that day closes |
| Entry | First closed permit on D+1 |
| Exit | 60 two-hour bars (5 UTC days) |
| Flat days | Earn 0 |

---

## Data

```bash
mlbot research validate 20260914_ai_financing_btc
mlbot research index --trusted --query ai-financing
mlbot data download-funding-rate --symbols BTCUSDT \
  --start-year 2020 --start-month 1 --end-year 2026 --end-month 6
```

| Item | This machine |
|---|---|
| Calendar | [`config/research/ai_financing_events.yaml`](../../config/research/ai_financing_events.yaml) — locked public mega rounds (OpenAI / Anthropic / xAI / Inflection), not a Crunchbase dump |
| Event count | 21 |
| Phase 1 price | Binance Vision daily in `data/klines_vision/`, **not** written into tick parquet |
| 2h price | Court needs `data/parquet_data` |
| Funding | `mlbot data download-funding-rate` → `data/funding_rate/parquet` |
| Segments | [`config/market_segment.yaml`](../../config/market_segment.yaml) |
| 2h layer | Planned `features_ai_financing_btc_120T` — **do not run until it exists** |

| Window | Span | Phase 1 events | How to read |
|---|---|---:|---|
| `bear_2022` | 2022-01-01 → 2023-11-01 | 7 | Spillover, if any, must not blow the book. |
| `bull_2023_2024` | 2023-06-01 → 2025-01-01 | 9 | Risk-on years should light up. |
| `recent_range_to_bear` | 2025-01-01 → 2026-05-31 | 7 | Recent. Cannot promote alone. |

Bear ends 2023-11 and bull starts 2023-06, so **2023-06 → 2023-11 overlaps**. Inflection 2023-06-29 and Anthropic 2023-09-25 sit in both windows. 7+9+7 is not 23 distinct prints.

---

## Features

| Column | What it is | Closed-bar use |
|---|---|---|
| `ai_financing_event` | 1 on the first 2h bar of D+1 | Known at that close; next bar may open |
| `ai_financing_in_window` | Five UTC days from D+1 | Dummy for the Phase 1 daily align |
| `ai_financing_days_since` / `ai_financing_log_usd` | Distance and size | Description only; size-weighting is a new sentence |
| `funding_rate` / `funding_rate_zscore_50` | Existing BTC funding nodes | Previous bar only |
| `ai_basket_funding_zscore` | Equal-weight FET / RENDER / NEAR / TAO funding z | **Flashlight only** — does not change “long BTC” |

```bash
PYTHONPATH=src python scripts/research/ai_financing_scan.py
```

Two-hour court still needs a layer (five KPIs empty):

```bash
PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/experiments/20260914_ai_financing_btc/strategies/ai_financing_btc \
  --symbols BTCUSDT --timeframe 120T \
  --root feature_store --layer features_ai_financing_btc_120T \
  --data-path data/parquet_data \
  --start-date 2022-01-01 --end-date 2026-06-01
mlbot research run 20260914_ai_financing_btc
```

---

## IC

Phase 1 **has** IC. Flashlight only. This is a *time-series* Spearman: the 5-day window dummy versus next-day BTC return. n = 1611 labelled days. It is not a cross-sectional rank.

| Score | Label | Spearman IC | p | Read |
|---|---|---:|---:|---|
| 5-day window dummy | Next-day BTC return | **−0.021** | 0.40 | No conditional expectation. |
| AI-basket funding z (lag 1) | BTC funding z | **0.098** | 0.003 | Crowding moves together — coexistence, not a trade. |
| AI-basket funding z (lag 1) | Next-day BTC return | 0.022 | 0.43 | Shared crowding does not forecast return. |

A near-zero IC cannot replace the 2h book and cannot flip the sentence to short the print.

---

## Validation

Phase 1 scan is done. Two-hour five KPIs have **not** run.

### Phase 1 event window (closed calendar D+1 → D+5, does not close)

| Window | n | Event 5d mean | Median | Day win | Any-5d baseline | Funding 8h mean pre / post |
|---|---:|---:|---:|---:|---:|---:|
| `bear_2022` | 7 | +1.12% | +0.97% | 71% | +0.01% | 6.3×10⁻⁵ / 6.8×10⁻⁵ |
| `bull_2023_2024` | 9 | +0.04% | −0.24% | 44% | +1.24% | 9.4×10⁻⁵ / 11.7×10⁻⁵ |
| `recent_range_to_bear` | 7 | +0.25% | −0.76% | 14% | −0.15% | 2.2×10⁻⁵ / 2.4×10⁻⁵ |

| Window | How to read the row |
|---|---|
| Bear | Positive versus a random 5-day hold; median also plus. Seven prints cannot close. |
| Bull | Mean almost flat, median minus, **loses to just holding BTC** (baseline +1.24%). The year that should spill over goes dark. |
| Recent | Mean is one OpenAI print on 2026-02-27 (**+8.5%**); median −0.76%; 6 of 7 windows negative. The mean is lying. |

Post-event 8h funding is only slightly higher than pre-event, order 10⁻⁵ — not a `|z| ≥ 1.5` fade story.

### Two-hour court (empty until `mlbot research run`)

| Window | CAGR | Calmar | Win rate | MaxDD | Sharpe |
|---|---:|---:|---:|---:|---:|
| `bear_2022` | (not run) | — | — | — | — |
| `bull_2023_2024` | (not run) | — | — | — | — |
| `recent_range_to_bear` | (not run) | — | — | — | — |

Empty means the human has not asked to finish the 2h book. No `--declare` without five KPIs.

---

## Conclusion

Class remains **beta**. Phase 1 only says: on daily bars, the locked calendar does not give a “BTC should rise after the print” conditional expectation.

- Window dummy vs next-day return IC ≈ 0, p = 0.40.
- Bull loses to buy-and-hold BTC; recent median is negative, mean pulled by one print.
- AI-basket and BTC funding crowd together (IC 0.098) and that does not forecast return.
- **None of this declares.** The written falsifier is 2h CAGR and MaxDD. The book is empty.

`mlbot research close … --declare` waits for the 2h book. Do not type `verdict:`. Do not hand-trade “this financing is different” before the court runs.

| Path | What it is |
|---|---|
| `config/experiments/20260914_ai_financing_btc/quick_scan/ai_financing_scan.json` | Phase 1 scan |
| [DECISION.md](../../config/experiments/20260914_ai_financing_btc/DECISION.md) | Paper |
| 2h `results/…` | **Not yet** |

---

## How to read the report

1. **Ask whether the table is the court.** `ai_financing_scan.json` is a flashlight. The empty five-KPI table is the court. Do not treat a 5-day mean as CAGR.
2. **Median before mean.** Recent mean +0.25%, median −0.76%. The gap is 2026-02-27 +8.5%.
3. **The baseline is “any 5 days,” not cash.** Bull +0.04% vs +1.24% means “after the print, holding BTC lost to holding BTC anyway.”
4. **Do not double-count overlap events.** Two prints sit in both bear and bull.
5. **Pre / post funding is not fade.** 10⁻⁵ is not `|z| ≥ 1.5`.
6. **Basket funding IC 0.098 is coexistence.** Same column vs next-day return is 0.
7. **IC cannot rewrite the sentence** into a short, or into a Crunchbase dump.
8. **Empty `verdict` is not “fine.”** It only means the 2h book has not run.

The paper is [DECISION.md](../../config/experiments/20260914_ai_financing_btc/DECISION.md).
