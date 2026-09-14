# When AI chip spend accelerates, is BTC in a bear?

Paper: [config/experiments/20260914_ai_chip_spend_btc_regime/](../../config/experiments/20260914_ai_chip_spend_btc_regime/)  
Class first as **beta / regime coexistence**. Harness is **phase1_scan_only**. Y is closed-bar MA200, not the segment labels. Human already `--declare reject`.

> When AI financing / compute spend is “too high,” BTC is in a bear or a range — not whether the next few days rise.

Not [long BTC after an AI financing print](20260914_ai_financing_btc.en.md). Not [BTC leads AI alts](20260911_btc_lead_ai_alts.en.md). Those sentences ask about a path after a print or a rip. This one asks: on acceleration days, which side of the average does the close sit on?

中文：[20260914_ai_chip_spend_btc_regime_CN.md](20260914_ai_chip_spend_btc_regime_CN.md)

---

## How this sentence walks the repo

You ask a web AI: “When AI compute spend is huge, cash is buying chips — should BTC be in a bear?”

It talks macro drainage and the next few days’ return. If Y is “the calendar is named bear,” the 2023 bounce is counted inside the 2022 window. If X is “the bill is already large,” high days lock to calendar year. Change the model, change which lab stars in the story.

You say the same sentence here. The template narrows it: not the next few days — closed-bar regime. X is Epoch chip-sales **QoQ** (expanding median, readable the day after quarter-end). Y is close versus MA200. The three calendar names only slice; they are not Y. The court is a Phase 1 scan: no 2h YAML, no CAGR table — it counts shares.

Pooled: acceleration days are below MA200 13.4% of the time; slower days 50.5%. Diff −37pp, the wrong sign. 2023’s bill doubled inside a BTC slow bull. A human `--declare reject`ed. Two **coexisting** regimes, not “spend up → press BTC below the average.” Do not flip into a short. Do not merge with “long BTC after a financing print.”

The seven sections below unpack the two tags and how to read the share table.

---

## Design

```text
Human sentence (too much spend → BTC bear / range, not next-few-days return)
  → template: X = quarterly chip-bill QoQ; Y = close below MA200
  → validate + lineage
  → Phase 1: tag each closed BTC daily bar, count non-bull share on high vs low days
  → human --declare reject
```

No 2h entry/exit YAML and no five-KPI CAGRs. The court tags **every closed BTC daily bar** with two slips, then counts shares. Independent bull/bear is about two windows, so Phase 1 cannot close as a five-KPI court; the written falsifier *is* the share, and the share is hit.

| Box | This sentence |
|---|---|
| Mechanism | After a completed Epoch chip-sales quarter, if QoQ is above its expanding median, BTC daily closes sit below the 200-day more often. |
| Regimes | Acceleration quarters should be non-bull. If acceleration sits inside a slow bull, the sentence fails. |
| Contract | On high days, do not long BTC on a 2023–24 bull template. Not a short. No adds. Closed-bar. |
| Falsifier | High-day non-bull share ≤ low-day share, or any sliced window flips sign. |
| Landing | Same sentence for hands. Machine: Phase 1 scan in this folder. No 2h YAML yet. |

```bash
mlbot research validate 20260914_ai_chip_spend_btc_regime
mlbot research index --trusted --query ai-chip-spend
PYTHONPATH=src python scripts/research/ai_chip_spend_regime_scan.py
mlbot research close 20260914_ai_chip_spend_btc_regime --declare reject
```

---

## Data

| Item | This machine |
|---|---|
| X | [`config/research/ai_chip_sales_quarterly.csv`](../../config/research/ai_chip_sales_quarterly.csv) (Epoch AI chip sales, CC BY) |
| Dollar amount | Sum of that quarter’s chip `Cost Estimate (USD)` |
| Readable | The UTC day **after** that quarter’s `end_date` |
| Y price | BTC daily closed bar |
| Segment calendar | [`config/market_segment.yaml`](../../config/market_segment.yaml) **slices only — not Y** |
| 2022Q4 / 2026Q2 | Marked incomplete in the source; still in the main table; the scan will slice |
| Harness | `phase1_scan_only` |

Do not use the dollar **level**. The bill walks from about $2.5B to $60B+. “Bill is large” locks to calendar year. The table uses **how fast it is rising**.

---

## Features

This court does not read a FeatureStore layer. X is the locked CSV; Y is daily close versus MA200 inside the scan. There is no 2h backtest to hide a `compute_*` in.

### X: is this day “spend accelerating”?

Sum Epoch chip `Cost Estimate` each completed calendar quarter. QoQ exists only from 2022Q2 (you need a previous quarter).

“High” is not a fixed number. It is: is this quarter’s QoQ above the **expanding median of every QoQ seen so far** (need four QoQs before the flag exists)? That keeps future quarters out of the past.

The quarter’s number is readable only the **next UTC day**. A 2023Q2 doubling is unknown on 2023-06-30. From 2023-07-01 every daily bar carries “last completed quarter = 2023Q2, QoQ +98%, high” until 2023Q3 prints.

2022 has no high/low flag, so it never enters the table. First tagged day: 2023-04-01 (2023Q1, +14%, high).

| Readable span (UTC) | Last completed quarter | QoQ | High? |
|---|---|---:|:---:|
| 2023-04-01 → 2023-06-30 | 2023Q1 | +14% | high |
| 2023-07-01 → 2023-09-30 | 2023Q2 | +98% | high |
| 2023-10-01 → 2023-12-31 | 2023Q3 | +55% | high |
| 2024-01-01 → 2024-03-31 | 2023Q4 | +27% | high |
| 2024-04-01 → 2024-06-30 | 2024Q1 | +56% | high |
| 2024-07-01 → 2024-09-30 | 2024Q2 | +21% | low |
| 2024-10-01 → 2024-12-31 | 2024Q3 | +21% | low |
| 2025-01-01 → 2025-03-31 | 2024Q4 | +22% | high |
| 2025-04-01 onward | 2025Q1 and later | +3% to +20% | low |

### Y: is this day “non-bull”?

One sentence: is this daily close below its own 200-day average? Below = non-bull = 1, else 0.

`bear_2022` / `bull_2023_2024` / `recent_range_to_bear` are **not** Y. They only slice the same X/Y. A calendar named “bear” does not write Y=1; a calendar named “bull” does not write Y=0.

Need 200 daily bars for the average. Days missing X or Y are dropped.

| Column | Meaning |
|---|---|
| Scope | Which dates. **Pooled** = every day that has both X and Y. |
| High non-bull | Share of *high-QoQ* days with close < MA200. The paper wants this **large**. |
| Low non-bull | Same share on low-QoQ days. Control. |
| Diff | High minus low, in percentage points. The paper wants a **plus**. Zero or minus hits the falsifier. |
| n high / n low | Day counts. A zero on either side means no comparison. |

---

## IC

**There is no return IC table, and there should not be one.**

Y is not “do the next few days rise.” Using a window dummy versus next-day return rewrites the sentence into the [financing-day paper](20260914_ai_financing_btc.en.md). Treating QoQ as a continuous score versus next-day return is a new folder. The flashlight *is* the share table.

---

## Validation

Ruler, written first: high non-bull share **≤** low, or any sliced window flips sign.

Pooled 13.4% / 50.5% / −37.1pp / 546 / 610 reads as:

- 1,156 tagged days (546 + 610).
- 546 high-QoQ days: **73** closes below MA200 (13.4%).
- 610 low-QoQ days: **308** closes below MA200 (50.5%).
- Acceleration days are *less* often below the average by 37 percentage points.

| Scope | High non-bull | Low non-bull | Diff | n high / n low |
|---|---:|---:|---:|---:|
| Pooled | 13.4% | 50.5% | **−37.1pp** | 546 / 610 |
| `bear_2022` | 27.7% | no sample | — | 213 / 0 |
| `bull_2023_2024` | 14.9% | 39.7% | **−24.8pp** | 396 / 184 |
| `recent_range_to_bear` | 15.6% | 55.1% | **−39.5pp** | 90 / 425 |

| Scope | How to read |
|---|---|
| Pooled | Already the wrong sign: acceleration days sit *above* MA200 more often. |
| `bear_2022` (through 2023-11-01, slice only) | 2022 has no high/low flag. The measurable slice is 2023-04-01 → 2023-10-31, ~213 days, all 2023Q1–Q3, all high. n_low = 0, no diff. The calendar name is “bear”; the measurable days are already the 2023 bounce (only 27.7% still below MA200). |
| `bull_2023_2024` | High days are the 2023–mid-2024 doubling quarters: 14.9% below MA200. Low days are late-2024 QoQ back near the median: 39.7%. Diff −24.8pp. |
| `recent_range_to_bear` | High is only the 2024Q4 slip (Jan–Mar 2025, 90 days), 15.6%. From 2025Q1 QoQ is low (425 days), 55.1% non-bull. Diff −39.5pp. The *level* is still huge; the *speed* is down; BTC spends more time under MA200. |

Every slice with two sides has the opposite sign. Falsifier hits.

---

## Conclusion

In 2023 the chip bill went from about $3.2B to about $12.6B while BTC walked a pre-/post-halving slow bull, mostly above MA200. The paper read “labs paying accelerator bills” as “cash leaves crypto, BTC should be non-bull.” This machine sees two **coexisting** regimes, not “spend up → press BTC below the average.”

From 2025 the level is higher still, QoQ is back below the expanding median, and BTC is *more* often under MA200. That is the same risk-on cycle, not a tradable lead–lag.

Class: **beta / regime coexistence**. The machine does not run “don’t long BTC when spend accelerates”; a human should not either. Declared reject.

Artifact: `config/experiments/20260914_ai_chip_spend_btc_regime/quick_scan/chip_spend_regime.json`

---

## How to read the report

1. **This is not a CAGR table.** 13.4% is “share of these days below the average,” not a return and not a five-KPI court.
2. **Diff needs n on both sides.** `bear_2022` n_low = 0 cannot support the sentence.
3. **The calendar name is not Y.** The 2022 window is called bear; the tagged days are the 2023 bounce.
4. **Level ≠ QoQ.** 2025 bills are larger, QoQ is low, non-bull is higher. The sentence locked speed.
5. **Expanding median has no future.** 2023Q1 “high” uses only QoQs known then.
6. **Readable the day after quarter-end.** 2023-06-30 cannot use 2023Q2 +98%.
7. **Do not flip into a short.** The contract is “don’t long on a bull template,” not “short.”
8. **Do not merge with the financing-day paper.** That X is 21 calendar events; this X is last-quarter QoQ on every daily bar.

The paper is [DECISION.md](../../config/experiments/20260914_ai_chip_spend_btc_regime/DECISION.md).
