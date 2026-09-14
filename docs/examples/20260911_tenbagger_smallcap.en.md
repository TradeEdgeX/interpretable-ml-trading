# Do tenbaggers need small cap and a 3–4 year hold?

Paper: [config/experiments/20260911_tenbagger_smallcap/](../../config/experiments/20260911_tenbagger_smallcap/)  
Class first as **fat tail / small-cap beta**, not point-selection alpha. Dropping Top-3 (or the largest tenbaggers) is classification only.  
Harness: **cohort_hold**, not 2h `event_backtest`. Human already `--declare reject`.

> A-share names that go 10× are usually under 10 billion yuan at entry and you have to hold three or four years.

The court measures a **historical rule that already happened**: on the entry date the name was small, you hold a fixed 3 or 4 trading years — is 10× denser than large caps, and is relative CAGR better? It **cannot** measure “who 10×s next.” That sentence has no tape and no closed-bar column.

中文：[20260911_tenbagger_smallcap_CN.md](20260911_tenbagger_smallcap_CN.md)

---

## How this sentence walks the repo

You ask a web AI: “Are tenbaggers always small-cap, held three or four years? Find me the next one.”

It starts listing names and stories. A story that has not happened has no tape and no closed-bar column. **This repo cannot measure it.** That is research, not a hypothesis. Counting how many later-10× names were small is a future label — the pool is larger, so the headcount is larger.

What you can say here is a rule that **already happened**: “entry-date cap ≤ 10 billion yuan, hold 3 or 4 years; is 10× denser than large caps that day, and is relative CAGR better?” Class: fat tail / small-cap beta. The court changes: `mlbot research run` only dispatches 2h event backtests; this sentence uses `cohort_hold`. Cap is that day’s amount / turnover, not today’s cap filled backward. The 924 bull cannot finish a 3y hold — “no sample,” not “small caps are working lately.”

The table: 3y 10× rate 0.11% vs 0.11% (88 vs 33 hits, same density). Crash entries: small loses to large on CAGR. A human `--declare reject`ed. Names that are not listed yet have no history; post-hoc counts are survivorship; reject binds hands as well as the robot. US math is the same and was not measured — do not paste this table onto Nasdaq.

The seven sections below unpack what can and cannot be measured.

---

## Design

```text
Human sentence (tenbaggers are small-cap, hold 3–4 years)
  → template splits: historical rule yes; “who’s next” no
  → validate + lineage
  → download listed + delisted daily, quarterly PIT cap — only after “measure this”
  → cohort_hold: entry-date buckets, small vs large, fixed 3y / 4y
  → human --declare reject
```

Do not use `mlbot research run`. The court is `scripts/research/cohort_hold.py`.

| What the human wants | This repo |
|---|---|
| Research **future** tenbaggers, story still unfinished | **No use.** No tape, no closed-bar column. |
| Measure a **historical rule**: small on the entry date, hold 3–4 years, is 10× denser than large? | **Useful.** Fat tail / small-cap beta. |

| Box | This sentence |
|---|---|
| Mechanism | Enter names with entry-date float cap ≤ 10 billion yuan; hold 3 or 4 trading years (252×3 / 252×4 daily bars); no timing adds. |
| Regimes | Small-cap pool should have a thicker right tail. Crash entries should look ugly on most paths. |
| Contract | Time exit. Do not label “tenbagger” after the fact and look back at cap. Closed-bar cap. Delists use the last close. |
| Falsifier | Any window: small-minus-large CAGR ≤ 0, or 10× rate no higher than the control *and* deeper drawdown. Recent cannot promote alone. |
| Landing | Robot: `scripts/research/cohort_hold.py`. Hands: buy only names that were small *that day*, write the entry date, hold the years. |

Headline five KPIs plus the 10× *rate* for classification. Do not close on the rate alone.

---

## Data

```bash
mlbot research validate 20260911_tenbagger_smallcap
mlbot data download-ashare \
  --universe listed,delisted --start-date 2016-01-01 --backend sina --workers 4
PYTHONPATH=src python scripts/research/cohort_hold.py \
  --mcap-yi 100 --hold-years 3,4
```

| Item | This machine |
|---|---|
| Universe | Listed Shanghai / Shenzhen A-shares + delists after 2014-06 |
| Listed tape | 5213 / 5215 names have qfq daily |
| Delist tape | 255 post-2014-06 deaths have a tape (older Sina deaths missing; 20 codes still empty) |
| Cap | **That day’s** amount / (turnover/100), not today’s cap filled backward |
| PIT | 46 quarterly points (`universe_hist`) |
| Finished 3y | 34 entry quarters |
| Finished 4y | 30 entry quarters |
| Meaning of `n` | **name × entry quarter**, not distinct codes |
| US | Same math, no public US downloader in this extract — **not measured** |

Windows are the **entry-date** segment, not the hold’s end date:

| Window | Entry span | Can a 3–4y hold finish? |
|---|---|---|
| `crash_2015` | 2015-06-01 → 2016-02-29 | Yes |
| `bear_2018` | 2018-01-01 → 2019-01-31 | Yes |
| `covid_2020` | 2020-01-01 → 2020-03-31 | Yes |
| `bear_2021` | 2021-07-01 → 2022-10-31 | Yes |
| `bull_924` | 2024-09-24 → 2025-05-31 | **No sample** (still short of 3y as of 2026-09) |
| `chop_recent` | 2025-06-01 → 2026-09-10 | **No sample** |

---

## Features

This court does not read a golden-cross `features.yaml` layer. Cap is computed on the entry date inside `cohort_hold`. Do not look back from “became a tenbagger later.”

| Series | Construction | Closed-bar use |
|---|---|---|
| qfq close | Daily | Path uses realized prices; delist = last close |
| Amount / turnover | That day’s tape | Cap = amount / (turnover/100) on the entry date only |
| Small / large | Entry cap ≤ 100亿 vs > 100亿 | Group known at entry; later cap growth does not re-bucket |
| 10× | Terminal / entry ≥ 10 | The *label* may look ahead; the *group* may not |

Names missing entry-date cap are dropped. Do not back-fill today’s total cap.

---

## IC

**There is no IC table, and there should not be one.**

IC is score versus future return. This sentence has no daily score and no top 20%. It asks about cohort density. Treating entry cap as a continuous score is a new folder.

---

## Validation

```bash
PYTHONPATH=src python scripts/research/cohort_hold.py --mcap-yi 100 --hold-years 3,4
mlbot research close 20260911_tenbagger_smallcap --declare reject
```

### 3-year hold (equal-weight cohort book)

| Window | Book | CAGR | Calmar | Win rate | MaxDD | Sharpe | n | 10× rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `crash_2015` | small | **−4.26%** | −0.09 | 13.2% | −47.31% | −0.15 | 2200 | 0.00% |
| `crash_2015` | large | −4.15% | −0.10 | 22.4% | −41.19% | −0.18 | 1543 | 0.00% |
| `bear_2018` | small | **+2.75%** | 0.08 | 62.5% | −35.12% | 0.26 | 9799 | 0.29% |
| `bear_2018` | large | +3.31% | 0.11 | 64.2% | −29.39% | 0.31 | 2487 | 0.24% |
| `covid_2020` | small | +7.75% | 0.28 | 69.6% | −27.31% | 0.59 | 2656 | 0.00% |
| `covid_2020` | large | +5.96% | 0.23 | 63.9% | −25.47% | 0.53 | 776 | 0.13% |
| `bear_2021` | small | +9.09% | 0.26 | 61.7% | −34.92% | 0.50 | 14361 | 0.09% |
| `bear_2021` | large | −0.03% | −0.00 | 42.6% | −35.34% | 0.09 | 6007 | 0.12% |
| `bull_924` / `chop_recent` | both | no sample | — | — | — | — | 0 | — |

| Window | Small − large CAGR | 10× rate (small vs large) | Falsifier |
|---|---:|---|---|
| `crash_2015` | **−0.11pp** | both 0 | Hit |
| `bear_2018` | **−0.56pp** | 0.29% vs 0.24% | Hit |
| `covid_2020` | +1.79pp | 0.00% vs 0.13% | Small CAGR better, 10× rarer |
| `bear_2021` | +9.12pp | 0.09% vs 0.12% | Small CAGR better, 10× still not denser |

### 4-year hold

| Window | Book | CAGR | Calmar | Win rate | MaxDD | Sharpe | n | 10× rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `crash_2015` | small | **−3.07%** | −0.06 | 19.0% | −51.20% | −0.06 | 2200 | 0.00% |
| `crash_2015` | large | −2.78% | −0.06 | 27.9% | −44.62% | −0.07 | 1543 | 0.00% |
| `bear_2018` | small | +4.01% | 0.11 | 64.3% | −35.12% | 0.32 | 9799 | 0.24% |
| `bear_2018` | large | +3.19% | 0.11 | 59.2% | −29.39% | 0.28 | 2487 | 0.12% |
| `covid_2020` | small | +6.06% | 0.19 | 54.6% | −32.22% | 0.42 | 2656 | 0.04% |
| `covid_2020` | large | +4.50% | 0.16 | 52.4% | −27.49% | 0.38 | 776 | 0.13% |
| `bear_2021` | small | +13.79% | 0.40 | 66.0% | −34.87% | 0.66 | 11198 | 0.38% |
| `bear_2021` | large | +3.66% | 0.10 | 46.2% | −35.14% | 0.28 | 4873 | 0.41% |
| `bull_924` / `chop_recent` | both | no sample | — | — | — | — | 0 | — |

`crash_2015` 4y relative CAGR is still negative. The falsifier already hits on crash entries.

### All finished cohorts (coverage only)

| Hold | Small CAGR | Large CAGR | 10× rate (small vs large) | Hits (small vs large) |
|---|---:|---:|---|---|
| 3y | +2.58% | +0.47% | **0.11% vs 0.11%** | 88 vs 33 |
| 4y | +3.49% | +1.25% | 0.14% vs 0.21% | — |

Small 3y paths include **1520** delists, large **216**. Drop-Top-3 barely moves the rate.

---

## Conclusion

Class: **thin fat tail, mainly small-cap beta.**

- “Tenbaggers are small-cap” counts names after the fact. Entry-date **density** is the same (0.11% vs 0.11% at 3y). Small names are just a bigger pool.
- Crash entries: small loses to large on CAGR. 2018 3y also. Falsifier hits.
- `bear_2021` small CAGR beats large — that is small-cap beta, not a denser 10× (0.09% vs 0.12%).
- US not measured. Do not paste this table onto Nasdaq.

Closed `--declare reject`. The robot does not run “buy then-small names and wait for 10×”; a human should not either.

Artifact: `results/tenbagger_smallcap/experiments/20260911_tenbagger_smallcap`

| Example | Can measure | Cannot measure |
|---|---|---|
| Golden cross / funding / Monday / alts / P99 | A contract that already happened | “It will work next time” |
| This page | Historical multiple distribution of small-cap holds | Which name 10×s next |

---

## How to read the report

1. **`n` is name × quarter.** 14361 is not 14361 distinct small caps.
2. **Entry window first, then relative CAGR.** `bear_2021` +9pp does not save `crash_2015`.
3. **10× rate is density, not a headcount.** 88 vs 33 hits look like a small-cap win; divide by the pool and both are 0.11%.
4. **Win rate is paths with multiple > 1**, not “tenbagger win rate.” Crash small 13.2% never even got capital back.
5. **Delists are the mechanism.** Drop them and the 10× density looks fake.
6. **`bull_924` no-sample is not a recent story.** The hold is not finished.
7. **Top-3 is classification.** The rate barely moves.
8. **Pooled 3y small +2.58% is coverage.** The ruler says “any window.” Crash already died.
9. **reject binds hands as well as the robot.**

The paper is [DECISION.md](../../config/experiments/20260911_tenbagger_smallcap/DECISION.md).
