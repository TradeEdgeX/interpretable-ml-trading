# Do hot winners keep beating equal-weight?

Paper: [config/experiments/20260911_ashare_cs_mom_amount/](../../config/experiments/20260911_ashare_cs_mom_amount/)  
Class first as **beta**: this buys the “rose a lot and traded hot” sleeve, not stock-picking. Industry and size were not stripped out first. A human already judged the sentence false.

> Names that rose the most over 20 days *and* are hot on their own amount keep beating same-universe equal-weight over the next 20 days.

This sentence scores the whole market every day on two locked columns: how much it rose, and how hot its own amount is. It is not index timing, and it is not an overnight hunt for a better factor. How score lines up with later return is a flashlight only; it cannot change the sentence.

Idea, report, and how to read it: [cs_panel.en.md](../cs_panel.en.md). Sector ranking is a different sentence: [hot sectors vs equal-weight](20260911_ashare_cs_sector_cost.en.md).

中文：[20260911_ashare_cs_mom_amount_CN.md](20260911_ashare_cs_mom_amount_CN.md)

---

## How this sentence walks the repo

You ask a web AI: “Do hot winners keep beating the market? Sweep whatever has a high IC overnight.”

Mining factors overnight and promoting when IC rises is a different loop. It optimizes a score. This repo optimizes whether **one sentence is dead**. The two columns lock first: `mom_20` and `amount_z_20`, half and half. IC is a flashlight. It cannot change the sentence and cannot close the case.

After “measure this,” print a table that scores every name each day and buys the hottest sleeve — not a single-name in-and-out book. Score at the close, buy the top 20% next open; the control is the same day’s names, one share each. In the 924 bull, equal-weight prints +76.57% — that is the small-cap market, not this sentence. The hottest sleeve is +24.13%, −52 points relative. Score versus the next 20 days is minus in all three windows: high score predicts a turn, not a continuation.

A human already judged it false. Do not flip this paper into a reversal (new sentence). Do not score the hot sleeve’s own speed against cash. Do not buy the hottest quintile at night after the table said no. Class: momentum + activity beta, measured useless.

The seven sections below unpack “score at the close, buy next open” and how to read the numbers.

---

## Design

```text
Human sentence (two columns locked)
  → write down: who pays, what is measured, which dates, how you lose
  → check: this sentence was not already closed
  → first see whether score and later return share a sign (flashlight only)
  → buy the hottest 20% each day; control is one share each
  → human writes whether it holds
```

This is not a single-name in-and-out book. Every day, rank the names that can be scored, buy the hottest 20%, and compare to the same day’s names one share each.

| Box | This sentence |
|---|---|
| Mechanism | At the close, `score = 0.5 * cs_z(mom_20) + 0.5 * cs_z(amount_z_20)`; buy the top 20% equal-weight. |
| Regimes | Momentum should light up in the trend window; the same exposure must not lose to equal-weight in the bear. |
| Contract | Next open in; one day at a time; fully invested top 20%; no short, no adds. Closed-bar. |
| Falsifier | Any window: top-minus-EW CAGR ≤ 0. IC cannot close alone. |
| Landing | Robot: `cs_panel.py`. Hands: at the close, buy the hottest quintile next open. Do not chase the same bar. |

Clock:

| Moment | What happens |
|---|---|
| T close | `mom_20` / `amount_z_20` / `score` become known |
| T+1 open | First fill, top 20% |
| T+2 open | One day of book return (next open → next-next open) |
| Control | Same-day scoreable universe, equal-weight, almost no turnover |

Score-weighted books are robustness, not a rewrite.

---

## Data

```bash
mlbot research validate 20260911_ashare_cs_mom_amount
mlbot research index --trusted --query cross-section
PYTHONPATH=src python scripts/research/cs_panel.py
```

| Item | This machine |
|---|---|
| Universe | `type=1` non-ST **5228** names; daily tape aligned **5175** |
| Bar | Daily |
| Path | `data/ashare/daily/` (not in git) |
| Listing file | `data/ashare/stock_basic/stock_basic.parquet` |
| How the table is made | Rank scoreable names each day, buy the hottest 20%, control is one share each |
| Calendar | [`config/market_segment_ashare.yaml`](../../config/market_segment_ashare.yaml) |
| Neutrality | **None** (industry or size) |
| Kill switch | Off |

| Window | Span | Book days | Role |
|---|---|---:|---|
| `bear_2021` | 2021-07-01 → 2022-10-31 | 323 | Hot names must not lose to EW. |
| `bull_924` | 2024-09-24 → 2025-05-31 | 163 | Momentum should pay. |
| `chop_recent` | 2025-06-01 → 2026-09-10 | 312 | Recent. Cannot pass on its own. |

EW will print a huge *absolute* CAGR in the 924 bull. That is small-cap market beta, **not** this sentence.

---

## Features

Two columns locked. Do not swap factors overnight. If a column is missing, add it to the pre-built table first; do not compute a new one while printing the book.

| Column | Construction | Closed-bar use |
|---|---|---|
| `mom_20` | close / close 20 bars ago − 1 | Known at T close; used T+1 open |
| `amount_z_20` | Amount z versus *itself* over 20 bars | Same |
| `cs_z(·)` | Cross-sectional z each day | Today’s closed cross-section only |
| `score` | 0.5 / 0.5 of the two z’s | Weights locked, not swept |
| Label (IC only) | Next-open 20-day open-to-open return | May look ahead; **must not** be an entry column |

Flipping this paper into a reversal is a new sentence.

---

## IC

Phase 1 has IC. Flashlight only. Spearman of that day’s `score` (and each column) versus the next-open 20-day return, then averaged across days.

| Column | `bear_2021` | `bull_924` | `chop_recent` |
|---|---:|---:|---:|
| `mom_20` | −0.071 | −0.121 | −0.056 |
| `amount_z_20` | −0.017 | −0.064 | −0.037 |
| `score` | **−0.053** | **−0.110** | **−0.054** |

| Read it as | Do not read it as |
|---|---|
| All three windows negative: high score predicts reversal, not continuation | “IC is negative, so short the top 20%” — new sentence |
| 924 bull is the most negative | “IC rose, so we can close” — IC never closes |
| Both columns share the minus sign | “Add a third column overnight” |

The book later dies in the same direction as the flashlight.

---

## Validation

Ruler, written first: any window top-minus-EW CAGR **≤ 0**.

### Top 20% EW vs universe EW (kill switch off)

| Book | Window | CAGR | Calmar | Win rate | MaxDD | Sharpe | Days |
|---|---|---:|---:|---:|---:|---:|---:|
| Top 20% | `bear_2021` | **−8.65%** | −0.28 | 51.7% | −30.36% | −0.27 | 323 |
| EW | `bear_2021` | +4.70% | 0.16 | 57.3% | −29.69% | 0.32 | 323 |
| Top 20% | `bull_924` | +24.13% | 0.93 | 49.1% | −25.99% | 0.69 | 163 |
| EW | `bull_924` | **+76.57%** | 4.47 | 53.4% | −17.15% | 1.57 | 163 |
| Top 20% | `chop_recent` | +13.84% | 0.71 | 55.1% | −19.45% | 0.68 | 312 |
| EW | `chop_recent` | +22.75% | 1.04 | 58.7% | −21.79% | 1.15 | 312 |

| Window | vs EW | Falsifier |
|---|---:|---|
| `bear_2021` | **−13.35pp** | Hit |
| `bull_924` | **−52.44pp** | Hit |
| `chop_recent` | **−8.91pp** | Hit |

`bull_924` +76.57% is the equal-weight control (small-cap market beta). The same window’s hot book is only +24.13%.

### Score-weighted robustness (not a rewrite)

`score_wt` = universe `max(score, 0)` weighted; `top_sw` = top 20% score-weighted.

| Book | Window | CAGR | Calmar | Win rate | MaxDD | Sharpe |
|---|---|---:|---:|---:|---:|---:|
| `score_wt` | `bear_2021` | −13.59% | −0.38 | 51.1% | −35.82% | −0.47 |
| `top_sw` | `bear_2021` | −18.77% | −0.46 | 49.8% | −40.37% | −0.68 |
| `score_wt` | `bull_924` | +7.63% | 0.25 | 49.1% | −31.09% | 0.38 |
| `top_sw` | `bull_924` | +1.52% | 0.05 | 49.1% | −33.49% | 0.26 |
| `score_wt` | `chop_recent` | +10.20% | 0.52 | 54.2% | −19.51% | 0.53 |
| `top_sw` | `chop_recent` | +7.64% | 0.38 | 53.2% | −20.10% | 0.42 |

In the 924 bull, equal-weight top 20% still printed +24.13%; score-weighting falls to +7.63% / +1.52%. The more you trust the score, the worse the relative book.

---

## Conclusion

Class: **beta continuation claim, measured useless (the sign is reversal).**

- Top 20% loses to EW in every window. Falsifier hits all three.
- IC is negative in every window, same direction as the book.
- A reversal needs a new paper. Do not flip this one.
- Industry / size neutrality is another paper (the sector sentence).

A human already judged it false. The robot does not run it; a human should not buy the hottest quintile versus one-share-each.

Artifact: `results/ashare_cs_mom_amount/experiments/20260911_ashare_cs_mom_amount/`

---

## How to read the report

1. **Same book, same window, then CAGR.** Do not score +24.13% against cash.
2. **EW green is not a pass.** 924 EW +76.57% is the market. The sentence asked for *relative* continuation (−52pp).
3. **Headline is relative CAGR, not ΣR and not IC.**
4. **Win rate near 50% does not save it.** EW often wins more days.
5. **163 bull days cannot promote alone** — and that window has the worst relative gap.
6. **Worse weighted books are the mechanism**, not bad luck in the equal-weight quintile.
7. **Do not add a third column overnight.**
8. **reject binds hands as well as the robot.**

The paper is [DECISION.md](../../config/experiments/20260911_ashare_cs_mom_amount/DECISION.md).
