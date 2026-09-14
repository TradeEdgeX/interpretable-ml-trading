# Score the whole market every day, buy the hottest sleeve

**中文:** [cs_panel_CN.md](cs_panel_CN.md)

This repo can measure more than a golden-cross on one name, in and out on a clock.

Some sentences say something else: every session, score **every name that can be scored**, buy the hottest sleeve, and compare to the same day’s names, one share each. If that wins, the sentence lives. If it loses, the sentence dies.

Being able to measure this is not the same as having an edge. Two sentences already walked it. A human judged both false.

| Sentence | What it compares | Human close |
|---|---|---|
| [Do hot winners keep beating equal-weight?](examples/20260911_ashare_cs_mom_amount.en.md) | Buy the hottest 20% each day; control is one share each | Loses in all three windows; high score predicts a turn, not a continuation. False |
| [Are hot sectors better than the same names, same fee?](examples/20260911_ashare_cs_sector_cost.en.md) | Score sectors first, then buy the names inside; 10bp each side | Bull −40 points, chop −25 points. False |

Walkthroughs still use the same seven sections: Design / Data / Features / IC / Validation / Conclusion / How to read the report.

---

## What question is this?

Lock the scoring columns first. Do not swap factors overnight and measure again.

Each close: score every name that can be scored that day. Next open: buy the high side (or short the low side). The control is those same names, one share each — or cash.

Report only annual speed / Calmar / win rate / max drawdown / Sharpe by window.

| Kind of sentence | Measured example |
|---|---|
| Rank names each day vs one-share-each | [Hot names](examples/20260911_ashare_cs_mom_amount.en.md) · false |
| Score sectors first, then buy names | [Hot sectors](examples/20260911_ashare_cs_sector_cost.en.md) · false |
| Long hot sectors, short cold ones | `20260911_ashare_cs_sector_alpha` · false |
| Weekly hot sectors vs cash | `20260911_ashare_cs_sector_beta` · false |

The calendar must be the A-share three windows (2021 bear, 924 bull, recent digestion). Do not reuse crypto 2022.

---

## When you know, when you can buy

Today’s score is known only at the **close**. The first fill is the **next open**. One day of the book is next open → the open after that.

Do not decide at today’s open with today’s already-finished numbers. The “next 20 days’ return” used as a label may look ahead; the score used to enter may not.

---

## How the public practice sentence scores

Two locked columns, half and half:

1. How much it rose over 20 days: today’s close / close 20 bars ago − 1
2. How hot its own amount is versus **itself** over the last 20 bars

Each day, standardize both columns across names that can be scored that day, then take half and half. Buy the top 20%, one share each. Control: every name that could be scored that day, one share each.

Whether today’s score and the next 20 days move together is a flashlight only. **It cannot close the case and cannot send you back to swap columns.**

The sector sentence uses the same pair: average inside the sector first, then rank sectors, buy names in the hottest 20% sectors. Industry is a 20-bucket coarse snapshot in this repo, not PIT 申万.

A new sentence needs a new paper, with columns and control locked first. If a column is missing, add it to the pre-built table first; do not compute a new one while printing the book.

---

## How a human walks it

```text
Human sentence (columns locked first)
  → write down: who pays, what is measured, which dates, how you lose
  → check: this sentence was not already closed
  → after “measure this,” print “score every day, buy the hottest sleeve”
  → human writes whether it holds
```

```bash
PYTHONPATH=src python -m cli.main research validate 20260911_ashare_cs_mom_amount
PYTHONPATH=src python scripts/research/cs_panel.py
PYTHONPATH=src python scripts/research/cs_sector.py
PYTHONPATH=src python scripts/research/cs_sector.py --mode ls \
  --out results/ashare_cs_sector/experiments/20260911_ashare_cs_sector_alpha
PYTHONPATH=src python scripts/research/cs_sector.py --mode weekly \
  --out results/ashare_cs_sector/experiments/20260911_ashare_cs_sector_beta
```

Daily default `data/ashare/daily/`. Listing file `data/ashare/stock_basic/stock_basic.parquet`. `data/` is not in git.

---

## What was already measured (not recommended strategies)

Versus **the same day’s names, one share each** (almost no turnover), the locked hot-name / hot-sector / reversal / sector long–short books have no stable edge.

Weekly hot sectors versus **cash** can print a positive annual speed in all three windows, but still lose to one-share-each in bull and chop. A human judged all of them false.

One-share-each itself is green in these three windows: that is the small-cap market rising, not stock-picking. Fine print is in the two walkthroughs.

---

## How to read the table

| Look first | Then | Do not |
|---|---|---|
| Same-window, same-names control annual speed | The hottest sleeve’s own speed | Score +24% against cash and call it “it makes money” |
| Relative gap (hottest − one-share-each) | Whether score and later return share a sign | Close on that sign, or flip the paper into a reversal because it is minus |
| Turnover and one-way cost | Return before the fee | “We would have passed if fees were zero” — cost was in the contract |
| Day count / name count | Win rate | Treat a ~50% win rate as “the side is right” |

Whether today’s score and the next 20 days’ open-to-open return move together is averaged across days. Labels may look ahead; the entry score may not. A rising or falling number cannot change the sentence.
