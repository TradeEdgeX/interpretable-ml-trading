# Cross-section multi-factor (`cs_panel`)

**中文:** [cs_panel_CN.md](cs_panel_CN.md)

This repo can test **cross-section** claims (score a universe each day, long a quantile), not only single-name event clocks. The court is `cs_panel`, not `event_backtest`. `mlbot research run` does not dispatch it.

Capability is not edge. The measured A-share sentences are `--declare reject`. The framework can sit the exam; it does not promise a pass.

Walkthroughs (Design / Data / Features / IC / Validation / Conclusion / How to read the report):

| Sentence | Entry | Control | Close |
|---|---|---|---|
| [Hot names vs EW](examples/20260911_ashare_cs_mom_amount.en.md) | `python scripts/research/cs_panel.py` | Same-universe EW | Relative CAGR ≤ 0 in all three; IC minus; reject |
| [Hot sectors vs EW](examples/20260911_ashare_cs_sector_cost.en.md) | `python scripts/research/cs_sector.py` | Same-universe same-10bp EW | Bull −40pp, chop −25pp; reject |

---

## What it can test

Lock a few columns first (do not mine overnight). Each session, score every name in the universe, long the high side (or short the low side), and compare to same-universe equal-weight or cash. Report only CAGR / Calmar / win rate / MaxDD / Sharpe by window.

| Sentence type | Court entry | Measured example |
|---|---|---|
| Name multi-factor vs EW | `python scripts/research/cs_panel.py` | [Hot names vs EW](examples/20260911_ashare_cs_mom_amount.en.md) · reject |
| Sector first, then names | `python scripts/research/cs_sector.py` | [Hot sectors vs EW](examples/20260911_ashare_cs_sector_cost.en.md) · reject |
| Sector long–short residual | `python scripts/research/cs_sector.py --mode ls` | `20260911_ashare_cs_sector_alpha` · reject |
| Weekly sector exposure vs cash | `python scripts/research/cs_sector.py --mode weekly` | `20260911_ashare_cs_sector_beta` · reject |

Look up the harness:

```bash
PYTHONPATH=src python -m cli.main research harness ashare_cs_mom_amount
```

Calendar is [`config/market_segment_ashare.yaml`](../config/market_segment_ashare.yaml) `bear_2021` / `bull_924` / `chop_recent`. Do not reuse crypto dates.

---

## Clock (closed-bar)

The factor is known only at the **T close**. The first fill is the **next open**. One day of book return is next-open → next-next-open.

Do not decide at the open with the same-index completed FeatureStore row. Labels (next 20-day return) may look ahead; entry columns may not.

---

## Algorithm (do not change the sentence after the lock)

The public dummy locks two columns, equal weight:

| Column | Construction |
|---|---|
| `mom_20` | close / close 20 bars ago − 1 |
| `amount_z_20` | amount z versus itself over the last 20 bars |

Each day, take a **cross-sectional z** of each column, then `score = 0.5 * cs_z(mom) + 0.5 * cs_z(amount_z)`.  
Book: equal-weight the top 20% of `score`. Control: equal-weight every name that could be scored that day.

Phase 1 may compute Spearman IC of that day’s score versus the next-open 20-day return. IC is a flashlight. **It cannot close the case and cannot send you back to swap columns.**

Sector sentences use the same pair: equal-weight mean inside the sector, z across sectors, buy names in hot sectors (or short cold ones). Industry is the 20-bucket coarse snapshot in `config/industry_map_ashare.yaml`, not PIT 申万.

A new sentence needs a new `DECISION.md` with columns and control locked first. Do not `compute_*` a new factor in the backtest; register it in FeatureStore and backfill.

---

## How a human walks it

```text
Human sentence (columns locked first)
  → template: sociology / math / stats / ruler / A-share three windows / five boxes
  → mlbot research validate <id>
  → mlbot research index --trusted --query cross-section
  → run cs_panel / cs_sector only after “measure this”
  → human --declare
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

## Measured closes (not recommended strategies)

Versus **same-universe equal-weight** (one share each, almost no turnover), the locked hot-name / hot-sector / reversal / sector long–short books have no stable edge.  
Weekly hot sectors versus **cash** print positive CAGR in all three windows, but still lose to EW in bull and chop. A human `--declare reject`ed all of them.

Equal-weight itself is green in these three windows: that is small-cap market beta, not a stock-picking logic. Fine print is in the two walkthroughs.

---

## How to read a cross-section report

| Look first | Then | Do not |
|---|---|---|
| Same-window, same-universe control CAGR | Absolute CAGR of the high book | Score +24% against cash and call it “it makes money” |
| Relative gap (high − EW) | Sign of IC | Close on IC, or flip the paper into a reversal because IC is minus |
| Turnover and one-way cost | Gross return | “We would have passed if fees were zero” — cost is the contract |
| Day count / name count | Win rate | Treat a ~50% win rate as “the side is right” |

IC is Spearman of that day’s score versus the next-open 20-day return, then averaged across days. Labels may look ahead; entry columns may not. A rising or falling IC cannot change the sentence.
