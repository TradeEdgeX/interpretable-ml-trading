# Example: do tenbaggers need small cap and a 3–4 year hold?

Landing: [README.md](../../README.md).  
Paper: [config/experiments/20260911_tenbagger_smallcap/](../../config/experiments/20260911_tenbagger_smallcap/).  
Harness is **cohort_hold**, not 2h `event_backtest`. Declared **reject**.

Unrealized “who 10×s next” cannot be measured. The historical rule can.

```bash
PYTHONPATH=src python -m cli.main research validate 20260911_tenbagger_smallcap
PYTHONPATH=src python -m cli.main data download-ashare \
  --universe listed,delisted --start-date 2016-01-01 --backend sina
PYTHONPATH=src python scripts/research/cohort_hold.py --mcap-yi 100 --hold-years 3,4
```

This machine: 5213/5215 listed names have daily qfq; 255 post-2014-06 delists have tape (older deaths missing on Sina). Cap = that day’s amount / (turnover/100). `n` is name × entry quarter.

**3y.** crash_2015 small −4.26% vs large −4.15% (both 0 tenbaggers). bear_2018 small +2.75% vs large +3.31%. covid_2020 small +7.75% vs large +5.96%. bear_2021 small +9.09% vs large −0.03%. `bull_924` / `chop_recent`: hold not finished → no sample.

**4y.** crash_2015 still worse for small. Later windows: small-cap book CAGR ahead; 10× **rate** stays 0–0.4% on both sides.

All completed 3y holds: 10× rate **0.11%** vs **0.11%** (88 vs 33 hits). Same density; small names are just a bigger pool. Drop-Top-3 barely moves the rate. Small 3y paths include 1520 delists, large 216.

Falsifier already hits crash_2015 (3y/4y) and bear_2018 3y (relative CAGR ≤ 0). Closed with `--declare reject`.

Class: **thin fat tail, mainly small-cap beta.** Counting names that later 10×d is survivorship, not an entry-date density.

US: same math, not measured here (no public US downloader).

中文：[20260911_tenbagger_smallcap_CN.md](20260911_tenbagger_smallcap_CN.md)
