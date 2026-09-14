# Does buying SPY / QQQ miss the US bull market?

Paper: [config/experiments/20260914_eq_us_spy_qqq_beta/](../../config/experiments/20260914_eq_us_spy_qqq_beta/)  
Class: **beta**. Harness is **eq_us_daily**, not 2h `event_backtest`. Declared **reject**.

Claim: buying SPY or QQQ “misses” the US bull; you must pick stocks or wait for an index washout.

```bash
mlbot research validate 20260914_eq_us_spy_qqq_beta
mlbot data download-us --symbols SPY,QQQ --start-date 2013-01-01
PYTHONPATH=src python scripts/research/eq_us_spy_qqq.py
mlbot research close 20260914_eq_us_spy_qqq_beta --declare reject
```

This machine: Nasdaq daily 2016-09-12 → 2026-09-11; court locked to 2026-08-17 (2,496 bars). Yahoo / Stooq were blocked. Split-adjusted close, not total return. S&P 500 PIT stock-picking was not re-run (no universe).

**Buy-and-hold.** SPY CAGR **13.7%** / MaxDD **−34.1%** / Sharpe 0.80. QQQ **20.4%** / **−35.6%** / 0.93.

**Same-window timing (QQQ).** RSI≤30 hold 40d: **10.6%** (21% in market). RSI to prior 252d high: **10.7%**. First −20% hold 60d: **8.4%**. MA200: **16.0%** / −21.9% MaxDD. All four SPY timing CAGRs sit at 5.0%–7.8% vs 13.7% buy-and-hold.

Windows: `us_bear_2022` SPY −31.0% / QQQ −42.5%; `us_bull_2023_2024` +24.2% / +39.3%; `us_recent` +19.0% / +25.0%. `us_covid_2020` is 24 bars — coverage only.

Class: **beta**. Washout timing loses to same-window buy-and-hold on CAGR. MA200 is insurance, not alpha. Closed with `--declare reject`.

中文：[20260914_eq_us_spy_qqq_beta_CN.md](20260914_eq_us_spy_qqq_beta_CN.md)
