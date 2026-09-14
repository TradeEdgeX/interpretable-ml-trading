# Gallery

**One-liner:** Measured sentences grouped by phenomenon. Each card is the same story with a different line: what a web AI would say, which steps this repo must walk, what the table printed, how a human declares. Includes rejects and papers that already have a table but still wait for `--declare`.

Fine print lives in `docs/examples/`. Each walkthrough starts with **How this sentence walks the repo**, then seven sections: Design / Data / Features / IC / Validation / Conclusion / How to read the report. This page is the comparison table, not a substitute for that story. Philosophy: [docs/philosophy.en.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/philosophy.en.md). First act (golden cross): [README.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/README.md).

## Trend

### MA golden cross (BTC · 2h)

Walkthrough: repo-root [README.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/README.md).

| Box | Content |
|---|---|
| Sentence | Go long when the close is above the 50-day and just crossed above the 200-day; exit on a break of the 50-day. |
| Class | Beta / trend exposure |
| Harness | `event_backtest` · `BTCUSDT` · 2h |
| Data | Binance trades → parquet; FeatureStore `ema_50_200_cross_*` |
| IC | None. Single-name 0/1 clock. |
| Close | Recent CAGR is negative. Against the written falsifier the sentence fails in the recent window. Human still `--declare`. |

| Window | CAGR | Calmar | Win rate | MaxDD | Sharpe(R) | Trades |
|---|---:|---:|---:|---:|---:|---:|
| Bear 2022 | +3.7% | 1.51 | 38.3% | −2.4% | 0.15 | 47 |
| Bull 2023–2024 | +3.0% | 0.97 | 33.3% | −3.1% | 0.15 | 33 |
| Recent range → bear | **−3.1%** | −0.68 | 24.3% | −4.5% | −0.36 | 37 |

## Funding

### Funding fade

Walkthrough: [docs/examples/20260910_funding_fade.en.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260910_funding_fade.en.md)

| Box | Content |
|---|---|
| Sentence | Fade a 50-print funding z-score at ±1.5; exit when z returns through 0. |
| Class | Crowding fade / mean-reversion |
| Harness | `event_backtest` · `BTCUSDT` · 2h |
| Data | Trades + `mlbot data download-funding-rate`; column `funding_rate_zscore_50` |
| IC | None. Single-name 0/1 clock. |
| Close | Recent CAGR −3.4%, MaxDD −6.9% deeper than both trend windows. Both falsifiers hit. Human still `--declare`. |

| Window | CAGR | Calmar | Win rate | MaxDD | Sharpe(R) | Trades |
|---|---:|---:|---:|---:|---:|---:|
| Bear 2022 | +2.7% | 0.93 | 47.7% | −2.9% | 0.08 | 128 |
| Bull 2023–2024 | +3.0% | 0.60 | 50.7% | −4.9% | 0.13 | 75 |
| Recent range → bear | **−3.4%** | −0.50 | 45.9% | **−6.9%** | −0.11 | 109 |

## Calendar

### Monday rebound

Walkthrough: [docs/examples/20260911_ashare_monday_rebound.en.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260911_ashare_monday_rebound.en.md)

| Box | Content |
|---|---|
| Sentence | After CSI 300 **Monday close is down**, go long and hold **four daily bars**. Earliest fill is Tuesday open. Not “buy Monday open, sell Friday close.” |
| Class | Calendar-alpha claim |
| Harness | `event_backtest` · `000300.SH` · daily |
| Data | `mlbot data download-ashare`; A-share calendar `bear_2021` / `bull_924` / `chop_recent` |
| IC | None. Calendar 0/1 permit. |
| Close | All three CAGRs are tiny positives; recent +0.06%, Calmar 0.14. Weak calendar alpha. Human still `--declare`. |

| Window | CAGR | Calmar | Win rate | MaxDD | Sharpe(R) | Trades |
|---|---:|---:|---:|---:|---:|---:|
| `bear_2021` | +0.61% | 1.16 | 52.9% | −0.53% | 0.12 | 34 |
| `bull_924` | +0.46% | 1.52 | 50.0% | −0.30% | 0.12 | 16 |
| `chop_recent` | +0.06% | 0.14 | 58.8% | −0.45% | 0.03 | 17 |

## Cross-symbol

### BTC leads AI alts

Walkthrough: [docs/examples/20260911_btc_lead_ai_alts.en.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260911_btc_lead_ai_alts.en.md)

| Box | Content |
|---|---|
| Sentence | Prior closed BTC 2h return ≥ +3% → long NEAR / FET / RENDER for six bars. |
| Class | Beta (alts vs BTC) |
| Harness | `event_backtest` · 2h |
| Data | Alt ticks from about 2023-01; `bear_2022` is **no sample** |
| IC | None. 0/1 permit. |
| Close | Recent CAGR −1.52%, MaxDD deeper than bull. A missing year cannot promote from the recent window. Human still `--declare`. |

| Window | CAGR | Calmar | Win rate | MaxDD | Sharpe(R) | Trades |
|---|---:|---:|---:|---:|---:|---:|
| `bear_2022` | no sample | — | — | — | — | 0 |
| `bull_2023_2024` | +2.81% | 2.29 | 50.0% | −1.22% | 0.26 | 46 |
| Recent | **−1.52%** | −0.54 | 30.8% | **−2.82%** | −0.24 | 26 |

## Fat-tail

### P99 big-order chase

Walkthrough: [docs/examples/20260911_p99_bb_break_chase.en.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260911_p99_bb_break_chase.en.md)

| Box | Content |
|---|---|
| Sentence | In-bar P99 print **and** `bb_position ≥ 1` → long; exit back inside the band or at 12 bars. |
| Class | Momentum / fat-tail right tail |
| Harness | `event_backtest` · `BTCUSDT` · 2h |
| Data | **Ticks**; missing months are NaN. Daily bars cannot stand in for P99. |
| IC | None. Two-flag 0/1 clock. |
| Close | Bull CAGR −0.25% hits the falsifier. Top-3 is classification only. Human still `--declare`. |

| Window | CAGR | Calmar | Win rate | MaxDD | Sharpe(R) | Trades |
|---|---:|---:|---:|---:|---:|---:|
| Bear 2022 | +0.24% | 0.32 | 45.5% | −0.74% | 0.09 | 22 |
| Bull 2023–2024 | **−0.25%** | −0.26 | 36.4% | −0.95% | −0.10 | 22 |
| Recent | +0.28% | 0.92 | 57.1% | −0.30% | 0.16 | 21 |

### Tenbagger cohort

Walkthrough: [docs/examples/20260911_tenbagger_smallcap.en.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260911_tenbagger_smallcap.en.md)

| Box | Content |
|---|---|
| Sentence | Entry-date float cap ≤ 10 billion yuan, hold 3 or 4 years: is 10× denser than large caps? |
| Class | Thin fat tail, mainly small-cap beta |
| Harness | `cohort_hold` (not `event_backtest`) |
| Data | Listed + delisted daily; cap = that day’s amount / turnover; `n` = name × entry quarter |
| IC | None. Density, not a daily score. |
| Close | 3y 10× rate **0.11% vs 0.11%**. Crash / 2018 relative CAGR ≤ 0. Declared reject. Cannot measure “who 10×s next.” |

| 3y hold | Small CAGR | Large CAGR | 10× rate (small vs large) |
|---|---:|---:|---|
| `crash_2015` | −4.26% | −4.15% | 0 vs 0 |
| `bear_2018` | +2.75% | +3.31% | 0.29% vs 0.24% |
| `covid_2020` | +7.75% | +5.96% | 0 vs 0.13% |
| `bear_2021` | +9.09% | −0.03% | 0.09% vs 0.12% |
| `bull_924` / `chop_recent` | no sample | no sample | — |

## Score the whole market every day

How it is measured: [docs/cs_panel.en.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/cs_panel.en.md)

### CS momentum + amount

Walkthrough: [docs/examples/20260911_ashare_cs_mom_amount.en.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260911_ashare_cs_mom_amount.en.md)

| Box | Content |
|---|---|
| Sentence | `score = 0.5 cs_z(mom_20) + 0.5 cs_z(amount_z_20)`, buy top 20%, keep beating universe EW. |
| Class | Beta continuation claim; measured sign is reversal |
| Harness | `cs_panel` |
| Data | Non-ST 5228 / aligned 5175; A-share three windows |
| IC | **Yes, flashlight only.** Daily-mean `score` IC −0.053 / −0.110 / −0.054. |
| Close | Loses to EW in every window. 924 EW +76.57%, top book +24.13%. Declared reject. |

| Book | `bear_2021` | `bull_924` | `chop_recent` |
|---|---:|---:|---:|
| Top 20% CAGR | −8.65% | +24.13% | +13.84% |
| EW CAGR | +4.70% | **+76.57%** | +22.75% |
| vs EW | **−13.35pp** | **−52.44pp** | **−8.91pp** |

### Hot sectors

Walkthrough: [docs/examples/20260911_ashare_cs_sector_cost.en.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260911_ashare_cs_sector_cost.en.md)

| Box | Content |
|---|---|
| Sentence | Rank sectors first, charge 10bp each side. Versus **same-universe, same-cost** EW, hot sectors should be better. |
| Class | Sector rotation / beta, not sector-picking alpha |
| Harness | `cs_sector.py` |
| Data | 20-bucket coarse snapshot, 3633 names; daily turnover ≈ 0.38 |
| IC | No separate sector IC. The name-level pair is already negative in all three windows. |
| Close | Bear +2.99pp vs EW; bull **−40.54pp**; chop **−24.69pp**. Gross books also lose in bull / chop. Declared reject. |

| Book | `bear_2021` | `bull_924` | `chop_recent` |
|---|---:|---:|---:|
| Same-universe EW (10bp) | +6.42% | **+74.59%** | **+30.48%** |
| Hot-sector top 20% (10bp) | +9.41% | +34.05% | +5.79% |

## Beta

### US SPY / QQQ buy-and-hold

Walkthrough: [docs/examples/20260914_eq_us_spy_qqq_beta.en.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260914_eq_us_spy_qqq_beta.en.md)

| Box | Content |
|---|---|
| Sentence | Buying the ETF misses the US bull; you must pick stocks or wait for a washout. |
| Class | Beta (equity exposure), not stock-selection alpha |
| Harness | `eq_us_daily` (not 2h `event_backtest`) |
| Data | Unlevered SPY / QQQ daily; lock 2026-08-17; split-adjusted close, not dividend total return |
| IC | None. Single-name 0/1 clocks. PIT stock-picking not re-run. |
| Close | All four washout / MA clocks compound slower than same-window buy-and-hold. Declared reject. |

| Symbol | Buy-and-hold CAGR | MaxDD | Best same-window timing CAGR |
|---|---:|---:|---|
| SPY | **13.7%** | −34.1% | MA200 7.8% |
| QQQ | **20.4%** | −35.6% | MA200 16.0% (shallower hole, slower compound — insurance) |

### AI chip-sales QoQ vs BTC regime

Walkthrough: [docs/examples/20260914_ai_chip_spend_btc_regime.en.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260914_ai_chip_spend_btc_regime.en.md)

| Box | Content |
|---|---|
| Sentence | When compute-spend QoQ is high, BTC closed bars sit below MA200 more often (not next-few-days return, not segment labels). |
| Class | Beta / regime coexistence |
| Harness | `phase1_scan_only` |
| Data | Epoch quarterly chip bills (CC BY) + BTC daily |
| IC | No return IC. The flashlight *is* the share table. |
| Close | High-QoQ non-bull 13.4% vs low 50.5% (−37pp). Opposite of the paper. Declared reject. |

| Scope | High non-bull | Low non-bull | Diff | n high / n low |
|---|---:|---:|---:|---|
| Pooled | 13.4% | 50.5% | **−37.1pp** | 546 / 610 |
| Bull | 14.9% | 39.7% | −24.8pp | 396 / 184 |
| Recent | 15.6% | 55.1% | −39.5pp | 90 / 425 |

### Long BTC after an AI financing print

Walkthrough: [docs/examples/20260914_ai_financing_btc.en.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260914_ai_financing_btc.en.md)

| Box | Content |
|---|---|
| Sentence | After a public mega AI financing date D closes, long BTC the next day for five UTC days. |
| Class | Beta (risk-on spillover) |
| How the table is made | Daily path after the print has been looked at; two-hour in-and-out not run |
| Data | Locked public mega rounds, 21 prints; daily bars aligned to the print date |
| Daily number | “Is this day inside a window after a print?” vs next-day return **−0.021** (p = 0.40). Does not close. |
| Close | No daily “it should rise.” The in-and-out report card is still empty. **Do not close the sentence yet.** |

| Window | Events | 5d mean | Median | Any-5d baseline |
|---|---:|---:|---:|---:|
| Bear | 7 | +1.12% | +0.97% | +0.01% |
| Bull | 9 | +0.04% | −0.24% | +1.24% |
| Recent | 7 | +0.25% | −0.76% | −0.15% |

## Fine print

- How to read the seven sections: [docs/examples/20260914_eq_us_spy_qqq_beta.en.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260914_eq_us_spy_qqq_beta.en.md) (the most complete walkthrough; use it as the template)
- How to choose layers: [docs/framework.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/framework.md) section 4
- Hypothesis template: [docs/hypothesis_template.en.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/hypothesis_template.en.md)
