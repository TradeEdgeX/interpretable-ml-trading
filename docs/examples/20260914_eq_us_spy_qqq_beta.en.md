# Does buying an ETF miss the US bull market?

Paper: [config/experiments/20260914_eq_us_spy_qqq_beta/](../../config/experiments/20260914_eq_us_spy_qqq_beta/)  
Class first as **beta** (S&P 500 / Nasdaq-100 equity exposure), not stock-selection alpha.  
Harness: **eq_us_daily**, not 2h `event_backtest`. Human already `--declare reject`.

> Buying SPY or QQQ “misses” the US bull; to catch the burst you must pick stocks, or wait for an index washout.

The court asks whether washout timing (or stock-picking) beats **same-window** unlevered buy-and-hold. If it does, the sentence lives. If it does not, the sentence dies. Buy-and-hold itself is the beta benchmark, not an alpha to promote.

中文：[20260914_eq_us_spy_qqq_beta_CN.md](20260914_eq_us_spy_qqq_beta_CN.md)

---

## How this sentence walks the repo

You ask a web AI: “Buying an ETF misses the US bull. To catch it you must pick stocks, or wait for a washout.”

It usually tells you about NVDA, about who bought March 2020, about how dumb it is to just hold the index. There is no shared ruler and no “who loses if this is wrong.” Change the model, change the hero.

You say the same sentence here, plus “measure this.” The AI may not answer from memory. This repo is a **hypothesis validator**: it does not maximize profit; it tries to learn quickly whether the sentence lives, dies, or lives only in some regimes. The path is locked —

| Step | Who | Does | Does not |
|---|---|---|---|
| 1 | Human sentence; AI helps fill the template | Who pays, what is measured, which tape, which KPI kills it | Rewrite it as “add RSI” |
| 2 | Program | `validate` the template; `index` for a prior close | Download or backtest |
| 3 | Human says “measure this” | Then download SPY / QQQ daily and run `eq_us_spy_qqq.py` | `mlbot research run` (that path only dispatches 2h event backtests) |
| 4 | Program prints the table | Same-window buy-and-hold vs four washout / MA clocks; five KPIs only | Type `verdict:` |
| 5 | Human reads | `--declare` against the written falsifier | Hand-trade a rejected sentence at night |

Class first: unlevered ETF buy-and-hold is **equity beta** on this tape, not an alpha to ship. The popular sentence lives only if washout timing or stock-picking beats **same-window** buy-and-hold on CAGR and does not deepen MaxDD. There is no universe here, so stock-picking was not measured. The half that was — four clocks — all compound slower. A human `--declare reject`ed. The robot does not write washout into YAML; a human should not wait for RSI to “catch the bull.”

The seven sections below unpack that walk. The golden-cross story is in the root [README.md](../../README.md). The loop is [hypothesis.en.md](../hypothesis.en.md); who writes which field is [rd_playbook.md](../agent/rd_playbook.md).

---

## Design

```text
Human sentence (ETF is too slow / must pick or wait)
  → template: sociology / math / stats / ruler / four US windows / five boxes
  → validate + lineage (no prior close of this sentence in the public tree)
  → daily court eq_us_daily: buy-and-hold vs four index clocks
  → human --declare
```

Do not use `mlbot research run` (that path only dispatches 2h `event_backtest`). There is no FeatureStore layer and no cross-sectional universe.

| Box | This sentence |
|---|---|
| Mechanism | Benchmark: 100% SPY or QQQ on every closed day. Controls: wait for RSI / first −20% / MA200. S&P 500 PIT stock-picking was not re-run in this extract. |
| Regimes | Long bull years should pay most of the compound to buy-and-hold. Crash years make a washout clock look “right”. Slow bears chop a fixed hold clock halfway down. |
| Contract | Daily closed bar. No adds. Timing friction 0. Buy-and-hold is always in. Kill switch off. |
| Falsifier | The sentence lives only if timing or stock-picking **beats same-window buy-and-hold on CAGR and does not deepen MaxDD**. Any window (or the full tape) where timing CAGR is lower, or stock-picking MaxDD is deeper than same-window SPY, hits the sentence. The recent window cannot promote alone. |
| Landing | Robot: `scripts/research/eq_us_spy_qqq.py`. Same sentence for hands. Do not write washout / stock-picking into YAML as alpha. |

Four clocks (signal known at close t; position starts on **t+1**; cash days earn 0):

| Mode | Enter | Exit |
|---|---|---|
| `rsi_hold_40` | Wilder RSI(14) ≤ 30 | Hold 40 trading days |
| `rsi_to_252_high` | Same | Hold until the 252-day high known at entry |
| `dd20_hold_60` | **First** close ≤ −20% from the running peak | Hold 60 trading days |
| `ma200` | Close above the 200-day average | Flatten the next day after a close below |

Headline five KPIs only: CAGR / Calmar / win rate / MaxDD / Sharpe. Do not sum R.

---

## Data

```bash
mlbot research validate 20260914_eq_us_spy_qqq_beta
mlbot data download-us --symbols SPY,QQQ --start-date 2013-01-01
```

| Item | This machine |
|---|---|
| Symbols | Unlevered `SPY`, `QQQ` |
| Bar | Daily |
| Source | Nasdaq public historical API (Yahoo / Stooq were blocked here) |
| Path | `data/eq/us/daily/{SPY,QQQ}.parquet` |
| Tape | 2016-09-12 → 2026-09-11 (2,514 bars) |
| Court lock | **2026-08-17** (2,496 bars) |
| Price | Split-adjusted close, **not** dividend total return |
| Calendar | [`config/market_segment_us.yaml`](../../config/market_segment_us.yaml). Do not reuse crypto `bear_2022` or A-share `bull_924`. |

The paper’s 2014-08 / 2015-08 full-sample windows collapse onto **one** tape because bars start in 2016-09 (n = 2,496). Timing is compared only to buy-and-hold in the same table.

Windows:

| Id | Span | Role |
|---|---|---|
| `us_covid_2020` | 2020-02-19 → 2020-03-23 | Crash coverage. **24** bars here — CAGR cannot close the case. |
| `us_bear_2022` | 2022-01-03 → 2022-10-14 | Slow bear: a fixed clock gets chopped mid-slide. |
| `us_bull_2023_2024` | 2023-01-01 → 2025-01-01 | Main compounding window for buy-and-hold. |
| `us_recent` | 2025-01-01 → 2026-08-17 | Recent. Cannot promote alone. |

S&P 500 PIT stock-picking: **not re-run** (no universe in this extract). `result.json` records `pit_stock_picking: not_run_no_universe`.

---

## Features

This court **does not read FeatureStore**. There is no `features.yaml`. Do not add a local `compute_*` fallback in a backtest hot path — the only series is daily close, and the clocks live in [`src/research/eq_us_spy_qqq.py`](../../src/research/eq_us_spy_qqq.py).

| Series | Construction | Closed-bar use |
|---|---|---|
| `close` | Nasdaq split-adjusted close | Book uses realized daily returns; position uses the **previous** signal |
| RSI(14) | Wilder (ewm of gains/losses, α = 1/14) | `rsi[t] ≤ 30` is first tradable on the next bar |
| 252-day high | `rolling(252).max()` | Target is known on the trigger bar; NaN (warm-up) does not enter |
| Peak drawdown | `close / cummax − 1` | Only the **first** breach of −20% starts a clock |
| MA200 | `rolling(200).mean()` | `(close > ma).shift(1)`: a close above today is held tomorrow |

Days that have not warmed RSI / MA200 / 252-day high stay flat. No look-ahead fill.

---

## IC

**There is no IC table, and there should not be one.**

Daily Spearman IC (score vs future return) is a flashlight for **cross-sectional ranks** — see the A-share momentum + amount example. This sentence is a single-name 0/1 clock: no universe, no score, no question of “does a more oversold day predict tomorrow.”

A flashlight cannot change the sentence and cannot close the case. The close uses same-window five KPIs only: did timing CAGR beat buy-and-hold, and did MaxDD get worse.

If someone wants RSI as a continuous score and an overnight IC, that is a different sentence and a new folder.

---

## Validation

```bash
PYTHONPATH=src python scripts/research/eq_us_spy_qqq.py
mlbot research close 20260914_eq_us_spy_qqq_beta --declare reject
```

Ruler, written first: timing (or stock-picking) full-sample CAGR **above** same-window buy-and-hold, **and** MaxDD no deeper. Kill switch off. Five KPIs only.

### Buy-and-hold (lock 2026-08-17 · kill switch off)

| Symbol | CAGR | Calmar | WR (day) | MaxDD | Sharpe |
|---|---:|---:|---:|---:|---:|
| SPY | **13.7%** | 0.40 | 55.2% | **−34.1%** | 0.80 |
| QQQ | **20.4%** | 0.57 | 56.3% | **−35.6%** | 0.93 |

### Same-window timing · QQQ

| Mode | CAGR | Calmar | WR (day) | MaxDD | Sharpe | In market |
|---|---:|---:|---:|---:|---:|---:|
| Buy-and-hold | **20.4%** | 0.57 | 56.3% | −35.6% | 0.93 | 100% |
| RSI≤30, hold 40d | 10.6% | 0.47 | 57.2% | −22.4% | 0.76 | 21% |
| RSI≤30, to 252d high | 10.7% | 0.36 | 55.0% | −29.9% | 0.67 | 35% |
| First −20% from peak, hold 60d | 8.4% | 0.27 | 55.6% | −31.4% | 0.66 | 17% |
| Above MA200 / flat below | 16.0% | 0.73 | 57.3% | −21.9% | 0.99 | 75% |

### Same-window timing · SPY

| Mode | CAGR | Calmar | WR (day) | MaxDD | Sharpe | In market |
|---|---:|---:|---:|---:|---:|---:|
| Buy-and-hold | **13.7%** | 0.40 | 55.2% | −34.1% | 0.80 | 100% |
| RSI≤30, hold 40d | 5.0% | 0.16 | 54.7% | −30.6% | 0.44 | 23% |
| RSI≤30, to 252d high | 5.8% | 0.20 | 54.1% | −28.7% | 0.44 | 42% |
| First −20% from peak, hold 60d | 5.5% | 0.32 | 52.5% | −17.2% | 0.57 | 13% |
| Above MA200 / flat below | 7.8% | 0.38 | 56.3% | −20.8% | 0.71 | 74% |

All four washout / MA clocks compound **slower** than same-window buy-and-hold. The falsifier hits.

### Buy-and-hold by window (n = trading days)

| Window | SPY CAGR | SPY MaxDD | QQQ CAGR | QQQ MaxDD | n |
|---|---:|---:|---:|---:|---:|
| `us_covid_2020` | −99.0% | −34.1% | −97.3% | −28.6% | 24 |
| `us_bear_2022` | −31.0% | −25.4% | −42.5% | −35.1% | 198 |
| `us_bull_2023_2024` | +24.2% | −10.3% | +39.3% | −13.6% | 502 |
| `us_recent` | +19.0% | −19.0% | +25.0% | −22.9% | 406 |

`us_covid_2020` is 24 bars. A two-week crash will print a nonsense CAGR. Coverage only.

---

## Conclusion

Class: **beta**. Unlevered SPY / QQQ buy-and-hold is the equity-exposure benchmark on this tape, not an alpha to ship.

- Washout clocks (RSI / −20%) all lose to same-window buy-and-hold on CAGR → “wait for a washout to catch the bull” is hit.
- MA200 cuts QQQ MaxDD from −35.6% to −21.9% and CAGR from 20.4% to 16.0%: **insurance**, not alpha.
- Stock-picking was not re-run. A missing universe cannot keep “you must pick stocks” alive.
- Dropping Top-3 names (NVDA and friends) is classification only; it does not kill this sleeve by itself.

Closed with `--declare reject`. The robot does not write washout / stock-picking into YAML as alpha; a human should not hand-trade “wait for RSI to catch the bull.” Do not type `verdict:`.

Artifacts:

- `results/eq_us_spy_qqq/experiments/20260914_eq_us_spy_qqq_beta/`
- Copy: [config/experiments/20260914_eq_us_spy_qqq_beta/result.json](../../config/experiments/20260914_eq_us_spy_qqq_beta/result.json)

Lineage:

```bash
mlbot research index --trusted --query spy-qqq
```

---

## How to read the report

Read `result.json` and the tables above with these rules. Do not invent a second ruler.

1. **Same window first, then CAGR.** Each timing row is compared only to `buy_hold` for the **same symbol, same `window`, same lock date**. Do not score QQQ timing against SPY buy-and-hold, or the recent window against the full tape.
2. **Headline is CAGR, not ΣR.** `pnl_r` / total R is not one of the five KPIs. Calmar = CAGR / |MaxDD|. Sharpe is from daily returns.
3. **Win rate is the fraction of in-market days that rose**, not trade-level win rate. An RSI sleeve that is in only ~21% of days can print a slightly higher win rate and still miss the bull — most up days were spent in cash.
4. **Time-in-market is the mechanism, not trivia.** Washout CAGRs are lower mainly because the sleeve is flat through most of the bull. That is how the original sentence pays: it gives normal up days to someone else.
5. **A shallower MaxDD does not revive the sentence.** The falsifier is “higher CAGR **and** MaxDD no deeper.” MA200 is shallower and slower: insurance, not a pass.
6. **These are not published SPY total-return figures.** The tape is split-adjusted close, no dividend reinvestment. The relative comparison still holds because both sides use the same price. Do not paste 13.7% / 20.4% onto a total-return chart.
7. **Do not story-tell the −99% `us_covid_2020` CAGR.** n = 24. The crash window only shows the calendar covers that episode.
8. **`pit_stock_picking: not_run_no_universe` is a gap, not a pass.** The “must pick stocks” half of the sentence was not measured. You may not claim it won or lost. The close covers the washout-timing half only.
9. **Windowed buy-and-hold is classification, not a second verdict.** `us_bull_2023_2024` QQQ +39.3% shows where the compound sat; `us_bear_2022` −42.5% shows that beta draws down. Neither turns a washout clock into alpha.

The paper is [DECISION.md](../../config/experiments/20260914_eq_us_spy_qqq_beta/DECISION.md).
