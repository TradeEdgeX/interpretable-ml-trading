# BTC surge → AI alts follow

Paper: [config/experiments/20260911_btc_lead_ai_alts/](../../config/experiments/20260911_btc_lead_ai_alts/)  
Class first as **beta**: the alts follow bitcoin’s risk-on, not “being good at picking which alt bar.” The table is on the paper; whether the sentence holds is still for you to write against the ruler.

> After BTC rips, AI narrative alts follow. Long the alts when the previous *closed* BTC 2h return is ≥ +3%; hold six 2h bars.

This sentence asks whether “BTC prices risk-on first, theme money arrives later” survives all three windows. A window where the alt was not listed yet cannot pass on the later two windows alone.

Not [long BTC after an AI financing print](20260914_ai_financing_btc.en.md). Not [funding fade](20260910_funding_fade.en.md).

中文：[20260911_btc_lead_ai_alts_CN.md](20260911_btc_lead_ai_alts_CN.md)

---

## How this sentence walks the repo

You ask a web AI: “After BTC rips, AI alts follow — should I chase?”

It talks narrative and which coin has more beta. It does not ask whether those alts were listed in 2022, or whether a green bull can promote a red recent window. Change the model, change the ticker.

You say the same sentence here. Class first: **beta**, the alts follow bitcoin’s heat, not “being good at picking which bar.” Entry may only look at the **previous already-closed** bitcoin two-hour return, already lined up on the alt’s own bar. Do not use the still-open same bar, and do not swap in “how much bitcoin rose over the last ninety bars.” A window where the alt was not listed yet is “no sample,” not a borrowed coin. After “measure this,” download trades, pre-align the lead number, and print the two-hour in-and-out table.

Local ticks start around 2023-01, so the 2022 bear is empty — the range rule working, not a bug. Bull +2.81%, recent −1.52% with a deeper hole. A missing year cannot pass on the later two windows. Switching to daily bars from 2023 is a **new range**. Do not hand-trade “a different AI coin this time.”

Not “long BTC after a financing print,” not funding fade. Three sentences share “AI” or “crowding”; the paying side and the measured object differ. The seven sections below unpack cross-symbol alignment and “no sample.”

---

## Design

```text
Human sentence (BTC rips, AI alts follow)
  → write down: who pays, what is measured, which dates, how you lose
  → check: this sentence was not already closed
  → download trades, pre-align “how much bitcoin just rose” — only after “measure this”
  → on the alt’s own two-hour bar: prior bitcoin return ≥ +3%, hold 6 bars
  → human writes whether it holds
```

How much bitcoin just rose must already be lined up on the alt’s bar, from the previous closed print. Do not drop bitcoin’s return in from another table while printing the book.

| Box | This sentence |
|---|---|
| Mechanism | Previous closed BTC 2h return ≥ +3% → long AI narrative alts. |
| Regimes | Risk-on should transmit. A window with no listing tape is “no sample,” not a borrowed coin. |
| Contract | Time-exit 6×2h; no adds; kill switch off; closed-bar. |
| Falsifier | Any window CAGR < 0, or recent MaxDD deeper than the trend window. |
| Landing | Robot: `strategies/btc_lead_alts`. Same sentence for hands. |

| Template box | How this sentence is written |
|---|---|
| Sociology | BTC prices risk appetite first. Theme buyers pay the people already in. |
| Math | BTC **prior** closed 2h return on the alt bar, threshold +3%, hold 6 bars. |
| Stats | Alt-vs-BTC beta, not “which alt bar to chase.” |
| Ruler | Any window CAGR < 0, or recent drawdown deeper than trend. |
| Data range | `NEARUSDT` / `FETUSDT` / `RENDERUSDT` · 2h · crypto three windows. `TAOUSDT` is not primary if it lacks a bear listing. |

Clock (signal known at close t; position starts on **t+1**):

| Moment | What happens |
|---|---|
| Permit | Previous alt bar’s `btc_prior_bar_return` ≥ +0.03 |
| Side | Fixed long |
| Exit | 6 two-hour bars (12 hours) |
| Flat bars | Earn 0 |

Headline five KPIs only.

---

## Data

```bash
mlbot research validate 20260911_btc_lead_ai_alts
mlbot data download --symbols BTCUSDT,NEARUSDT,FETUSDT,RENDERUSDT \
  --start-year 2022 --start-month 1 --end-year 2026 --end-month 6
mlbot data convert --symbols BTCUSDT,NEARUSDT,FETUSDT,RENDERUSDT
```

| Item | This machine |
|---|---|
| Tested names | `NEARUSDT` / `FETUSDT` / `RENDERUSDT` |
| Lead name | `BTCUSDT` (builds `btc_prior_bar_return` only; not in the alt book) |
| Bar | Two-hour (`120T`) |
| Tape | `data/parquet_data` |
| Local listing | Alt ticks start around **2023-01**, so `bear_2022` is **no sample** |
| Layer | `features_btc_lead_alts_120T` |
| Calendar | [`config/market_segment.yaml`](../../config/market_segment.yaml) |
| Kill switch | Off |

| Window | Span | Coverage here |
|---|---|---|
| `bear_2022` | 2022-01-01 → 2023-11-01 | **No sample** |
| `bull_2023_2024` | 2023-06-01 → 2025-01-01 | Has sample. |
| `recent_range_to_bear` | 2025-01-01 → 2026-05-31 | Recent. Cannot pass on its own. |

---

## Features

| Column | Construction | Closed-bar use |
|---|---|---|
| `btc_prior_bar_return` | BTC’s previous closed 2h return aligned onto each alt bar | `ret[t] ≥ +3%` is first tradable on the next alt bar |
| `atr_f` | Volatility for size | Previous bar only |
| Alt `close` | The tested name’s 2h close | Book uses realized bar returns |

Do not use `btc_roc90` (longer BTC momentum — a different sentence) or the *current* BTC bar (not closed yet).

```bash
PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/experiments/20260911_btc_lead_ai_alts/strategies/btc_lead_alts \
  --symbols NEARUSDT,FETUSDT,RENDERUSDT \
  --timeframe 120T --root feature_store --layer features_btc_lead_alts_120T \
  --data-path data/parquet_data --start-date 2022-01-01 --end-date 2026-06-01 \
  --allow-partial --no-reuse
```

`--allow-partial` is for missing listing months. Write “no sample.” Do not invent bear-window numbers.

---

## IC

**There is no cross-sectional IC table, and there should not be one.**

This is a 0/1 permit on “did BTC just rip.” A continuous `btc_prior_bar_return` versus next-day alt return is a new folder. A missing bear tape cannot be patched with bull or recent IC.

---

## Validation

```bash
PYTHONPATH=src python -m scripts.event_backtest --variant-grid \
  config/experiments/20260911_btc_lead_ai_alts/btc_lead_alts_grid.yaml
mlbot research close 20260911_btc_lead_ai_alts
```

### Three windows (NEAR / FET / RENDER · 2h · kill switch off)

| Window | CAGR | Calmar | Win rate | MaxDD | Sharpe(R) | Trades |
|---|---:|---:|---:|---:|---:|---:|
| `bear_2022` | no sample | — | — | — | — | 0 |
| `bull_2023_2024` | +2.81% | 2.29 | 50.0% | −1.22% | 0.26 | 46 |
| `recent_range_to_bear` | **−1.52%** | −0.54 | 30.8% | **−2.82%** | −0.24 | 26 |

| Written falsifier | Hit? |
|---|---|
| Any window CAGR < 0 | Yes. Recent −1.52%. |
| Recent MaxDD deeper than trend | Yes. −2.82% vs −1.22%. |
| Can a no-sample bear promote? | No. |

Bull Calmar looks pretty because drawdown is shallow and trade count is modest. It cannot revive the recent window or invent a 2022 tape.

---

## Conclusion

Class: **beta**. Alts following a BTC rip is risk-on transmission, not independent AI-narrative alpha.

- Bear no-sample is the range rule working.
- Bull +2.81%, Sharpe 0.26, is a small pay in a risk-on year, not a point-entry to ship.
- Recent CAGR negative, win rate 30.8%, deeper drawdown: late theme money stopped paying.
- Dropping Top-3 trades is classification only.

A human writes whether it holds, against the ruler. Do not hand-trade “a different AI coin this time.”

| Path | What it is |
|---|---|
| `results/btc_lead_alts/experiments/20260911_btc_lead_ai_alts/btc_lead_3pct/bull_2023_2024` | Bull report |
| `results/btc_lead_alts/experiments/20260911_btc_lead_ai_alts/btc_lead_3pct/recent_range_to_bear` | Recent report |
| [DECISION.md](../../config/experiments/20260911_btc_lead_ai_alts/DECISION.md) | Paper |

---

## How to read the report

1. **Sample first, then CAGR.** “No sample” on `bear_2022` is the first conclusion, not an empty cell.
2. **Headline is CAGR, not ΣR.** Calmar 2.29 comes from a shallow hole, not a high CAGR.
3. **Win-rate collapse is the mechanism.** 50.0% → 30.8%: after BTC rips, alts more often do not follow.
4. **26 trades do not authorize a +1% threshold.** That is a new sentence.
5. **Six-bar time exit is the contract.**
6. **`btc_prior_bar_return` must be the prior bar.** A same-bar BTC return makes CAGR look fake.
7. **Switching to daily bars from 2023 is a new range.** Rewrite the paper.
8. **Do not merge with the financing-day sentence.** Different X.

The paper is [DECISION.md](../../config/experiments/20260911_btc_lead_ai_alts/DECISION.md).
