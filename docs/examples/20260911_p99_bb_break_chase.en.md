# P99 large trade + Bollinger upper chase

Paper: [config/experiments/20260911_p99_bb_break_chase/](../../config/experiments/20260911_p99_bb_break_chase/)  
Class first as **momentum / fat-tail right tail**, not point-selection alpha. Dropping Top-3 is classification only — not a solo kill.  
Harness: **event_backtest** (BTCUSDT · 2h). The table is on the paper; the human still `--declare`. Do not type `verdict:`.

> When a 2h bar prints a P99-sized trade *and* the close breaks the Bollinger upper band, chase; flatten when price is back inside the band or after 12 bars.

The court asks whether “aggressive take + crowded momentum” survives all three windows. Months without ticks must stay NaN. A daily high/low cannot stand in for “was there one unusually large print inside the bar.”

中文：[20260911_p99_bb_break_chase_CN.md](20260911_p99_bb_break_chase_CN.md)

---

## How this sentence walks the repo

You ask a web AI: “A huge print and a Bollinger upper-band break — chase it?”

It talks momentum and “smart money.” The band is often a daily high/low, or a line computed in the chat. There is no “how large was the single print inside the bar,” and no “did the year that should pay the right tail print a negative CAGR?”

You say the same sentence here. Class first: **momentum / fat-tail right tail**. Dropping Top-3 classifies; it cannot kill this sleeve alone — but only after a right tail is actually measured. Grain is locked: P99 comes from ticks; missing months are NaN; daily bars cannot stand in. After “measure this,” download trades, build the layer, run the 2h court. Exit is written: back inside the band or 12 bars.

The table: bull CAGR −0.25%, exactly the year the right tail should pay. A small recent plus and a 57% win rate are inside-band noise, not a fat tail. Do not cut the right tail to raise win rate, and do not prove the sentence with win rate. Dropping either flag is a new paper.

The seven sections below unpack ticks, closed-bar, and “low trade count is the phenomenon.”

---

## Design

```text
Human sentence (P99 print and upper-band break → chase)
  → template / validate / lineage
  → download ticks, convert, build FeatureStore — only after “measure this”
  → 2h court: both flags long; exit inside band or at 12 bars
  → human --declare
```

P99 must come from tick / aggTrades: max notional inside the bar versus a rolling quantile.

| Box | This sentence |
|---|---|
| Mechanism | `bar_max_notional_ge_p99 ≥ 1` **and** `bb_position ≥ 1` to go long. |
| Regimes | Crowded momentum should continue; the right tail should pay. Dropping the largest trades classifies, does not kill alone. |
| Contract | Exit when `bb_position < 1` or after 12×2h. No adds. Kill switch off. Closed-bar. |
| Falsifier | Any window CAGR < 0, or recent MaxDD deeper than trend. |
| Landing | Robot: `strategies/p99_bb_chase`. Same sentence for hands. |

| Template box | How this sentence is written |
|---|---|
| Sociology | A large print is aggressive take; an upper-band break is crowded momentum. Late arrivals pay. |
| Math | Bar-max notional vs rolling P99; Bollinger position ≥ 1. Death is back inside or time. |
| Stats | Momentum / fat-tail right tail. Do not claim on win rate alone. |
| Ruler | Any window CAGR < 0, or recent drawdown deeper than trend. |
| Data range | `BTCUSDT` · 2h · crypto three windows · layer `features_p99_bb_chase_120T`. |

Clock (signal known at close t; position starts on **t+1**):

| Moment | What happens |
|---|---|
| Permit | Previous bar has both the P99 flag and `bb_position ≥ 1` |
| Side | Fixed long |
| Exit | Previous `bb_position < 1`, or 12 bars held |
| This machine | Almost every exit is back inside the band, not time |

Headline five KPIs only.

---

## Data

```bash
mlbot research validate 20260911_p99_bb_break_chase
mlbot data download --symbols BTCUSDT \
  --start-year 2022 --start-month 1 --end-year 2026 --end-month 5
mlbot data convert --symbols BTCUSDT
```

| Item | This machine |
|---|---|
| Symbol | `BTCUSDT` |
| Bar | Two-hour (`120T`) |
| Grain | **tick / aggTrades** → parquet. Daily bars cannot build this P99 |
| Path | `data/parquet_data` |
| Layer | `features_p99_bb_chase_120T` |
| Missing months | The column is **NaN**; that month does not permit. Do not invent numbers |
| Calendar | [`config/market_segment.yaml`](../../config/market_segment.yaml) |
| Kill switch | Off |

| Window | Span | Role |
|---|---|---|
| `bear_2022` | 2022-01-01 → 2023-11-01 | Chase must not blow the book. |
| `bull_2023_2024` | 2023-06-01 → 2025-01-01 | The right tail should pay. |
| `recent_range_to_bear` | 2025-01-01 → 2026-05-31 | Recent. Cannot promote alone. |

Using daily amount as a stand-in for P99 is a different sentence.

---

## Features

| Column | Construction | Closed-bar use |
|---|---|---|
| `bar_max_notional_ge_p99` | Max single-trade notional in the 2h bar ≥ rolling P99 | Flag known at close; next bar may open |
| `bb_position` | Close versus band; ≥ 1 means outside the upper band | Entry wants ≥ 1; exit reads previous < 1 |
| `atr_f` | Volatility for size | Previous bar only |

```bash
PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/experiments/20260911_p99_bb_break_chase/strategies/p99_bb_chase \
  --symbols BTCUSDT --timeframe 120T \
  --root feature_store --layer features_p99_bb_chase_120T \
  --data-path data/parquet_data \
  --start-date 2022-01-01 --end-date 2026-05-31 --no-reuse
```

Grid: `p99_bb_chase_grid.yaml`.

---

## IC

**There is no cross-sectional IC table, and there should not be one.**

This is a two-flag 0/1 chase clock. Fat-tail claims also must not close on IC: averaging score-versus-next-return dilutes the few right-tail paths. The close is three-window five KPIs; Top-3 is classification.

---

## Validation

```bash
PYTHONPATH=src python -m scripts.event_backtest --variant-grid \
  config/experiments/20260911_p99_bb_break_chase/p99_bb_chase_grid.yaml
mlbot research close 20260911_p99_bb_break_chase
```

### Three windows (BTCUSDT · 2h · kill switch off)

| Window | CAGR | Calmar | Win rate | MaxDD | Sharpe(R) | Trades |
|---|---:|---:|---:|---:|---:|---:|
| `bear_2022` | +0.24% | 0.32 | 45.5% | −0.74% | 0.09 | 22 |
| `bull_2023_2024` | **−0.25%** | −0.26 | 36.4% | −0.95% | −0.10 | 22 |
| `recent_range_to_bear` | +0.28% | 0.92 | 57.1% | −0.30% | 0.16 | 21 |

| Written falsifier | Hit? |
|---|---|
| Any window CAGR < 0 | Yes. Bull −0.25%. |
| Recent MaxDD deeper than trend | No. Recent −0.30% is shallower. The first line already kills the sentence. |

Almost every exit is `structural_exit_bb_position_lt1`. The bull year is where the right tail should pay; CAGR is negative. Twenty-odd trades per window is this sentence’s true density, not a license to lower P99.

---

## Conclusion

Class: **momentum / fat-tail claim, measured near useless.**

- Bull CAGR negative hits the falsifier. The year that should light up does not.
- Bear and recent CAGRs sit near zero.
- Top-3 classification cannot revive a sleeve that never showed a right tail.
- Dropping either flag is a new sentence.

Declare with `--declare`. Do not hand-trade “this print is different.”

| Path | What it is |
|---|---|
| `results/p99_bb_chase/experiments/20260911_p99_bb_break_chase/p99_bb/bear_2022` | Bear report |
| `results/p99_bb_chase/experiments/20260911_p99_bb_break_chase/p99_bb/bull_2023_2024` | Bull report |
| `results/p99_bb_chase/experiments/20260911_p99_bb_break_chase/p99_bb/recent_range_to_bear` | Recent report |
| [DECISION.md](../../config/experiments/20260911_p99_bb_break_chase/DECISION.md) | Paper |

---

## How to read the report

1. **Read the bull window first.** Fat-tail / momentum should pay there. Negative CAGR already kills a small recent plus.
2. **Headline is CAGR, not win rate.** Recent 57.1% win rate is small inside-band noise, not a right tail.
3. **Low trade count is the phenomenon.** Lowering P99 to get more trades changes the sentence.
4. **Almost all exits are back inside the band.** Momentum did **not** continue.
5. **Do not headline ΣR.** A fat-tail claim is easy to fool with one large R. Here even CAGR is negative in the bull.
6. **Missing-tick months must be NaN.** A sudden jump in trade count is a grain bug.
7. **Top-3 is classification only.**
8. **Do not merge with funding fade.** Fade shorts crowding; this sentence chases it.

The paper is [DECISION.md](../../config/experiments/20260911_p99_bb_break_chase/DECISION.md).
