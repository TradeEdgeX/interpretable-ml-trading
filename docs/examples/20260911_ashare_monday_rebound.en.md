# A-share Monday down → next four sessions up

Paper: [config/experiments/20260911_ashare_monday_rebound/](../../config/experiments/20260911_ashare_monday_rebound/)  
Class first as a **calendar-alpha claim**, not index timing and not stock-picking.  
Harness: **event_backtest** (CSI 300 · daily). The table is on the paper; the human still `--declare`. Do not type `verdict:`.

> After CSI 300 Monday close is down, go long and hold four trading days. Measure on the A-share calendar.

The court asks whether “weekend news is priced on Monday, the next four sessions clear the panic” survives the three A-share windows. Permission is known at **Monday close**; the earliest fill is Tuesday open. This is not “buy Monday open, sell Friday close.”

中文：[20260911_ashare_monday_rebound_CN.md](20260911_ashare_monday_rebound_CN.md)

---

## How this sentence walks the repo

You ask a web AI: “A-shares fall on Monday and rise the next four days — is there a Monday effect?”

It often talks weekend news and “buy the open, sell Friday.” The calendar may be US; the fill may be Monday open — already a different sentence. There is no CSI 300 bear / 924 bull / recent window, and no ruling for “CAGR near zero.”

You say the same sentence here. The template locks: the paying side is Monday panic; `monday_down` is known at Monday **close**, earliest fill Tuesday open; hold four daily bars, not “must flatten Friday”; the calendar must be [`market_segment_ashare.yaml`](../../config/market_segment_ashare.yaml), not crypto 2022. After “measure this,” download daily bars, build the daily layer, run the event court.

The table: tiny positive CAGRs, recent +0.06%, Calmar falling from ~1.5 to 0.14. The hard CAGR / MaxDD gates are not hit, but calendar alpha does not pay for a sentence. A web AI will say “it’s still green.” This repo asks you to `--declare` against the written ruler: a few basis points are not a landing contract. A book that “didn’t lose” is not a license to trade every Monday by hand.

Grain follows the object: this is a daily calendar, not 2h ticks. The seven sections below unpack closed-bar and the A-share windows.

---

## Design

```text
Human sentence (Monday down, next four sessions up)
  → template: sociology / math / stats / ruler / A-share three windows / five boxes
  → validate + lineage
  → download daily bars, build FeatureStore — only after “measure this”
  → daily court: monday_down=1, hold 4 daily bars
  → human --declare
```

The grid sets `market_segment_path: config/market_segment_ashare.yaml`. Do **not** reuse crypto `bear_2022`. The auxiliary tape is retired; daily download is for this court example only — [RETIRED.md](../RETIRED.md).

| Box | This sentence |
|---|---|
| Mechanism | After Monday close is down (`monday_down=1`), long CSI 300; time-exit after 4 daily bars. |
| Regimes | Panic should clear. None of the three windows should blow the book. |
| Contract | Monday close first, Tuesday open earliest. Four bars, no adds, kill switch off, closed-bar. |
| Falsifier | Any window CAGR < 0, or recent MaxDD deeper than both trend windows. |
| Landing | Robot: `strategies/ashare_monday`. Same sentence for hands. |

| Template box | How this sentence is written |
|---|---|
| Sociology | Weekend bad news is priced Monday. The paying side is Monday panic, not “the market.” |
| Math | Daily return. `monday_down` known at Monday close. Hold 4 sessions. |
| Stats | Calendar alpha. A-share regime windows, not crypto dates. |
| Ruler | Any window CAGR < 0, or recent drawdown deeper than both trends. |
| Data range | `000300.SH` daily · `market_segment_ashare.yaml` · layer `features_ashare_monday_1D`. |

Clock (signal known at close t; position starts on **t+1**):

| Moment | What happens |
|---|---|
| Monday close | `monday_return < 0` becomes `monday_down=1` |
| Tuesday open | Earliest entry |
| Next four daily bars | Time exit (not “must flatten Friday”) |
| Flat days | Earn 0 |

Headline five KPIs only. Do not sum R.

---

## Data

```bash
mlbot research validate 20260911_ashare_monday_rebound
mlbot research index --trusted --query monday
mlbot data download-ashare --symbols 000300.SH --start-date 2019-01-01
```

| Item | This machine |
|---|---|
| Symbol | CSI 300 `000300.SH` |
| Bar | Daily (`1D`) |
| Source | `mlbot data download-ashare` |
| Path | `data/ashare/daily/` |
| Layer | `features_ashare_monday_1D` |
| Calendar | [`config/market_segment_ashare.yaml`](../../config/market_segment_ashare.yaml) |
| Kill switch | Off |

| Window | Span | Role |
|---|---|---|
| `bear_2021` | 2021-07-01 → 2022-10-31 | CSI 300 bear after the peak. |
| `bull_924` | 2024-09-24 → 2025-05-31 | Fast bull after the 24 Sep policy package. |
| `chop_recent` | 2025-06-01 → 2026-09-10 | Recent digestion. Cannot promote alone. |

`crash_2015` / `bear_2018` / `covid_2020` live in the same file for the tenbagger cohort. They are **not** this sentence’s three windows.

---

## Features

| Column | Construction | Closed-bar use |
|---|---|---|
| `weekday` | Day of week (0 = Monday) | Open reads the previous bar |
| `monday_return` | That Monday bar’s return | Known at Monday close |
| `monday_down` | 1 if `monday_return < 0` | First tradable Tuesday open |
| `atr_f` | Volatility for size | Previous bar only |

Pack: `config/experiments/20260911_ashare_monday_rebound/strategies/ashare_monday/`

| File | What it locks |
|---|---|
| `features.yaml` | `atr_f` + `weekday_monday_f` |
| `prefilter.yaml` | `monday_down ≥ 1` and `weekday == 0` |
| `direction.yaml` | `fixed_direction: long` |
| `execution.yaml` | `time_stop_bars: 4` |
| `meta.yaml` | `timeframe: 1D`, layer `features_ashare_monday_1D` |
| Grid | `monday_rebound_grid.yaml` (A-share calendar path) |

```bash
PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/experiments/20260911_ashare_monday_rebound/strategies/ashare_monday \
  --symbols 000300.SH --timeframe 1D \
  --root feature_store --layer features_ashare_monday_1D \
  --data-path data/ashare/daily \
  --start-date 2021-01-01 --end-date 2026-09-10 --no-reuse
```

Do not compute “is today Monday” inside the backtest.

---

## IC

**There is no cross-sectional IC table, and there should not be one.**

This is a calendar 0/1 permit. If someone wants “how much Monday fell” as a continuous score versus the next four days, that is a new folder. The close uses three-window five KPIs only.

---

## Validation

```bash
PYTHONPATH=src python -m scripts.event_backtest --variant-grid \
  config/experiments/20260911_ashare_monday_rebound/monday_rebound_grid.yaml
mlbot research close 20260911_ashare_monday_rebound
```

Ruler, written first: any window CAGR **< 0**, or recent MaxDD **deeper** than both trend windows.

### Three windows (000300.SH · daily · kill switch off)

| Window | CAGR | Calmar | Win rate | MaxDD | Sharpe(R) | Trades |
|---|---:|---:|---:|---:|---:|---:|
| `bear_2021` | +0.61% | 1.16 | 52.9% | −0.53% | 0.12 | 34 |
| `bull_924` | +0.46% | 1.52 | 50.0% | −0.30% | 0.12 | 16 |
| `chop_recent` | +0.06% | 0.14 | 58.8% | −0.45% | 0.03 | 17 |

| Written falsifier | Hit? |
|---|---|
| Any window CAGR < 0 | No. All three are tiny positives. |
| Recent MaxDD deeper than both trends | No. Recent −0.45% is shallower than bear −0.53% and deeper than bull −0.30%, not “deeper than both.” |

CAGR sits near zero. Recent Calmar falls from 1.16 / 1.52 to 0.14: the same shallow drawdown, almost no pay. Calendar alpha is weak. The verdict stays empty for the human `--declare`.

---

## Conclusion

Class: **calendar-alpha claim, measured near useless.**

- Both hard CAGR / MaxDD gates are not hit.
- Recent CAGR +0.06%, Sharpe 0.03: even if panic clears, it does not pay for a sentence.
- 16–34 trades is the density of down Mondays, not a license to include Tuesdays.
- Do not read this table as “the Monday effect works.” A few basis points are not a landing contract.

Declare with `--declare`. Do not hand-trade every Monday because the book “didn’t lose.”

| Path | What it is |
|---|---|
| `results/ashare_monday/experiments/20260911_ashare_monday_rebound/monday_down/bear_2021` | Bear report |
| `results/ashare_monday/experiments/20260911_ashare_monday_rebound/monday_down/bull_924` | 924 bull report |
| `results/ashare_monday/experiments/20260911_ashare_monday_rebound/monday_down/chop_recent` | Recent report |
| [DECISION.md](../../config/experiments/20260911_ashare_monday_rebound/DECISION.md) | Paper |

---

## How to read the report

1. **A-share calendar first.** These are `bear_2021` / `bull_924` / `chop_recent`. Crypto 2022 does not explain CSI 300.
2. **Headline is CAGR, not win rate.** Recent win rate is the highest (58.8%) and CAGR is the smallest.
3. **Calmar collapse is the mechanism.** Drawdowns stay under 1% while CAGR falls from 0.61% to 0.06%.
4. **Entry is not Monday open.** “Buy Monday open, sell Friday” is a different sentence.
5. **Four bars ≠ Friday.** Tuesday in, four daily bars, exit around the following week.
6. **16 trades in `bull_924` cannot promote alone.**
7. **Do not sum four-day ΣR.**
8. **Auxiliary tape is retired.** Index daily bars only.

The paper is [DECISION.md](../../config/experiments/20260911_ashare_monday_rebound/DECISION.md).
