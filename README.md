# interpretable-ml-trading

**中文**: [README_CN.md](README_CN.md)

You have a trading idea and you are not sure it actually works.  
Open this repo and talk to the AI.

This repo is a **hypothesis validator**. First write the idea as a template: what social / mathematical / statistical phenomenon it is, what kills it, which data range measures it. The AI only checks that the template is complete and fits the ruler, then orchestrates commands. It **must not use the web as evidence**, and it must not declare for you.

A web AI answers from articles and its own memory. You cannot reproduce that answer, and you can barely falsify it.  
Here the AI has to use the trades you downloaded, features already computed in this tree, and one locked ruler.

Template: [docs/hypothesis_template.en.md](docs/hypothesis_template.en.md).

Validation only — no auto-mined factors. Execution comes later as a generic layer (Nautilus). The same rules you already validated can plug in.

---

## Complete example: a moving-average golden cross

### What you ask a web AI

> Long BTC when price is above the 50-day and golden-crosses the 200; exit if it breaks the 50. Does that make money?

It usually says: classic trend rule, works over the long run, fake signals in a range, add more filters.  
No bear / bull / last-few-years numbers. No contract for what counts as wrong. Ask another model, get another story.

### What you say here — same sentence

> Price above the 50-day, golden-cross the 200, exit if it breaks the 50.  
> I expect it to pay in trend years and not bleed too much in range years. Measure this.

You still just talk. The difference is the AI **must take the path below**.

### 1. Write it as a few testable lines

| Box | This sentence |
|---|---|
| Mechanism | Close above the 50-day, and it just crossed above the 200, before a long is allowed. Shorts are symmetric. |
| Regimes | Trend years should pay. Range-year drawdown must not be worse; CAGR must not go negative. |
| Contract | A close through the 50-day ends it. No adds, no averaging down. No extra take-profit. |
| Falsifiers | Range-window drawdown worse than the trend window, or any window with negative CAGR. |
| Landing | Same sentence for the robot and for your hands. |

| Template slot | This sentence |
|---|---|
| Sociology | The textbook 50/200 is a coordination ritual. Chasers pay in a one-way year; they pay each other’s fees in a range. |
| Mathematics | Closed-bar EMA50 / EMA200 sign change; void is a close through EMA50. |
| Statistics | Treat as beta / trend exposure, not entry alpha. Three windows, listed separately. |
| Validation standard | Any window with negative CAGR, or a deeper range drawdown. |
| Data range | BTCUSDT · 2h · `bear_2022` / `bull_2023_2024` / `recent_range_to_bear`. |

A web AI often stops here and suggests RSI. This repo first fills the [hypothesis template](docs/hypothesis_template.en.md) (who pays, what is measured, how it is sampled, which window, which KPI kills it). Only a passing template may be measured.

### 2. Not the web — your machine

| What | Where it comes from | What a web AI does not have |
|---|---|---|
| Bars | Binance trades you downloaded, converted to parquet | Not a screenshot from a blog, not prices from model memory |
| Features | Monthly FeatureStore columns: 50-day, 200-day, golden / death cross (100+ registered features in this tree) | Not a moving average typed into the chat and aligned to the wrong bar |
| Clock | A decision at bar open may only read the **previous closed** row | Web backtests often use the same bar’s high/low; CAGR looks fake-good |
| Ruler | Bear 2022-01→2023-11, bull 2023-06→2025-01, recent 2025-01→2026-05. Kill switch off. Report CAGR, Calmar, win rate, MaxDD, Sharpe | Not “it looked fine for the last six months” |

If a column is missing, backfill the FeatureStore. Do not invent it inside the backtest.  
Demo pack: `config/strategies/ma_cross/`. The golden cross uses `ema_50_200_cross_*`, not a webpage daily chart.

### 3. What the machine filled (BTC, 2-hour bars, EMA50/200)

Not the daily chart from a webpage. Entries are the cross; every exit is a close through EMA50. Numbers come from local trades and the FeatureStore.

| Window | CAGR | Calmar | Win rate | MaxDD | Sharpe(R) | Trades |
|---|---|---|---|---|---|---|
| Bear 2022 | +3.7% | 1.51 | 38.3% | −2.4% | 0.15 | 47 |
| Bull 2023–2024 | +3.0% | 0.97 | 33.3% | −3.1% | 0.15 | 33 |
| Recent range → bear | −3.1% | −0.68 | 24.3% | −4.5% | −0.36 | 37 |

Recent CAGR is negative. That hits the falsifier you wrote down.  
**You write the verdict.** Fail: the robot does not run it, and you do not hand-trade it tonight.

A web AI cannot produce this table.

### 4. You can also say

> Has anyone already measured a −8% stop?  
> Don’t backtest. Just open an experiment folder.  
> This idea is dead. Record that.

Already judged → restate the close. Do not scan it again.  
If you did not ask to measure, it must not download or backtest.

### More examples

- Fade crowded funding: [docs/examples/20260910_funding_fade.en.md](docs/examples/20260910_funding_fade.en.md) · [中文](docs/examples/20260910_funding_fade_CN.md)
- A-share Monday down → four sessions: [docs/examples/20260911_ashare_monday_rebound.en.md](docs/examples/20260911_ashare_monday_rebound.en.md) · [中文](docs/examples/20260911_ashare_monday_rebound_CN.md)
- BTC surge → AI alts follow: [docs/examples/20260911_btc_lead_ai_alts.en.md](docs/examples/20260911_btc_lead_ai_alts.en.md) · [中文](docs/examples/20260911_btc_lead_ai_alts_CN.md)
- P99 large trade + Bollinger chase: [docs/examples/20260911_p99_bb_break_chase.en.md](docs/examples/20260911_p99_bb_break_chase.en.md) · [中文](docs/examples/20260911_p99_bb_break_chase_CN.md)
- Tenbagger / small-cap hold (cohort, measured): [docs/examples/20260911_tenbagger_smallcap.en.md](docs/examples/20260911_tenbagger_smallcap.en.md) · [中文](docs/examples/20260911_tenbagger_smallcap_CN.md)

---

## Install once

So the AI can download bars, build features, and run the ruler on your machine. After that, keep talking.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
```

Download and FeatureStore commands: [docs/usage.en.md](docs/usage.en.md).  
How features are computed, order-flow / math support, and how to add a column without rebuilding the layer: [docs/features.en.md](docs/features.en.md).  
Day to day you still say “measure this.”

How the person, the instructions, and the commands share one ruler: [docs/ARCHITECTURE.en.md](docs/ARCHITECTURE.en.md).  
Stack, how to use, and how to choose timeframe / calendar / symbols (with the examples): [docs/framework.en.md](docs/framework.en.md).

---

The rest of the docs: [docs/README.md](docs/README.md)
