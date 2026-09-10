# interpretable-ml-trading

**中文**: [README_CN.md](README_CN.md)

You have a trading idea and you are not sure it actually works.  
Open this repo and talk to the AI.

A web AI answers from articles and its own memory. You cannot reproduce that answer, and you can barely falsify it.  
The AI in this repo **must not use the web as evidence**. It has to use the trades you downloaded, features already computed in this tree, and one locked ruler. Then a person and a robot can share one conclusion you can audit.

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

A web AI often stops here and suggests RSI. This repo starts measuring here.

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

A second full loop (fade crowded funding; exit when the z-score returns through 0):  
[docs/examples/20260910_funding_fade.en.md](docs/examples/20260910_funding_fade.en.md) · [中文](docs/examples/20260910_funding_fade_CN.md)

---

## Not here yet

- Predicting tomorrow’s close
- AI mining a pile of factors and shipping when the score looks high
- Wired exchange orders
- Using web articles or Nansen / Glassnode / Dune screenshots as a substitute for local data

---

## Install once

So the AI can download bars, build features, and run the ruler on your machine. After that, keep talking.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
```

Download and FeatureStore commands: [docs/usage.en.md](docs/usage.en.md). Day to day you still say “measure this.”

---

The rest of the docs: [docs/README.md](docs/README.md)
