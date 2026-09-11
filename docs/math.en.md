# Math

**中文:** [math.md](math.md)  
**Landing:** [../README.md](../README.md)

Loop: [hypothesis.en.md](hypothesis.en.md). This page only locks the math the harness still enforces. Implementations live in `config/feature_dependencies.yaml` and `src/features/`. How to build, add a column, order flow: [features.en.md](features.en.md).

---

## 1. Closed-bar clock (no look-ahead)

FeatureStore / feature-bus:

- Row index = bar **open** \(T\)
- That row’s OHLC and features use \([T, T+\Delta)\)
- So the \(T\) row is knowable only at wall \(T+\Delta\) (close)

Public dummy `event_backtest` aligns to bar close, then decides.

**Illegal:** decide at bar open with the same-index completed FeatureStore row. A grid once printed 50–70% CAGR that way. Those numbers are vacated. See [lessons.md#closed-bar](lessons.md#closed-bar).

Labels (`forward_rr*`) may look forward. They must never be model X or an entry feature.

---

## 2. Feature layers: cause vs caution

| Layer | May decide | Must not |
|---|---|---|
| Structure / volume / order flow | Gate: trade or not | Use slow math as the master veto |
| Math (Hurst / WPT / spectrum / Hilbert) | Execution: noise penalty, smaller exposure | Act as a gate that zeroes historical failures |
| Slow regime | Whether this *kind* of market is in play | Act as 2h entry alpha |

One sentence: **math does not answer “is this a break”; it answers “how careful now.”**

Keep three objects out of one feature column:

| | What it is | Example |
|---|---|---|
| **Factor / regime** | Which side you are willing to stand on | `ema_1200_position` band; a daily 200 is permission, not an entry |
| **Contract geometry** | Inside / outside / void; defines R | Opposite SR, box edge, break of 200 |
| **Feature** | A **measurement** of geometry or flow | `wide_sr_dist_atr`, `srb_sr_success_breakout_score`, CVD |

FeatureStore does not sign the ticket. Archetype YAML writes enter / stop / add / exit. Stacking measurements into one score tempts you to treat “together” as stronger alpha.

Path-2.5 evidence tiers are not the live SRB path; “math is not a gate” still is. Older BPC note: [architecture/path2.5_math_features.md](architecture/path2.5_math_features.md)

---

## 3. Objective functions (how not to fool yourself)

**Promotion (three bars)** — [LAYER_PROMOTION_CRITERIA.md](../config/experiments/LAYER_PROMOTION_CRITERIA.md)

1. On canonical segments, expectation is not worse everywhere (bear 2022 / bull 2023–24 / recent)
2. MaxDD does not deteriorate
3. The rule is explainable and regime-aware

Also:

- Rank **edge** with kill-switch **OFF**. Halts at different times fake winners.
- Fat-tail sleeves also run **ex-Top-3**. Almost every trend book goes negative in bull after Top-3 — that is the shape.
- Public KPIs: **CAGR, Calmar (CAGR/|MaxDD|), win rate, MaxDD%, Sharpe**.  
  `pnl_r = pnl / current risk_budget` shrinks fat-tail exits as equity compounds. Do not headline Total R.

IC / deciles / label lift generate hypotheses only.

---

## 4. Detector vs harvester (different objectives)

- Detector: hit rate may be low; question is “did we step on the right path?”
- Harvester: leverage, adds, hold, structural exit; question is “how thick, how far from liquidation?”

Rolling risk must be measured with the **dedicated harness** (`trend_rolling_simulate.py`). Proxying roll P&L with B-layer event backtests is a contract error. Coin-margined rolling reached wipeout-scale DD at matched params → **USDT-M only**.

---

## 5. Account and sizing arithmetic

- Startup: REST equity retries 3× / 30s, then **refuse to start** (no silent 10k).
- Each new open: refresh equity again; snapshot fail **blocks the open**; closes still allowed.
- Kill-switch G0: block new risk / adds / spot buys; allow reduce / close / spot sells. Not a mid-hold flatten.
- One process, one key pair. Missing keys fail closed.

Coin-margined “high returns” are often inverse convexity, not alpha. Align evaluation to the same-window USD hold of the coin plus comparable MaxDD.

---

## 6. A-shares (a different contract)

The oversold detector shrinks the pool. The book is **8 equal slots + a fixed hold clock**, not a stop machine. The hold plateau is 40–50 sessions. Shallow / breakeven stops have been falsified repeatedly. New levers must pass a detector-perturbation gate (several RSI×amp cells, same sign). A single-threshold “optimum” is void.

---

## 7. Ablation baselines

New geometry or filters must beat two baselines before promote:

1. **Buy and hold** (same window, same universe, comparable MaxDD, USD curve)
2. **Slow regime only** (e.g. stand on one side of EMA1200 / daily 200)

If the stacked layers do not beat (2), drop the box and S/R; demote the contract to the MA. A clean visual only earns the right to measure.

---

Home: [README.md](../README.md) · prev [Lessons](lessons.md) · next [Usage](usage.en.md)
