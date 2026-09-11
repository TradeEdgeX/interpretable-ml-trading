# Philosophy

**中文:** [philosophy.md](philosophy.md)  
Loop: [hypothesis.en.md](hypothesis.en.md).

---

## 1. What the system optimizes

Not maximum profit. This repo is a **hypothesis validator**: on real market regimes, confirm as fast as possible that a *complete* trading sentence works, fails, or works only under named conditions. Fill the [template](hypothesis_template.en.md) first. The AI checks the template against this tree, not web memory.

> A full loop × real time cost = real knowledge. Half-paths and time-compressed backtests do not count.

“Fast” means time-to-falsification, not time-to-riches.

---

## 2. Information layers (no jurisdiction crossing)

| Saying | Question | Data |
|---|---|---|
| Price tells you WHAT | What happened | Price / structure |
| Volume tells you IF | Was it accepted | Participation |
| Order flow tells you WHO | Who pushed | Trades / clustering |
| Math tells you HOW CAREFULLY | How conservative to be | Noise / tails / spectrum |

Structure may say “this is a break.” Math may not overrule that into “then it is not a break.” Math may only say “noise is high, size down.”

> Gate protects correctness. Execution decides exposure. Math controls caution. Account gates block new risk; they do not flatten a hold mid-path.

---

## 3. Heuristics are the prior; the framework falsifies

“Do not break into an order-flow vacuum.” “A shallow pullback is a fake.” These are trading philosophy, not model discoveries.

Trees, nets, IC, and lift can **falsify** the scenes where a heuristic dies. The decision is still a human sentence (YAML or a checklist). Tools can change; the loop cannot:

```
human hypothesis
  → scan features (sharper hypotheses only)
  → event backtest (canonical segments, kill-switch off)
  → human review: explainable, same sign across segments, risk not worse
  → lock — or REJECT
```

Positive IC is not a ship. A pretty recent window is not a ship. A kill-switch-truncated curve is not an edge ranking.  
A rejected sentence is also forbidden to hand-trade.

---

## 4. Popper, not hyperparameter search

| Step | In this repo |
|---|---|
| Hypothesis | One mechanism + a contract (YAML or checklist) |
| Test | Backtest / failure audit / ex-Top-3 (classification only) |
| Falsify | Sign flip, look-ahead, friction → 0 |
| Update | Delete the rule, mark RETIRED, stop doing it by hand |

The next round only attacks failures the last verdict does not already cover. The objective is unknown error outside known error — not Sharpe.

---

## 5. Classify before you twist knobs

| Class | Source of money | Keep if | Do not |
|---|---|---|---|
| **Alpha** | Better conditional expectation | Ex-Top-3 still > 0; sign-stable | Cite bull right-tail as alpha |
| **Fat-tail harvest** | Few extreme paths | Sum dominated by Top-3 | Cut the right tail to raise win rate |
| **Beta overlay** | Named-factor exposure | Residual after the factor ≈ 0 | Use a short detector to “protect” slow beta |
| **Dead** | None | Cross-segment ≤ 0 | Keep sweeping thresholds |

Detectors choose a path. Harvesters (adds, hold, structural exit) decide how thick the bite is. Do not book execution convexity as signal alpha.

**Factor = a common paid source; beta = your exposure on it; alpha = what remains after those exposures are subtracted.**  
Most “statistical alpha” lives inside beta. After a same-window hold and comparable drawdown, the residual is often ≈ 0.

Long form: [design/alpha_vs_fattail_vs_beta_CN.md](design/alpha_vs_fattail_vs_beta_CN.md). Ex-Top-3 is not a capacity veto for trend sleeves: [design/ex_top3_capacity_confound_CN.md](design/ex_top3_capacity_confound_CN.md).

---

## 6. The chart is the contract, not a second forecast

A trade needs a contract: when you are in, when you are wrong, how much that costs, whether you may add, when the ticket is over. Geometry folds bars into **inside / outside / void**. YAML and a handwritten checklist write the same thing.

| Layer | Answers only | Does not answer |
|---|---|---|
| Slow MA / regime | Long / short / flat permission today | The entry print |
| Large level | Whether a ticket is worth writing | Direction (regime already did) |
| Structure | This ticket’s stop, void, opposite side | Moving the stop onto another layer |

Ablation is not “no contract.” It is *which* geometry writes it. **There is no zero-contract trade.**  
Do not ship a combined “MA + level + structure-arrow” indicator.

---

## Further reading

| Essay | Path |
|---|---|
| Hypothesis template | [hypothesis_template.en.md](hypothesis_template.en.md) |
| Falsify a hypothesis | [hypothesis.en.md](hypothesis.en.md) |
| Lessons | [lessons.md](lessons.md) |
| Human claim vs auto-mine | [design/2026-08-23_human_hypothesis_vs_auto_mine_CN.md](design/2026-08-23_human_hypothesis_vs_auto_mine_CN.md) |

---

Home: [README.md](../README.md) · prev [Hypothesis](hypothesis.en.md) · next [Lessons](lessons.md)
