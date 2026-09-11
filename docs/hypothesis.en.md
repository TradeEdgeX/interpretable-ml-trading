# Falsify a hypothesis

**中文:** [hypothesis.md](hypothesis.md)

You do not need this page day to day — talk to the AI ([README.md](../README.md)).  
This page is for the AI: how to turn the user’s sentence into a few testable lines.

A backtest and a handwritten checklist execute the same sentence. Only the actor changes.

---

## 1. The loop

```text
Human writes the claim
  → AI helps register features / rules / a checklist (does not swap the claim)
  → The program measures on a locked harness (closed bar, three segments, kill-switch off)
  → Human declares: holds / fails / needs more evidence
  → Holds: YAML, or a human checklist
     Fails: the robot does not run it; the human may not hand-trade the same sentence
```

Discovery throughput is not the main loop. Grading is.

This is the opposite of “AI mines factors overnight; a higher IC becomes the next baseline.” That loop optimizes a score. This one asks whether **one sentence died**.

---

## 2. One sentence, two landings

A hypothesis is not a strategy name. Example:

> Go long when price is above the 50-day average and has just crossed the 200.  
> Void if price closes back through the 50.  
> Should pay in trend years and not bleed too much in range years.  
> If the range year drawdown is worse or CAGR is negative, the sentence is dead.

| Landing | Use |
|---|---|
| **Quant** | YAML (`ma_cross`-style): permission, side, void, size. Five KPIs from `event_backtest`. |
| **Manual** | Checklist: allowed today?, entry, void price, no average-down, clock. Execute when hit; do not debate. |

A court `reject` binds both: **do not spend money on this sentence.**  
Do not hand-trade at night what the backtest already killed.

`promote` only means: on the agreed ruler, the sentence is not yet dead.  
It is not “go full size tonight.” The robot follows YAML. The human follows the checklist. Neither invents a third method at the screen.

---

## 3. Template first, then five boxes, then measure

This repo is a hypothesis validator. Five boxes alone read like a strategy sheet. First name the social / mathematical / statistical phenomenon, the ruler, and the data range that can kill the sentence.

Full slots, how the AI checks them, and what it may orchestrate: [hypothesis_template.en.md](hypothesis_template.en.md).

Do not download or backtest until the template exists.

```text
Claim              the human sentence (the AI does not swap it)
Sociology          who pays; which group behavior repeats
Mathematics        the measured object + closed bar
Statistics         the sampling claim + classify first (alpha / fat-tail / beta / useless)
Validation standard which window and which of the five KPIs kill it
Data range         symbols, timeframe, the three canonical windows
Five boxes         mechanism / regimes / contract / falsifiers / landing
```

The AI checks structure (completeness, fit to lessons), **not** truth from memory:

```bash
mlbot research validate <id>
mlbot research index --trusted --query <english-slug>
```

Template fails → edit paper only. Trusted hit → restate the close. Stop.  
It must not swap your sentence for a nearby pack (a 50/200 cross ≠ an EMA dead zone ≠ a structure break).

---

## 4. Feature discovery (measurement, not score mining)

A new column must name a word in the claim (“just crossed”, “distance to void”).

1. Name the event (human).
2. Implement a FeatureStore column; register it in `config/feature_dependencies.yaml`.
3. Closed bar: an open-indexed row is knowable only at close. See [math.en.md](math.en.md) §1.
4. Missing column → backfill. No local `compute_*` in the backtest path.

Features measure. Entry / stop / add / exit are the **contract**, in YAML or on the checklist.  
Do not stack ten measurements into one score and treat the stack as a stronger hypothesis.

Automatic search (IC climbing, feature-group search) is Phase 1 torchlight only: it may show *where* the sentence looks dead.  
A higher score cannot close and cannot edit a locked rule. See [design/2026-08-23_human_hypothesis_vs_auto_mine_CN.md](design/2026-08-23_human_hypothesis_vs_auto_mine_CN.md).

---

## 5. What “training” means here

This tree does **not** close on a fitted model. There is no `mlbot train`. TimeSeriesSplit averages are not a verdict.

Training here means making the ruler reproducible:

| Step | Does | Is not |
|---|---|---|
| FeatureStore | Monthly closed-bar columns | Fitting a predictor |
| Phase 1 scan | Where the sentence lights up | A close |
| Phase 3 backtest | Bear / bull / recent, kill-switch off | Tuning until it looks nice |

Rule overfitting is sweeping thresholds and keeping the pretty window — not memorizing one bar in 2022.  
Gates: freeze the claim, report three segments, fail if one segment fails, never hide drawdown behind a kill switch. Q&A: [agent/rd_qa.yaml](agent/rd_qa.yaml).

---

## 6. Measure

Start the harness only after the human asked to measure this claim.

```bash
mlbot research init 20260910_<slug> --strategy ma_cross
mlbot research run 20260910_<slug>
mlbot research close 20260910_<slug>
mlbot research close 20260910_<slug> --declare reject
```

| Required | Why |
|---|---|
| Matching harness (public dummy = `event_backtest` / `ma_cross`) | A different machine is a different exam |
| Three segments | One pooled run averages away the bad year |
| Kill-switch off | Variants halt at different times; edge is not comparable |
| CAGR / Calmar / win rate / MaxDD / Sharpe | Do not headline summed R |

Classify first: alpha, fat-tail harvest, beta overlay, or dead. Different rulers.  
[design/alpha_vs_fattail_vs_beta_CN.md](design/alpha_vs_fattail_vs_beta_CN.md)

Commands: [usage.en.md](usage.en.md). Who writes numbers vs verdict: [agent/rd_playbook.md](agent/rd_playbook.md).

---

## 7. Reading a verdict with your hands

| `verdict` | Robot | Human |
|---|---|---|
| `reject` | Do not write YAML | **Do not trade the sentence.** Not “half size” or “other symbol” |
| `park` / `needs-more` | Do not deploy | Do not start. You lack evidence, not courage |
| `promote` | Locked YAML only | Same-sentence checklist: permission, void, clock |

A manual checklist has the same four boxes as YAML:

1. **Permission** — is this regime in-scope today
2. **Entry** — is the main signal present (math noise is not a structure veto)
3. **Void** — price hit → end. Changing the exit is a new claim
4. **Clock** — no adds, no swaps, no “wait and see” before the clock

The framework can guide discretionary work because it **removes invention at the screen**. It cannot replace watching the tape for a feeling.

---

## 8. Do not turn this into

- Auto-mined factors that ship when the score stops moving
- A CV average instead of three-segment judgment
- A single recent window
- Hand-trading a rejected sentence
- Adding an untested take-profit or average-down after a promote
- Calling this repo an interpretable ML trading bot

---

Home: [README.md](../README.md) · next [Philosophy](philosophy.en.md) · [Lessons](lessons.md)
