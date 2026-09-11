# Hypothesis template

**中文:** [hypothesis_template.md](hypothesis_template.md)  
Loop: [hypothesis.en.md](hypothesis.en.md)

This repo is a **hypothesis validator**, not a factor miner and not an order bot.  
A person writes the phenomenon into a template. The AI checks **completeness and ruler fit** against lessons locked in this tree — it must not say “this usually works” from web memory. Only a passing template may orchestrate commands.

---

## Fill this first

Copy into `DECISION.md`. Empty cells, TODO, and slogans do not count.

```markdown
## Claim
(The human sentence. The AI must not swap the question.)

## Phenomenon

### Sociology
Who pays whom, which group behavior repeats, why this is not a slogan.
Name the paying side (chasers, forced exits, textbook coordination). Do not write “markets are efficient.”

### Mathematics
What is measured: a cross, a distance, an envelope, a tail, a counting process.
It must land on a FeatureStore column. Say **closed-bar** (a decision at open may only read the previous close).

### Statistics
The sampling claim: better conditional expectation / a few fat-tail paths / exposure on a named factor.
Classify first: alpha / fat-tail harvest / beta overlay / useless.
Win rate alone is not the claim.

## Validation contract

### Validation standard
Which window and which of the five KPIs kill the sentence.
CAGR / Calmar / win rate / MaxDD / Sharpe only. No total R.
Kill switch off when ranking edge.

### Data range
Symbols, timeframe, calendar. Public court default:
`bear_2022` / `bull_2023_2024` / `recent_range_to_bear`
(`config/market_segment.yaml`).
`recent_6m_oos` cannot close alone.

## Five boxes
| Box | Content |
|---|---|
| Mechanism | What must be true before a trade is allowed |
| Regimes | Where it should hold; where it should fail |
| Contract | Cost of being wrong, adds, when it ends |
| Falsifiers | Same sentence as the validation standard |
| Landing | Robot YAML and human checklist, one sentence |
```

Anatomy answers why this is a social / mathematical / statistical phenomenon.  
The five boxes answer how the robot and the hands execute, and how the sentence dies.  
Both layers are required. Five boxes alone read like a strategy sheet, not a validator.

---

## The AI checks the template (experience = this tree, not the web)

```bash
mlbot research validate <id>
```

Pass: slots filled, a class named, five KPIs named, canonical segments named, closed-bar named.  
Fail: edit the template only. No download, no FeatureStore, no backtest.

The only “experience” the agent may use:

| Source | What it blocks |
|---|---|
| [lessons.md](lessons.md) | Same-bar open decisions, kill-switch ranking, total R, last-six-months only |
| [philosophy.en.md](philosophy.en.md) §5 | Wrong ruler (ex-Top-3 used to kill a trend sleeve) |
| [features.en.md](features.en.md) | `compute_*` inside the backtest |
| `mlbot research index --trusted` | Rescanning a declared sentence |

**Do not** treat training data or web articles as validation. That story changes with the model — the thing this tree shuts off.

---

## Orchestrate only after the template passes

| Slot | Allowed command | Still forbidden |
|---|---|---|
| Math named a new word | Register `*_f`; FeatureStore only if the column is missing | Building the store unasked |
| Data range | `mlbot data` (if they asked to measure) | Hand-rolled klines |
| Standard + three windows | `mlbot research run` (kill switch off) | A different harness, one pooled window |
| Five-box landing | YAML or checklist, same sentence | A live take-profit / average-down never measured |
| Numbers exist | `mlbot research close`; human `--declare` | AI types `verdict` |

If they did not say “measure this”: `validate` + `init` + lineage, then stop.

---

## Golden-cross anatomy (shape, not a new result)

| Slot | This sentence |
|---|---|
| Sociology | The textbook 50/200 is a coordination ritual. In a one-way year late trend money pays those already onside; in a range they pay each other’s fees. |
| Math | Closed-bar EMA50 / EMA200; the event is a sign change of `ema50-ema200`; void is a close through EMA50. |
| Statistics | Treat as **beta / trend exposure**, not entry alpha. Three windows, listed separately; recent negative CAGR kills the sentence. |
| Standard | Any window CAGR &lt; 0, or range drawdown worse than the trend window. |
| Range | BTCUSDT · 2h · the three windows in `market_segment.yaml`. |

Numbers still come from the local FeatureStore. See the table in [README.md](../README.md).

---

Landing: [README.md](../README.md) · [hypothesis.en.md](hypothesis.en.md)
