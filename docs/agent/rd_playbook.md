# Court playbook (any agent)

Product loop: [`docs/hypothesis.md`](../hypothesis.md).  
This file only answers **who writes what**, and which CLI command is allowed to touch which field.

Cursor `.mdc` / skills point here. Do not duplicate the loop in five rules.

Court vs CV / IC: [`rd_qa.yaml`](rd_qa.yaml) · [`rd_qa_CN.md`](rd_qa_CN.md) · [`rd_qa_EN.md`](rd_qa_EN.md)

---

## Who writes what

| Artifact | Who | Tool |
|---|---|---|
| Hypothesis sentence | **Human** | chat + `DECISION.md` body |
| Hypothesis template (anatomy + range) | **Human**, AI restates | `docs/hypothesis_template.md` · `mlbot research validate` |
| Feature code + `feature_dependencies.yaml` | Agent (after the column is named) | editor |
| FeatureStore / Phase 1 / Phase 3 numbers | **Program** (only if human asked to measure) | CLI |
| `gate_status` | **Program** | `mlbot research close` |
| `verdict` | **Human via CLI** | `mlbot research close --declare` |

Agents must not type `verdict:` into `DECISION.md`.  
Do not start FeatureStore / `event_backtest` unless the human asked to measure **this** sentence.

Public dummy family: `ma_cross`. Fixture: `config/experiments/_examples/`.

---

## Front-matter

```yaml
---
topic: 20260910_ma_cross_demo
strategy: ma_cross
harness: event_backtest
segments: [bear_2022, bull_2023_2024, recent_range_to_bear]
kill_switch: false
verdict:
kpi: {}
supersedes: []
tags: [ma-cross, demo]
---
```

`kpi` keys: `cagr` / `calmar` / `win_rate` / `maxdd` / `sharpe`. No `total_r`.

---

## Commands (do not mix them)

| Command | Writes | Does not |
|---|---|---|
| `mlbot research validate <id>` | nothing (prints template OK / INCOMPLETE) | truth, `verdict`, backtest |
| `mlbot research init <id>` | experiment dir from `_template` | backtest, `verdict` |
| `mlbot research run <id>` | unique `*_grid.yaml` via `event_backtest`, KS OFF | `verdict` |
| `mlbot research close <id>` | `gate_status` | `verdict` |
| `mlbot research close <id> --declare reject` | `verdict` (human) | invent promote |
| `mlbot research index` / `scorecard --write` / `review --write` | machine indexes | any judgment |
| `mlbot research stale` | lists promote rows whose FeatureStore layer is newer than the experiment | `verdict` |
| `mlbot research standardize [--write]` | empty-verdict YAML on old folders | backtest, `verdict` |
| `mlbot research replay <old_id>` | **new** court dir | overwrite parent, run |

`run_grid.py` in an experiment folder is not the court. Rewrite as `*_grid.yaml`.

`gate_status=court_ok` needs: matching harness, KS OFF, the three crypto segments, and a `capital_report.json` the gate can read.

---

## New hypothesis

```text
Human: I want <mechanism>.
Agent:
  1. Fill docs/hypothesis_template.md (sociology / math / stats /
     standard / data range / five boxes). Do not swap the claim.
     Check completeness against lessons in this tree — not web memory.
  2. mlbot research init … (if no folder yet)
     mlbot research validate <id>
     Incomplete → edit paper only.
  3. mlbot research index --trusted --query <english-slug>
     Hit → restate the close. Stop.
  4. mlbot research harness ma_cross
  5. If they only wanted the paperwork: stop after validate + init.
  6. If they asked to measure: FeatureStore if the column is new,
     Phase 1 optional, Phase 3, then
     mlbot research close <id>
     Human --declare. A reject binds the robot and the hands.
```

```bash
mlbot research index --trusted --query stop-loss
```

| `record_class` | Meaning |
|---|---|
| **trusted** | Human `--declare`. Restate; do not rescan. |
| **open** | Empty `verdict`. Continue the checklist. |
| **legacy** | Pre-court paperwork. Not a close until replay + `--declare`. |
| **example** | `_examples/`. Not in the index. Do not run. |

---

## Forbidden

- Hand-writing `verdict`
- Starting a backtest when the user only asked to shape the sentence
- Swapping the claim for a nearby pack
- Promote from IC / one window / kill-switch ON
- Treating `legacy` / inferred prose as a prior reject
- Quoting a promote KPI after a FeatureStore rebuild without `mlbot research stale`
- Telling the human to hand-trade a `reject`

---

Home: [README.md](../../README.md) · prev [Usage](../usage.md) · next [Architecture](../ARCHITECTURE.md) · [EN](../ARCHITECTURE.en.md) · [Lessons](../lessons.md)
