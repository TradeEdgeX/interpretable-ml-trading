---
name: rd-experiment
description: >-
  Turn a trading idea into a falsifiable hypothesis template, check
  lineage, and (only if asked) measure it on the court. Use when the
  user proposes a rule, factor, stop, clock, or moving-average idea —
  for a robot or for a handwritten checklist.
---

# Hypothesis validator

Follow [`docs/hypothesis_template.md`](../../../docs/hypothesis_template.md),
[`docs/hypothesis.md`](../../../docs/hypothesis.md), and
[`docs/agent/rd_playbook.md`](../../../docs/agent/rd_playbook.md).
Lessons: [`docs/lessons.md`](../../../docs/lessons.md).
Map: [`README_CN.md`](../../../README_CN.md).
Court vs CV: [`docs/agent/rd_qa.yaml`](../../../docs/agent/rd_qa.yaml).

This repo is a **hypothesis validator**. Do not skip the template.

1. Restate the claim in the template: sociology, math, statistics,
   validation standard, data range, then the five boxes
   (mechanism, regimes, contract, falsifiers, landing).
   Do not swap it for a nearby pack. Same sentence for YAML and for a
   manual checklist.
   Template check uses **this tree’s lessons**, not web memory.
2. `mlbot research validate <id>` after the folder exists.
   Incomplete → edit paper only. No download, no backtest.
3. `mlbot research index --trusted --query <english-slug>`
   Trusted hit → restate the close. Stop.
4. If they only wanted the sentence / paperwork:
   `mlbot research init YYYYMMDD_<slug> --strategy ma_cross` and stop.
   Do not download or backtest.
5. Missing measurement → Phase 0 (code + `feature_dependencies.yaml` + FeatureStore).
   No local `compute_*` in the backtest.
6. If they asked to measure: Phase 1 optional, then `mlbot research run <id>`
   (unique `*_grid.yaml`, kill switch OFF).
7. `mlbot research close <id>` then the human `--declare`.
   Never type `verdict:` yourself.
8. A `reject` means the robot does not run it **and** the human does not hand-trade it.
9. Public demo family is `ma_cross`.
