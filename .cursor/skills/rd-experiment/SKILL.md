---
name: rd-experiment
description: >-
  Turn a trading idea into a falsifiable hypothesis, check lineage, and
  (only if asked) measure it on the court. Use when the user proposes a
  rule, factor, stop, clock, or moving-average idea — for a robot or for
  a handwritten checklist.
---

# Hypothesis court

Follow [`docs/hypothesis.md`](../../../docs/hypothesis.md) and
[`docs/agent/rd_playbook.md`](../../../docs/agent/rd_playbook.md).
Lessons: [`docs/lessons.md`](../../../docs/lessons.md).
Map: [`README_CN.md`](../../../README_CN.md).
Court vs CV: [`docs/agent/rd_qa.yaml`](../../../docs/agent/rd_qa.yaml).

1. Restate the claim in five boxes (mechanism, regimes, contract, falsifiers, landing).
   Do not swap it for a nearby pack. Same sentence for YAML and for a manual checklist.
2. `mlbot research index --trusted --query <english-slug>`
   Trusted hit → restate the close. Stop.
3. If they only wanted the sentence / paperwork:
   `mlbot research init YYYYMMDD_<slug> --strategy ma_cross` and stop.
   Do not download or backtest.
4. Missing measurement → Phase 0 (code + `feature_dependencies.yaml` + FeatureStore).
   No local `compute_*` in the backtest.
5. If they asked to measure: Phase 1 optional, then `mlbot research run <id>`
   (unique `*_grid.yaml`, kill switch OFF).
6. `mlbot research close <id>` then the human `--declare`.
   Never type `verdict:` yourself.
7. A `reject` means the robot does not run it **and** the human does not hand-trade it.
8. Public demo family is `ma_cross`.
