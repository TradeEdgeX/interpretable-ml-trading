---
topic: 20260823_ma50_ma200_cross
role: harness-example
strategy: ma_cross
harness: event_backtest
segments: [bear_2022, bull_2023_2024, recent_range_to_bear]
kill_switch: false
verdict:
kpi: {}
supersedes: []
tags: [ma-cross, ema-cross, harness-example]
---

# DECISION — harness fixture (do not run)

This directory is a **filled-in example** of the agent court, not a live
research experiment. `_examples/` is skipped by `mlbot research index`.

Copy the shape with `mlbot research init <date>_<slug> --strategy ma_cross`
when a human actually has a hypothesis. Do not backfill FeatureStore or
run `event_backtest` just because this folder exists.
