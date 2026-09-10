---
# Machine-readable lineage — read by `mlbot research index`.
# Fill `verdict` only when the experiment is closed; harness / segments /
# kill_switch become mandatory at that point.
topic: "{{TOPIC}}"
strategy: "{{STRATEGY}}"
harness: event_backtest          # must match the engine (see KNOWN_HARNESSES)
segments: [bear_2022, bull_2023_2024, recent_range_to_bear]
kill_switch: false               # edge ranking requires OFF
verdict:                         # do not type this; mlbot research close --declare
kpi: {}                          # {cagr, calmar, win_rate, maxdd, sharpe} — never total_r
supersedes: []                   # experiment ids this one overrides
tags: []                         # English slugs only: stop-loss, take-profit, dryup
---

# DECISION — {{TOPIC}}

## 假设

| ID | 假设 | Phase 1 证据 | 决策 |
|----|------|--------------|------|
| H1 | | | pending |

## 分段结果（canonical 三段，kill switch OFF）

| 段 | 年化 | Calmar | WR | MaxDD | Sharpe |
|----|-----:|-------:|---:|------:|-------:|
| bear_2022 | | | | | |
| bull_2023_2024 | | | | | |
| recent_range_to_bear | | | | | |

## Promote

- [ ] 三条杠（LAYER_PROMOTION_CRITERIA §1）
- [ ] canonical 三段齐 + kill switch OFF
- [ ] 对口 harness（本抽取副本只有 `event_backtest` / `ma_cross`）
- [ ] 肥尾定性时补去 Top-3
- [ ] `mlbot research close {{TOPIC}}` 后 `--declare`（禁止手写 verdict）
- [ ] `mlbot research index` 通过
