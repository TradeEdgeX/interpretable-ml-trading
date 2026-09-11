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

模板：[docs/hypothesis_template.md](../../../docs/hypothesis_template.md)。  
空格 / TODO 不算填完。先 `mlbot research validate {{TOPIC}}`，再谈下载和回测。

## 原句

（人说的那一句。AI 不许改题。）

## 现象解剖

### 社会学

（谁付钱、什么群体行为会重复。不要写「市场有效」。）

### 数学

（测量对象 + 闭棒。必须能落到 FeatureStore 的一列。）

### 统计学

（抽样声称 + 先分类：alpha / 肥尾收割 / beta 增强 / 无用。）

## 验证合同

### 验证标准

（哪一段、哪项五项 KPI 坏了，这句话就死。熔断关。不要合计 R。）

### 数据范围

品种：  
周期：  
日历：`bear_2022` / `bull_2023_2024` / `recent_range_to_bear`（`config/market_segment.yaml`）  
`recent_6m_oos` 不能单独结案。

## 五格

| 格 | 内容 |
|---|---|
| 机制 | |
| 预期市况 | |
| 合同 | |
| 证伪条件 | |
| 落地 | 机器人 YAML / 手做清单 / 两者同一句 |

## 分段结果（canonical 三段，kill switch OFF）

| 段 | 年化 | Calmar | WR | MaxDD | Sharpe |
|----|-----:|-------:|---:|------:|-------:|
| bear_2022 | | | | | |
| bull_2023_2024 | | | | | |
| recent_range_to_bear | | | | | |

## Promote

- [ ] `mlbot research validate {{TOPIC}}` 通过
- [ ] 三条杠（LAYER_PROMOTION_CRITERIA §1）
- [ ] canonical 三段齐 + kill switch OFF
- [ ] 对口 harness（本抽取副本只有 `event_backtest` / `ma_cross`）
- [ ] 肥尾定性时补去 Top-3
- [ ] `mlbot research close {{TOPIC}}` 后 `--declare`（禁止手写 verdict）
- [ ] `mlbot research index` 通过
