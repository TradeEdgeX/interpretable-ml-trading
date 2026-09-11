---
topic: "20260911_p99_bb_break_chase"
strategy: "ma_cross"
harness: event_backtest
segments: [bear_2022, bull_2023_2024, recent_range_to_bear]
kill_switch: false
verdict:
kpi: {}
supersedes: []
tags: [p99, bollinger, chase, fattail]
---

# DECISION — 20260911_p99_bb_break_chase

## 原句

根内有 P99 大单、又破布林上轨，就追涨。测一下。

## 现象解剖

### 社会学

大单是主动吃货；破上轨是拥挤动量，后到的人付钱。

### 数学

根内成交额最大值 vs 滚动 P99（tick → `bar_max_notional_ge_p99`）；`bb_position` ≥ 1。闭棒。出场：回到带内（`bb_position < 1`）或时间到（12 根 2h）。

### 统计学

动量 / 肥尾右尾声称。去 Top-3 只作分类，不单独否。抽样币圈三段。

## 验证合同

### 验证标准

任一段年化 < 0，或近窗回撤深于趋势段。年化 / Calmar / 胜率 / MaxDD / Sharpe。熔断关。

### 数据范围

`BTCUSDT` · 2h · `bear_2022` / `bull_2023_2024` / `recent_range_to_bear` · 层 `features_p99_bb_chase_120T`。

## 五格

| 格 | 内容 |
|---|---|
| 机制 | P99 大单且 `bb_position≥1` 做多。 |
| 预期市况 | 拥挤动量延续；三段不该做穿。 |
| 合同 | 回到带内或 12 根时间出场；不加仓；熔断关。闭棒。 |
| 证伪条件 | 任一段年化 < 0，或近窗回撤深于趋势段。 |
| 落地 | 机器：本目录 `strategies/ma_cross`。人手同一句。 |

## 谱系

公开仓库此前无已结案 P99+布林追涨句。

## 分段结果（BTCUSDT · 2h · 熔断关 · 层 `features_p99_bb_chase_120T`）

| 段 | 年化 | Calmar | 胜率 | 最大回撤 | Sharpe(R) | 笔数 |
|----|-----:|-------:|---:|------:|-------:|---:|
| bear_2022 | +0.24% | 0.32 | 45.5% | −0.74% | 0.09 | 22 |
| bull_2023_2024 | −0.25% | −0.26 | 36.4% | −0.95% | −0.10 | 22 |
| recent_range_to_bear | +0.28% | 0.92 | 57.1% | −0.30% | 0.16 | 21 |

出场几乎全是 `structural_exit_bb_position_lt1`（回到带内）。牛段年化为负，按证伪线已打中。判决仍留空。

产物：`results/ma_cross/experiments/20260911_p99_bb_break_chase/p99_bb/{bear_2022,bull_2023_2024,recent_range_to_bear}`
