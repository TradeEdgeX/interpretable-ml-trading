---
topic: "20260914_ai_chip_spend_btc_regime"
strategy: "ai_chip_spend_btc"
harness: phase1_scan_only
segments: [bear_2022, bull_2023_2024, recent_range_to_bear]
kill_switch: false
verdict:
kpi: {}
supersedes: []
tags: [ai-chip-spend, regime, beta, epoch]
---

# DECISION — 20260914_ai_chip_spend_btc_regime

## 原句

AI 算力开支过多时，BTC 处于熊或震荡，而不是慢趋势牛。不要看随后几天涨跌，看闭棒市况。

## 现象解剖

### 社会学

付钱的是还按 2023–24 牛市模板做多 BTC 的人。现金和风险预算被抽去买加速器；前沿实验室和云厂商在付芯片账单，不是在给 BTC 永续添多头。这不是有效市场口号。

### 数学

X 是锁定季频 `config/research/ai_chip_sales_quarterly.csv`（Epoch AI chip sales，CC BY；金额 = 当季各芯片 `Cost Estimate (USD)` 之和）。该季数字在 `end_date` 的下一根 UTC 日才可用。主强度用环比 `ai_chip_spend_qoq`，过高 = 环比大于自身扩窗中位数（至少 4 季）。不用金额水平：水平几乎单调上升，会和日历年重合。Y 是 BTC 日线闭棒：收盘低于 200 日均线记非牛。不用 `market_segment.yaml` 标签当 Y。开盘决策只读上一根已收盘。

### 统计学

先当 **beta / 市况共存**，不是选点 alpha。抽样声称：高强度日里闭棒非牛比例高于低强度日。三段只分列，不当答案。独立牛熊大约两段，Phase 1 不能结案。

## 验证合同

### 验证标准

高强度日闭棒非牛比例 ≤ 低强度日，这句话死。分列 `bear_2022` / `bull_2023_2024` / `recent_range_to_bear` 若有一段反向，也死。不报合计 R。熔断关。年化 / Calmar / 胜率 / MaxDD / Sharpe 只在以后写成交易合同时才填。

### 数据范围

品种：`BTCUSDT` 日线闭棒。X：Epoch 芯片销售 2022Q1–2026Q1 完整季（2022Q4 / 2026Q2 标了 incomplete，主表仍纳入，扫描会分列）。日历分段只用来分列。`recent_6m_oos` 只作参考窗，不单独宣判。

## 五格

| 格 | 内容 |
|---|---|
| 机制 | 上一完整季芯片销售环比高于扩窗中位数时，当天闭棒更常低于 MA200。 |
| 预期市况 | 开支加速的季对应 BTC 非牛；若加速发生在慢牛里，这句话失效。 |
| 合同 | 高强度时不许按牛市模板做多。不是开空。不加仓。闭棒。 |
| 证伪条件 | 高强度日非牛比例 ≤ 低强度日，或任一分列段反向。 |
| 落地 | 人手同一句。机器：本目录 Phase 1 扫描；还没有 2h 进出 YAML。 |

## 谱系

不是 `20260914_ai_financing_btc`（那句是公告后做多）。不是 `20260911_btc_lead_ai_alts`。不是用三段标签当市况的答案本。公开仓库此前无已结案「芯片销售环比过高 → BTC 闭棒非牛」句。

## Phase 1 探照灯（不能结案）

Y = BTC 日线收盘 &lt; SMA200。X = 芯片销售环比高于扩窗中位数。产物：`quick_scan/chip_spend_regime.json`。

| 范围 | 高强度非牛 | 低强度非牛 | 差（高−低） | n 高 / n 低 |
|---|---:|---:|---:|---:|
| 合并 | 13.4% | 50.5% | −37.1pp | 546 / 610 |
| bear_2022 | 27.7% | 无样本 | — | 213 / 0 |
| bull_2023_2024 | 14.9% | 39.7% | −24.8pp | 396 / 184 |
| recent_range_to_bear | 15.6% | 55.1% | −39.5pp | 90 / 425 |

方向和纸上声称相反：开支环比高的日子，闭棒更常在 MA200 上。2023 年芯片账单翻倍发生在 BTC 慢牛里。判决仍空着。
