---
topic: "20260911_btc_lead_ai_alts"
strategy: "btc_lead_alts"
harness: event_backtest
segments: [bear_2022, bull_2023_2024, recent_range_to_bear]
kill_switch: false
verdict:
kpi: {}
supersedes: []
tags: [btc-lead, ai-alts, beta]
---

# DECISION — 20260911_btc_lead_ai_alts

## 原句

BTC 大涨之后，AI 山寨会跟涨。测一下。

## 现象解剖

### 社会学

BTC 先定价风险偏好，散户/主题盘后买叙事币，付钱给已在场的人。

### 数学

闭棒 BTC 前一根 2h 收益（山寨棒上对齐的 FeatureStore 列 `btc_prior_bar_return`，不走 `btc_roc90` 注入）。阈值 +3%/2h。山寨持有 6 根 2h。

### 统计学

山寨对 BTC 的 **beta**，不是选点 alpha。抽样 `bear_2022` / `bull_2023_2024` / `recent_range_to_bear`。

## 验证合同

### 验证标准

任一段年化 < 0，或近窗回撤深于两段趋势。年化 / Calmar / 胜率 / MaxDD / Sharpe。熔断关。

### 数据范围

`NEARUSDT` / `FETUSDT` / `RENDERUSDT` · 2h · 上述三段 · 层 `features_btc_lead_alts_120T`。`TAOUSDT` 若缺熊段上市则不作主证。

## 五格

| 格 | 内容 |
|---|---|
| 机制 | BTC 前一根闭棒收益 ≥ +3% 做多 AI 山寨。 |
| 预期市况 | 风险偏好传导；三段不该做穿。 |
| 合同 | 时间出场 6 根 2h；不加仓；熔断关。闭棒。 |
| 证伪条件 | 任一段年化 < 0，或近窗回撤深于趋势段。 |
| 落地 | 机器：本目录 `strategies/btc_lead_alts`。人手同一句。 |

## 谱系

公开仓库此前无已结案 BTC→AI 山寨领涨句。

## 分段结果（NEAR/FET/RENDER · 2h · 熔断关 · 层 `features_btc_lead_alts_120T`）

| 段 | 年化 | Calmar | 胜率 | 最大回撤 | Sharpe(R) | 笔数 |
|----|-----:|-------:|---:|------:|-------:|---:|
| bear_2022 | 无样本 | — | — | — | — | 0 |
| bull_2023_2024 | +2.81% | 2.29 | 50.0% | −1.22% | 0.26 | 46 |
| recent_range_to_bear | −1.52% | −0.54 | 30.8% | −2.82% | −0.24 | 26 |

`bear_2022`：本机 tick 自 2023-01 起，无上市样本，不能靠近窗单独 promote。近窗年化为负且回撤深于牛段。判决仍留空。

产物：`results/btc_lead_alts/experiments/20260911_btc_lead_ai_alts/btc_lead_3pct/{bull_2023_2024,recent_range_to_bear}`
