---
topic: "20260914_ai_financing_btc"
strategy: "ai_financing_btc"
harness: event_backtest
segments: [bear_2022, bull_2023_2024, recent_range_to_bear]
kill_switch: false
verdict:
kpi: {}
supersedes: []
tags: [ai-financing, funding, btc, beta]
---

# DECISION — 20260914_ai_financing_btc

## 原句

AI的融资和BTC的资金和涨跌之间有某种关系。

## 现象解剖

### 社会学

付钱的是后知后觉的风险偏好盘：公开的大型 AI 实验室融资公告（OpenAI / Anthropic / xAI / Inflection）是协调仪式，告诉权益和加密两边的杠杆盘「科技风险偏好还在」。已在场的 BTC 多头收后知后觉者的资金费率；若公告其实是抽走现金去买算力，付钱的会变成 BTC 多头自己。这不是有效市场口号。

### 数学

测量对象是锁定日历 `config/research/ai_financing_events.yaml`（公开大额轮，默认 ≥ 3 亿美元）。公告日 D（UTC）收盘后才可用。FeatureStore 列：`ai_financing_event`（D+1 第一根 2h 棒为 1）、`ai_financing_in_window`（D+1 起 5 个 UTC 日）、`ai_financing_days_since`、`ai_financing_log_usd`。BTC 资金费率用已有列 `funding_rate` / `funding_rate_zscore_50`。开盘决策只读上一根已收盘。AI 叙事币篮子资金费率 `ai_basket_funding_zscore` 只作 Phase 1 探照灯，不改这一句。

### 统计学

先当 **beta**：AI 融资公告上的风险偏好外溢，不是选点 alpha。抽样声称是公告日后 5 个 UTC 日 BTC 闭棒收益条件期望为正，且资金费率相对段内基线更偏多。三段分列：`bear_2022` / `bull_2023_2024` / `recent_range_to_bear`。

## 验证合同

### 验证标准

任一段年化 < 0，或近窗回撤深于两个趋势段，这句话死。只报年化 / Calmar / 胜率 / MaxDD / Sharpe。熔断关。不要合计 R。Phase 1 的 IC / 事件窗均值不能结案。

### 数据范围

品种：`BTCUSDT`（主证）。周期：2 小时法院；Phase 1 事件窗用日线对齐公告日。日历：`bear_2022` / `bull_2023_2024` / `recent_range_to_bear`（`config/market_segment.yaml`）。`recent_6m_oos` 只作参考窗，不单独宣判。费率：`mlbot data download-funding-rate`。价格：`data/parquet_data` 或 Phase 1 用 Binance Vision 日线，不写进 tick parquet。

## 五格

| 格 | 内容 |
|---|---|
| 机制 | 公开 AI 大额融资公告日 D 结束后，D+1 第一根已收盘许可出现时做多 BTC。 |
| 预期市况 | 风险偏好外溢年该赚；抽流动性年该失效。三段不该做穿。 |
| 合同 | 持有 5 个 UTC 日（60 根 2h）；不加仓；不摊；熔断关；闭棒。 |
| 证伪条件 | 任一段年化 < 0，或近窗回撤深于两个趋势段。 |
| 落地 | 机器：本目录 `strategies/ai_financing_btc`。人手同一句：公告次日做多 BTC，五日后走。 |

## 谱系

不是 `20260911_btc_lead_ai_alts`（那句是 BTC 领涨 AI 山寨）。不是 `20260910_funding_fade`（那句是费率极端反手）。公开仓库此前无已结案「AI 公司融资公告 → BTC 资金费 / 收益」句。

## Phase 1 探照灯（日线事件窗，不能结案）

日历 21 笔；闭日历 D+1→D+5。窗口虚拟变量对次日收益 Spearman IC = −0.021（p = 0.40，n = 1611）。

| 段 | 事件数 | 事件窗均值 | 中位数 | 日胜率 | 段内任意 5 日基线 | 费率前 / 后（8h 均值） |
|----|-----:|---------:|------:|-----:|----------------:|----------------------:|
| bear_2022 | 7 | +1.12% | +0.97% | 71% | +0.01% | 6.3e-5 / 6.8e-5 |
| bull_2023_2024 | 9 | +0.04% | −0.24% | 44% | +1.24% | 9.4e-5 / 11.7e-5 |
| recent_range_to_bear | 7 | +0.25% | −0.76% | 14% | −0.15% | 2.2e-5 / 2.4e-5 |

`bear_2022` 与 `bull_2023_2024` 日历重叠 2023-06→2023-11，Inflection 2023-06-29 与 Anthropic 2023-09-25 两笔进了两段。近窗均值被 2026-02-27 OpenAI 一笔 +8.5% 拉开。AI 篮子费率 z 滞后 1 对 BTC 费率 z：IC = 0.098（p = 0.003）；对 BTC 次日收益 IC = 0.022（p = 0.43）。产物：`quick_scan/ai_financing_scan.json`。

## 分段结果（canonical 三段，kill switch OFF）

2h `event_backtest` 尚未跑。下表留空，等 `mlbot research run`。

| 段 | 年化 | Calmar | WR | MaxDD | Sharpe |
|----|-----:|-------:|---:|------:|-------:|
| bear_2022 | | | | | |
| bull_2023_2024 | | | | | |
| recent_range_to_bear | | | | | |

## Promote

- [x] `mlbot research validate 20260914_ai_financing_btc` 通过
- [ ] 三条杠（LAYER_PROMOTION_CRITERIA §1）
- [ ] canonical 三段齐 + kill switch OFF
- [ ] 对口 harness（`event_backtest` / `ai_financing_btc`）
- [ ] `mlbot research close 20260914_ai_financing_btc` 后 `--declare`（禁止手写 verdict）
- [ ] `mlbot research index` 通过
