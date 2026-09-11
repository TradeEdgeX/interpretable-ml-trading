---
topic: "20260911_ashare_monday_rebound"
strategy: "ashare_monday"
harness: event_backtest
segments: [bear_2021, bull_924, chop_recent]
kill_switch: false
verdict:
kpi: {}
supersedes: []
tags: [ashare, calendar, monday, rebound]
---

# DECISION — 20260911_ashare_monday_rebound

## 原句

A 股周一跌了，后四个交易日会涨。测一下。

## 现象解剖

### 社会学

周末坏消息在周一开盘被定价；恐慌盘付钱。若后四天只是情绪出清，接盘的人收复。付费方是周一的恐慌盘，不是「市场」。

### 数学

日线收益。`monday_down`（周一收盘可知）。持有 4 个交易日。闭棒：周一收盘才知道跌了，最早周二开盘进。

### 统计学

先当日历 alpha 声称。抽样用 `market_segment_ashare.yaml`：`bear_2021` / `bull_924` / `chop_recent`，**不**套币圈三段。

## 验证合同

### 验证标准

任一段年化 < 0，或近窗回撤深于两段趋势。只用年化 / Calmar / 胜率 / MaxDD / Sharpe。熔断关。

### 数据范围

`000300.SH` 日线 · `market_segment_ashare.yaml` 三段 · 特征库 `features_ashare_monday_1D`。

## 五格

| 格 | 内容 |
|---|---|
| 机制 | 周一收盘下跌（`monday_down=1`）后做多指数；持有 4 根日线时间出场。 |
| 预期市况 | 情绪出清后收复；三段都不该把账户做穿。 |
| 合同 | 周二开盘进、四根日线时间出场；不加仓、不摊。闭棒。 |
| 证伪条件 | 任一段年化 < 0，或近窗回撤深于两段趋势。 |
| 落地 | 机器：本目录 `strategies/ashare_monday`。人手同一句。 |

## 谱系

公开仓库此前无已结案 A 股周一反弹。

## 分段结果（000300.SH · 日线 · 熔断关 · 特征库 `features_ashare_monday_1D`）

| 段 | 年化 | Calmar | 胜率 | 最大回撤 | Sharpe(R) | 笔数 |
|----|-----:|-------:|---:|------:|-------:|---:|
| bear_2021 | +0.61% | 1.16 | 52.9% | −0.53% | 0.12 | 34 |
| bull_924 | +0.46% | 1.52 | 50.0% | −0.30% | 0.12 | 16 |
| chop_recent | +0.06% | 0.14 | 58.8% | −0.45% | 0.03 | 17 |

三段年化均为正，但近窗 Calmar 明显弱于两段趋势。判决仍留空，等人 `--declare`。

产物：`results/ashare_monday/experiments/20260911_ashare_monday_rebound/monday_down/bear_2021`
产物：`results/ashare_monday/experiments/20260911_ashare_monday_rebound/monday_down/bull_924`
产物：`results/ashare_monday/experiments/20260911_ashare_monday_rebound/monday_down/chop_recent`
