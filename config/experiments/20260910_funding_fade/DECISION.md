---
topic: "20260910_funding_fade"
strategy: "funding_fade"
harness: event_backtest
segments: [bear_2022, bull_2023_2024, recent_range_to_bear]
kill_switch: false
verdict:
kpi: {}
supersedes: []
tags: [funding, fade, zscore]
---

# DECISION — 20260910_funding_fade

人说的原句：资金费率 50 根观察的 z 分数到 1.5 就做空，到 −1.5 就做多；z 分数回到 0 就走。  
预期：拥挤的一边该回吐；三段都不该把账户做穿。

本机量的是 **2 小时棒上的 `funding_rate_zscore_50`**（币安永续资金费率，不是网页截图）。出场约 94% 是 z 分数穿过 0；其余是宽初始止损。

过程全文：[docs/examples/20260910_funding_fade_CN.md](../../../docs/examples/20260910_funding_fade_CN.md)

## 五格

| 格 | 内容 |
|---|---|
| 机制 | 资金费率 50 次观察的稳健 z 分数 ≥ 1.5 开空；≤ −1.5 开多。方向取 z 分数的反号。 |
| 预期市况 | 费率极端拥挤的一段该回吐；熊 / 牛 / 近窗都不该把账户做穿。 |
| 合同 | z 分数回到 0 就走（多头 z≥0，空头 z≤0）。不加仓、不摊、不跟踪止盈。 |
| 证伪条件 | 任一段年化为负，或近窗回撤深于两个趋势段。 |
| 落地 | 机器：本目录 `strategies/funding_fade`。人手同一句。 |

## 谱系

公开仓库此前 0 条已结案资金费率 fade。

## 分段结果（BTCUSDT · 2 小时 · 熔断关 · 特征库 `features_funding_fade_120T`）

| 段 | 年化 | Calmar | 胜率 | 最大回撤 | Sharpe(R) | 笔数 |
|----|-----:|-------:|---:|------:|-------:|---:|
| bear_2022 | +2.7% | 0.93 | 47.7% | −2.9% | 0.08 | 128 |
| bull_2023_2024 | +3.0% | 0.60 | 50.7% | −4.9% | 0.13 | 75 |
| recent_range_to_bear | −3.4% | −0.50 | 45.9% | −6.9% | −0.11 | 109 |

近窗年化为负，且最大回撤深于两个趋势段。按写好的证伪线，这句话已经被这段数字打中。判决仍留空，等人 `--declare`。

产物：`results/funding_fade/experiments/20260910_funding_fade/funding_z15/bear_2022`
产物：`results/funding_fade/experiments/20260910_funding_fade/funding_z15/bull_2023_2024`
产物：`results/funding_fade/experiments/20260910_funding_fade/funding_z15/recent_range_to_bear`
