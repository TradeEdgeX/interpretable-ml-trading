---
topic: "20260910_ma50_ma200_cross"
strategy: "ma_cross"
harness: event_backtest
segments: [bear_2022, bull_2023_2024, recent_range_to_bear]
kill_switch: false
verdict:
kpi: {}
supersedes: []
tags: [ma-cross, ema-cross, golden-cross]
---

# DECISION — 20260910_ma50_ma200_cross

人说的原句：价格在 50 日线上，金叉 200 日线就做多，跌破 50 就走。  
预期：趋势年能赚钱，震荡年不该亏太多。

本机量的是 **2 小时棒上的 EMA50 / EMA200**（不是网页上那张日线图），出场 100% 是收盘跌破 EMA50。

## 五格

| 格 | 内容 |
|---|---|
| 机制 | 收盘刚上穿：EMA50 上穿 EMA200 开多；下穿开空。 |
| 预期市况 | 趋势年赚钱；震荡年回撤不该更深，年化不该变负。 |
| 合同 | 收盘跌破 EMA50 就走（空头对称）。不加仓、不摊、不跟踪止盈。 |
| 证伪条件 | 震荡段回撤深于趋势段，或任一段年化为负。 |
| 落地 | 机器：本目录 `strategies/ma_cross`。人手同一句。 |

## 谱系

公开仓库此前 0 条已结案金叉。

## 分段结果（BTCUSDT · 2 小时 · 熔断关 · 特征库 `features_ma_cross_120T_8610a95efc`）

| 段 | 年化 | Calmar | 胜率 | 最大回撤 | Sharpe(R) | 笔数 |
|----|-----:|-------:|---:|------:|-------:|---:|
| bear_2022 | +3.7% | 1.51 | 38.3% | −2.4% | 0.15 | 47 |
| bull_2023_2024 | +3.0% | 0.97 | 33.3% | −3.1% | 0.15 | 33 |
| recent_range_to_bear | −3.1% | −0.68 | 24.3% | −4.5% | −0.36 | 37 |

近窗年化为负。按写好的证伪线，这句话已经被这段数字打中。判决仍留空，等人 `--declare`。

产物：`results/ma_cross/experiments/20260910_ma50_ma200_cross/ema50_200_cross/bear_2022`
产物：`results/ma_cross/experiments/20260910_ma50_ma200_cross/ema50_200_cross/bull_2023_2024`
产物：`results/ma_cross/experiments/20260910_ma50_ma200_cross/ema50_200_cross/recent_range_to_bear`
