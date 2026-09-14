# 已量展厅

**一句话：** 这里按现象列出已经量过的句子。每张卡片是同一条故事换一句台词：网上的 AI 会讲什么、本仓库必须走哪几步、表上写了什么、人怎么宣判。包括已被判 reject 的，以及纸面已出表、判决还留给人 `--declare` 的。

细则在仓库的 `docs/examples/`。每篇先讲「这一句在本仓库里怎么走完」，再按 **设计 / 数据 / 特征 / IC / 验证 / 结论 / 报告解读** 七段展开。展厅只放对照表，不代替那篇故事。哲学：[docs/philosophy.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/philosophy.md)。金叉第一幕：[README_CN.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/README_CN.md)。

## 趋势

### 均线金叉（BTC · 2h）

过程全文在仓库根目录 [README_CN.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/README_CN.md)。

| 格 | 内容 |
|---|---|
| 句子 | 收盘在 50 日线上方且刚上穿 200 日线开多；跌破 50 日线走。 |
| 分类 | beta / 趋势暴露 |
| 评测机 | `event_backtest` · `BTCUSDT` · 2 小时 |
| 数据 | 币安成交 → parquet；特征库 `ema_50_200_cross_*` |
| IC | 没有。单品种 0/1 时钟。 |
| 结论 | 近窗年化为负，对照证伪线，这句话在近窗不成立。判决留给人 `--declare`。 |

| 段 | 年化 | Calmar | 胜率 | MaxDD | Sharpe(R) | 笔数 |
|---|---:|---:|---:|---:|---:|---:|
| 熊市 2022 | +3.7% | 1.51 | 38.3% | −2.4% | 0.15 | 47 |
| 牛市 2023–2024 | +3.0% | 0.97 | 33.3% | −3.1% | 0.15 | 33 |
| 近窗震荡到熊 | **−3.1%** | −0.68 | 24.3% | −4.5% | −0.36 | 37 |

## 费率

### 费率反转（funding fade）

过程：[docs/examples/20260910_funding_fade_CN.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260910_funding_fade_CN.md)

| 格 | 内容 |
|---|---|
| 句子 | 资金费率 50 次观察的 z 分数到 ±1.5 反手；z 回到 0 就走。 |
| 分类 | 拥挤回吐 / 偏均值修复 |
| 评测机 | `event_backtest` · `BTCUSDT` · 2 小时 |
| 数据 | 成交 + `mlbot data download-funding-rate`；列 `funding_rate_zscore_50` |
| IC | 没有。单品种 0/1 时钟。 |
| 结论 | 近窗年化 −3.4%，回撤 −6.9% 深于两段趋势。两条证伪线都打中。判决留给人 `--declare`。 |

| 段 | 年化 | Calmar | 胜率 | MaxDD | Sharpe(R) | 笔数 |
|---|---:|---:|---:|---:|---:|---:|
| 熊市 2022 | +2.7% | 0.93 | 47.7% | −2.9% | 0.08 | 128 |
| 牛市 2023–2024 | +3.0% | 0.60 | 50.7% | −4.9% | 0.13 | 75 |
| 近窗震荡到熊 | **−3.4%** | −0.50 | 45.9% | **−6.9%** | −0.11 | 109 |

## 日历

### 周一反弹

过程：[docs/examples/20260911_ashare_monday_rebound_CN.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260911_ashare_monday_rebound_CN.md)

| 格 | 内容 |
|---|---|
| 句子 | **周一收盘下跌**后做多沪深300，持有 **4 根日线**。最早周二开盘进。不是「周一开盘买、周五收盘卖」。 |
| 分类 | 日历 alpha 声称 |
| 评测机 | `event_backtest` · `000300.SH` · 日线 |
| 数据 | `mlbot data download-ashare`；A 股日历 `bear_2021` / `bull_924` / `chop_recent` |
| IC | 没有。日历 0/1 许可。 |
| 结论 | 三段年化都是很小的正数；近窗 +0.06%、Calmar 0.14。日历 alpha 很弱。判决留给人 `--declare`。 |

| 段 | 年化 | Calmar | 胜率 | MaxDD | Sharpe(R) | 笔数 |
|---|---:|---:|---:|---:|---:|---:|
| `bear_2021` | +0.61% | 1.16 | 52.9% | −0.53% | 0.12 | 34 |
| `bull_924` | +0.46% | 1.52 | 50.0% | −0.30% | 0.12 | 16 |
| `chop_recent` | +0.06% | 0.14 | 58.8% | −0.45% | 0.03 | 17 |

## 跨品种

### BTC 领涨 AI 山寨

过程：[docs/examples/20260911_btc_lead_ai_alts_CN.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260911_btc_lead_ai_alts_CN.md)

| 格 | 内容 |
|---|---|
| 句子 | BTC 前一根闭棒 2h 收益 ≥ +3% 做多 NEAR / FET / RENDER，持有 6 根。 |
| 分类 | beta（山寨对 BTC） |
| 评测机 | `event_backtest` · 2 小时 |
| 数据 | 山寨 tick 约从 2023-01 起；`bear_2022` **无样本** |
| IC | 没有。0/1 许可。 |
| 结论 | 近窗年化 −1.52%，回撤深于牛段。缺年的段不能靠近窗单独 promote。判决留给人 `--declare`。 |

| 段 | 年化 | Calmar | 胜率 | MaxDD | Sharpe(R) | 笔数 |
|---|---:|---:|---:|---:|---:|---:|
| `bear_2022` | 无样本 | — | — | — | — | 0 |
| `bull_2023_2024` | +2.81% | 2.29 | 50.0% | −1.22% | 0.26 | 46 |
| 近窗 | **−1.52%** | −0.54 | 30.8% | **−2.82%** | −0.24 | 26 |

## 肥尾

### P99 大单追涨

过程：[docs/examples/20260911_p99_bb_break_chase_CN.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260911_p99_bb_break_chase_CN.md)

| 格 | 内容 |
|---|---|
| 句子 | 根内 P99 大单 **且** `bb_position ≥ 1` 做多；回到带内或 12 根出场。 |
| 分类 | 动量 / 肥尾右尾 |
| 评测机 | `event_backtest` · `BTCUSDT` · 2 小时 |
| 数据 | **tick**；缺月写 NaN。日线代替不了 P99。 |
| IC | 没有。两列 0/1 时钟。 |
| 结论 | 牛段年化 −0.25%，证伪线打中。去 Top-3 只分类。判决留给人 `--declare`。 |

| 段 | 年化 | Calmar | 胜率 | MaxDD | Sharpe(R) | 笔数 |
|---|---:|---:|---:|---:|---:|---:|
| 熊市 2022 | +0.24% | 0.32 | 45.5% | −0.74% | 0.09 | 22 |
| 牛市 2023–2024 | **−0.25%** | −0.26 | 36.4% | −0.95% | −0.10 | 22 |
| 近窗 | +0.28% | 0.92 | 57.1% | −0.30% | 0.16 | 21 |

### 十倍股队列

过程：[docs/examples/20260911_tenbagger_smallcap_CN.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260911_tenbagger_smallcap_CN.md)

| 格 | 内容 |
|---|---|
| 句子 | 入场日流通市值 ≤ 100 亿，死拿 3 或 4 年，十倍是不是比大票更密。 |
| 分类 | 肥尾极薄，主形态小市值 beta |
| 评测机 | `cohort_hold`（不是 `event_backtest`） |
| 数据 | 在市 + 退市日线；市值 = 当日成交额 / 换手率；`n` = 名字 × 入场季 |
| IC | 没有。问的是队列密度，不是每日分数。 |
| 结论 | 3 年十倍率两边都是 **0.11%**。股灾 / 2018 相对年化 ≤ 0。已 `--declare reject`。不能量「未来谁会十倍」。 |

| 3 年持有 | 小票年化 | 大票年化 | 十倍率（小 vs 大） |
|---|---:|---:|---|
| `crash_2015` | −4.26% | −4.15% | 0 vs 0 |
| `bear_2018` | +2.75% | +3.31% | 0.29% vs 0.24% |
| `covid_2020` | +7.75% | +5.96% | 0 vs 0.13% |
| `bear_2021` | +9.09% | −0.03% | 0.09% vs 0.12% |
| `bull_924` / `chop_recent` | 无样本 | 无样本 | — |

## 每天给全市场打分

怎么测：[docs/cs_panel_CN.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/cs_panel_CN.md)

### CS 动量 + 成交额

过程：[docs/examples/20260911_ashare_cs_mom_amount_CN.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260911_ashare_cs_mom_amount_CN.md)

| 格 | 内容 |
|---|---|
| 句子 | `score = 0.5 cs_z(mom_20) + 0.5 cs_z(amount_z_20)`，买前 20%，相对全市场等权继续涨。 |
| 分类 | beta 续涨声称，量出来符号是反转 |
| 评测机 | `cs_panel` |
| 数据 | 非 ST 5228 / 日线齐 5175；A 股三段 |
| IC | **有，只探照灯。** `score` 三段日均 IC −0.053 / −0.110 / −0.054。 |
| 结论 | 三段都输等权。924 牛等权 +76.57%，高分只有 +24.13%。已 `--declare reject`。 |

| 组 | `bear_2021` | `bull_924` | `chop_recent` |
|---|---:|---:|---:|
| 高分 20% 年化 | −8.65% | +24.13% | +13.84% |
| 等权对照年化 | +4.70% | **+76.57%** | +22.75% |
| 相对等权 | **−13.35pp** | **−52.44pp** | **−8.91pp** |

### 热门板块

过程：[docs/examples/20260911_ashare_cs_sector_cost_CN.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260911_ashare_cs_sector_cost_CN.md)

| 格 | 内容 |
|---|---|
| 句子 | 先排板块，再扣单边 10bp。相对**同宇宙、同成本**等权，热板块该更好。 |
| 分类 | 板块轮动 / beta，不是选板块 alpha |
| 评测机 | `cs_sector.py` |
| 数据 | 20 档粗分快照 3633 只；日均换手约 0.38 |
| IC | 没有另出板块 IC。个股句同一对因子三段 IC 已是负号。 |
| 结论 | 熊相对 +2.99pp；牛 **−40.54pp**；震荡 **−24.69pp**。毛收益在牛 / 震荡也输。已 `--declare reject`。 |

| 书 | `bear_2021` | `bull_924` | `chop_recent` |
|---|---:|---:|---:|
| 同宇宙等权（10bp） | +6.42% | **+74.59%** | **+30.48%** |
| 热板块前 20%（10bp） | +9.41% | +34.05% | +5.79% |

## Beta

### 美股 SPY / QQQ 买入持有

过程：[docs/examples/20260914_eq_us_spy_qqq_beta_CN.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260914_eq_us_spy_qqq_beta_CN.md)

| 格 | 内容 |
|---|---|
| 句子 | 买 ETF 会错过美股牛市；要跟上得选股或等超跌。 |
| 分类 | beta（股权暴露），不是选股 alpha |
| 评测机 | `eq_us_daily`（不是 2h `event_backtest`） |
| 数据 | 无杠杆 SPY / QQQ 日线；锁到 2026-08-17；拆分红后收盘，不含分红再投 |
| IC | 没有。单品种 0/1 时钟。选股本机未重跑。 |
| 结论 | 四条超跌 / 均线时钟全期年化全部低于同窗买入持有。已 `--declare reject`。 |

| 标的 | 买入持有年化 | MaxDD | 同窗择时最好年化 |
|---|---:|---:|---:|
| SPY | **13.7%** | −34.1% | MA200 7.8% |
| QQQ | **20.4%** | −35.6% | MA200 16.0%（回撤更浅，年化更低，是保险） |

### AI 芯片销售环比 vs BTC 市况

过程：[docs/examples/20260914_ai_chip_spend_btc_regime_CN.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260914_ai_chip_spend_btc_regime_CN.md)

| 格 | 内容 |
|---|---|
| 句子 | 算力开支环比过高时，BTC 闭棒更常低于 MA200（不是随后几天涨跌，也不是三段标签）。 |
| 分类 | beta / 市况共存 |
| 评测机 | `phase1_scan_only` |
| 数据 | Epoch 季频芯片账单（CC BY）+ BTC 日线 |
| IC | 没有收益 IC。探照灯就是比例表。 |
| 结论 | 高强度非牛 13.4% / 低强度 50.5%（差 −37pp）。方向和纸上相反。已 `--declare reject`。 |

| 范围 | 高强度非牛 | 低强度非牛 | 差 | n 高 / n 低 |
|---|---:|---:|---:|---|
| 合并 | 13.4% | 50.5% | **−37.1pp** | 546 / 610 |
| 牛段 | 14.9% | 39.7% | −24.8pp | 396 / 184 |
| 近窗 | 15.6% | 55.1% | −39.5pp | 90 / 425 |

### AI 融资公告后做多 BTC

过程：[docs/examples/20260914_ai_financing_btc_CN.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260914_ai_financing_btc_CN.md)

| 格 | 内容 |
|---|---|
| 句子 | 公开大额 AI 融资公告日 D 结束后，次日做多 BTC，持有 5 个 UTC 日。 |
| 分类 | beta（风险偏好外溢） |
| 怎么出的表 | 日线看过了；按 2 小时进出场还没跑 |
| 数据 | 锁死的公开大额轮 21 笔；日线对齐公告日 |
| 日线数字 | 「这天是不是公告后」和次日涨跌 **−0.021**（p = 0.40）。不能下结论。 |
| 结论 | 日线上没有「该涨」。进出场成绩单还是空的。**还不能下结论。** |

| 段 | 事件 | 窗内均值 | 中位数 | 任意 5 日基线 |
|---|---:|---:|---:|---:|
| 熊 | 7 | +1.12% | +0.97% | +0.01% |
| 牛 | 9 | +0.04% | −0.24% | +1.24% |
| 近窗 | 7 | +0.25% | −0.76% | −0.15% |

## 还想看细则

- 七段怎么读：[docs/examples/20260914_eq_us_spy_qqq_beta_CN.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260914_eq_us_spy_qqq_beta_CN.md)（结构最完整的一篇，可当样板）
- 各层怎么选：[docs/framework.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/framework.md) 第 4 节
- 假设模板：[docs/hypothesis_template.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/hypothesis_template.md)
