# 热板块相对同宇宙等权，是不是选板块 alpha？

实验：[config/experiments/20260911_ashare_cs_sector_cost/](../../config/experiments/20260911_ashare_cs_sector_cost/)  
评测机是 **cs_panel**（`scripts/research/cs_sector.py`），不是 2h `event_backtest`。判决未写。

> 先按板块排序，再扣再平衡成本。相对同一宇宙、同一 10bp 的等权，热板块该更好。

个股两因子那张纸（[动量+成交额](20260911_ashare_cs_mom_amount_CN.md)）已经相对等权死了。这句换成板块打分，对照和成本一起改合同，不是把旧表改个名。

---

## 规则

| 格 | 内容 |
|---|---|
| 机制 | 板块成员 `mom_20` / `amount_z_20` 等权均值，板块间 z，买前 20% 板块里的名字等权。 |
| 预期市况 | 去掉小票混杂后，相对对照仍该同号。 |
| 合同 | 日频换仓；两边单边 10bp；闭棒（收盘打分，下一根开盘）。 |
| 证伪 | 任一段相对同成本等权年化 ≤ 0。 |
| 落地 | 机器和人手同一句。 |

统计先当 **板块轮动 / beta**，不是选股 alpha。行业是 `config/industry_map_ashare.yaml` 20 档粗分快照，不是申万一级、不是时点修订。

日历用 [`config/market_segment_ashare.yaml`](../../config/market_segment_ashare.yaml)。

---

## 书（熔断关 · 单边 10bp）

| 书 | bear_2021 | bull_924 | chop_recent |
|---|---:|---:|---:|
| 同宇宙等权 | +6.42% | **+74.59%** | **+30.48%** |
| 热板块前 20% | **+9.41%** | +34.05% | +5.79% |

相对对照：熊 +2.99pp；牛 **−40.54pp**；震荡 **−24.69pp**。牛和震荡毛收益也输等权。热板块日均单边换手约 0.38，费用一年大约 9–10 个点。

热板块自己三段年化都绿，**不是**这句的过关线。合同比的是相对等权。

分类：**板块轮动 / beta 声称，相对等权没有稳定超额。** 不是选板块 alpha。

```bash
PYTHONPATH=src python -m cli.main research close 20260911_ashare_cs_sector_cost
PYTHONPATH=src python -m cli.main research close 20260911_ashare_cs_sector_cost --declare reject
```

---

## 不是下一句

「没有 alpha，板块 beta、赚钱且稳定」是另一张纸：[`20260911_ashare_cs_sector_beta`](../../config/experiments/20260911_ashare_cs_sector_beta/)。周频、对照现金、MaxDD 不深于 −40%。**未量。** 不能拿上表的绝对年化事后当那句已经过关。
