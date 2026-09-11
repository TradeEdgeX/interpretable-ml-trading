# 涨得多又热的票，相对全市场继续涨？

实验：[config/experiments/20260911_ashare_cs_mom_amount/](../../config/experiments/20260911_ashare_cs_mom_amount/)  
评测机是 **cs_panel**，不是 2h `event_backtest`。判决未写。

> 过去 20 日涨得多、且自身成交额也热的股票，未来 20 日相对全市场等权继续涨。

这句是横截面两因子，不是大盘择时，也不是过夜挖 IC。

---

## 流程（框架能验这类算法的那条路）

```text
人出句（两列事先锁死）
  → 模板：社会 / 数学 / 统计 / 尺子 / A 股三段 / 五格
  → validate + 谱系
  → Phase 1：按日 Spearman IC（探照灯，不结案）
  → cs_panel：前 20% 等权书 vs 全市场等权，五项 KPI 分窗
  → 人 --declare
```

IC 涨了或跌了都不能改题、不能当判决。

---

## 规则

| 格 | 内容 |
|---|---|
| 机制 | 收盘 `score = 0.5 * cs_z(mom_20) + 0.5 * cs_z(amount_z_20)`，买前 20%。 |
| 预期市况 | 趋势段动量该亮；熊段不该相对等权更差。 |
| 合同 | 下一根开盘进；一天一记；不空、不加。闭棒。 |
| 证伪 | 任一段相对等权年化 ≤ 0。 |
| 落地 | 机器和人手同一句。 |

统计先当 **beta**（动量 + 活跃度），不是选股 alpha。未做行业 / 市值中性。

日历用 [`config/market_segment_ashare.yaml`](../../config/market_segment_ashare.yaml)。

---

## 查过没有

```bash
PYTHONPATH=src python -m cli.main research index --trusted --query cross-section
PYTHONPATH=src python -m cli.main research validate 20260911_ashare_cs_mom_amount
```

公开仓库当时 **0** 条已结案同一句。模板过。

---

## 量

```bash
PYTHONPATH=src python scripts/research/cs_panel.py
```

宇宙：`type=1` 非 ST 5228，日线齐 5175。`mlbot research run` 不跑这个家族（它只派 `event_backtest`）。

---

## IC（探照灯）

| 列 | bear_2021 | bull_924 | chop_recent |
|---|---:|---:|---:|
| mom_20 | −0.071 | −0.121 | −0.056 |
| amount_z_20 | −0.017 | −0.064 | −0.037 |
| score | −0.053 | −0.110 | −0.054 |

三段都是负号：高分对未来 20 日开盘收益是反转，不是续涨。

---

## 书（熔断关）

| 组 | 段 | 年化 | Calmar | 胜率 | 最大回撤 | Sharpe |
|---|---|---:|---:|---:|---:|---:|
| long_top | bear_2021 | −8.65% | −0.28 | 51.7% | −30.36% | −0.27 |
| ew | bear_2021 | +4.70% | 0.16 | 57.3% | −29.69% | 0.32 |
| long_top | bull_924 | +24.13% | 0.93 | 49.1% | −25.99% | 0.69 |
| ew | bull_924 | +76.57% | 4.47 | 53.4% | −17.15% | 1.57 |
| long_top | chop_recent | +13.84% | 0.71 | 55.1% | −19.45% | 0.68 |
| ew | chop_recent | +22.75% | 1.04 | 58.7% | −21.79% | 1.15 |

三段高分组都跑不赢等权。`bull_924` 的 +76.57% 是等权对照（全市场），高分组同一段只有 +24.13%。

按分数加权更差：`score_wt` / `top_sw` 在 924 牛是 +7.63% / +1.52%。IC 为负时，分数加得越重，越亏相对收益。

分类：**这句话是 beta 续涨声称，量出来是无用（符号是反转）。** 反转要另写一句，不能把这一张纸翻面当策略。

```bash
PYTHONPATH=src python -m cli.main research close 20260911_ashare_cs_mom_amount
PYTHONPATH=src python -m cli.main research close 20260911_ashare_cs_mom_amount --declare reject
```
