---
topic: 20260911_ashare_cs_reversal
strategy: ashare_cs_reversal
harness: cs_panel
segments:
- bear_2021
- bull_924
- chop_recent
kill_switch: false
verdict: reject
kpi: {}
supersedes: []
tags:
- ashare
- cross-section
- reversal
- momentum
- amount
---

# DECISION — 20260911_ashare_cs_reversal

## 原句

那就当负因子反过来买试试看。

（续涨那张纸翻面当过关 —— **不是**这一句。因子仍是 `mom_20` + `amount_z_20`，书改买分低的。）

## 现象解剖

### 社会学

涨得多、成交额热的票，追涨的人已经付过钱。反过来买的是被冷落的名字，付费方是还在热票里的人。这是反转，不是把续涨说圆。

### 数学

同一套闭棒列和 `score`。  
书：每个交易日买 `score` 最低的 20% 等权；另本按 `max(-score, 0)` 加权。对照仍是全市场等权。下一根开盘进。

### 统计学

抽样声称：冷票相对等权，20 日条件期望更好。  
统计先当 **反转 / 拥挤回吐**，不是选股 alpha。IC 探照灯已是负号，仍不能结案。

## 验证合同

### 验证标准

`bear_2021` / `bull_924` / `chop_recent` 任一段：低分组相对等权年化 ≤ 0，这句话就死。  
只用年化 / Calmar / 胜率 / MaxDD / Sharpe。熔断关。

### 数据范围

全部 A 股日线 · `cs_panel` · `market_segment_ashare.yaml` 三段。

## 五格

| 格 | 内容 |
|---|---|
| 机制 | 收盘 `score` 后 20% 做多；等权。 |
| 预期市况 | 热票拥挤后回吐；牛段不该相对等权更差。 |
| 合同 | 下一根开盘进；一天一记；不空、不加。闭棒。 |
| 证伪条件 | 任一段相对等权年化 ≤ 0。 |
| 落地 | 人手：买最冷的一档，不买最热的。 |

## 谱系

续涨句 `20260911_ashare_cs_mom_amount` 已量、相对等权失败。公开仓库无已结案反转句。

## 分段结果（cs_panel · 低分 20% vs 等权 · 熔断关）

| 组 | 段 | 年化 | Calmar | 胜率 | 最大回撤 | Sharpe |
|---|---|-----:|-------:|---:|------:|-------:|
| long_low | bear_2021 | −4.89% | −0.14 | 56.3% | −35.00% | −0.08 |
| ew | bear_2021 | +4.70% | 0.16 | 57.3% | −29.69% | 0.32 |
| long_low | bull_924 | +108.34% | 4.82 | 58.3% | −22.50% | 1.94 |
| ew | bull_924 | +76.57% | 4.47 | 53.4% | −17.15% | 1.57 |
| long_low | chop_recent | +22.28% | 0.76 | 55.1% | −29.12% | 1.00 |
| ew | chop_recent | +22.75% | 1.04 | 58.7% | −21.79% | 1.15 |
| inv_score_wt | bear_2021 | −1.64% | −0.05 | 57.0% | −33.44% | 0.05 |
| inv_score_wt | bull_924 | +106.22% | 5.03 | 58.3% | −21.10% | 1.94 |
| inv_score_wt | chop_recent | +22.71% | 0.82 | 56.1% | −27.63% | 1.06 |

924 牛低分组跑赢等权。熊段相对等权 ≤ 0，近窗也略差。证伪条件已触发。判决留空。

产物：`results/ashare_cs_reversal/experiments/20260911_ashare_cs_reversal/`

## Promote

- [ ] `mlbot research validate 20260911_ashare_cs_reversal` 通过
- [ ] `mlbot research close` 后 `--declare`
