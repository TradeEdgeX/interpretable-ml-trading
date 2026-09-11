---
topic: 20260911_ashare_cs_sector_beta
strategy: ashare_cs_sector_beta
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
- sector
- beta
- weekly
- rebalance
---

# DECISION — 20260911_ashare_cs_sector_beta

## 原句

没有 alpha，找到板块 beta 也行，反正能赚钱，而且稳定即可。

（把上一本日频热板块多头的绝对年化事后改成过关、对照改成 000300、或把「相对等权」那张纸改尺子 —— **不是**这一句。上一本合同是相对同成本等权，牛/震荡已死。）

## 现象解剖

### 社会学

板块轮动是一群人同时换座位。付费方是还留在冷板块、或追着日频热度交手续费的人。  
beta 的钱来自「待在热的那一侧」，不是从同宇宙等权里抠超额。日换三成仓，是把暴露税给券商，不是在拿 beta。

### 数学

行业列：`config/industry_map_ashare.yaml` 20 档粗分快照，不是申万一级、不是时点修订。  
板块分事先锁死，与上一本相同：成员闭棒 `mom_20` / `amount_z_20` 等权均值，板块间 z，`score = 0.5 * z_mom + 0.5 * z_amount`。  
**书是多头暴露，不是残差：** 买分最高的 20% 板块里的名字等权，不空。  
**时钟锁周频，不是日频：** 每周最后一个交易日收盘打分，下一根开盘换仓，一周内不调。闭棒。  
再平衡日扣单边 10bp；非换仓日换手记 0。  
对照是现金 / 0。不许用同宇宙等权当对照（那是 alpha 尺子），也不许改成 000300。

### 统计学

抽样声称：周频拿热板块暴露，三段绝对年化同号为正，回撤可接受。  
统计先当 **beta / 板块暴露**。不要用相对等权、不要用去 Top-3 来杀这本袖套。  
上一本日频表的绝对年化已经看过，不结本案。

## 验证合同

### 验证标准

`bear_2021` / `bull_924` / `chop_recent` 任一段：年化 ≤ 0，或 MaxDD 深于 −40%，这句话就死。  
稳定 = 三段同号为正，且没有一段回撤破 −40%（事先锁死，不扫 Calmar）。  
只用年化 / Calmar / 胜率 / MaxDD / Sharpe。熔断关。不要合计 R。

### 数据范围

全部 A 股日线 + `config/industry_map_ashare.yaml`。周期 1D，再平衡周频。  
日历：`market_segment_ashare.yaml` 的 `bear_2021` / `bull_924` / `chop_recent`。  
评测机 **cs_panel**。已量。

## 五格

| 格 | 内容 |
|---|---|
| 机制 | 周五（当周最后交易日）收盘看热板块前 20%，下周开盘换进去，拿一周。 |
| 预期市况 | 三段都该拿到正的板块暴露；只有牛段亮 = 不稳定。 |
| 合同 | 周频换仓；多头满仓等权；单边 10bp；对照现金；闭棒。 |
| 证伪条件 | 任一段年化 ≤ 0，或 MaxDD 深于 −40%。 |
| 落地 | 人手：每周看一次热板块，换完拿一周，不天天调。 |

## 谱系

`20260911_ashare_cs_sector_cost` 已量、日频、相对等权死。  
`20260911_ashare_cs_sector_alpha` 是多空残差，已量、牛/震荡死。不是这一句。

## 分段结果

人说了测。书：周频热板块前 20% 多头，再平衡日 10bp，对照现金。

| 书 | bear_2021 | bull_924 | chop_recent |
|---|---:|---:|---:|
| 周频热板块（10bp） | +12.17% | +59.81% | +3.53% |
| MaxDD | −23.65% | −19.98% | −25.02% |
| 毛收益 | +15.60% | +65.69% | +7.04% |

**类：beta / 板块暴露。** 对现金：三段年化都正，MaxDD 未破 −40%。对同宇宙等权：牛 −14.8pp、震荡 −27.0pp。人看完「并不比拿着全部不动更好」，`--declare reject`。

## Promote

- [x] 人说了测，周频书已跑
- [x] `mlbot research close` 后 `--declare reject`
