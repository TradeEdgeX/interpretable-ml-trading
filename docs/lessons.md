# 教训（一条一条，别再踩）

**主入口：** [README_CN.md](../README_CN.md) · 上一篇 [哲学](philosophy.md) · 下一篇 [数学](math.md)

规则和 skill 只链到本文，不另写第二套故事。长文在附录。

---

## 闭棒 {#closed-bar}

FeatureStore 的行索引是 bar **开盘** \(T\)，该行 OHLC / 特征用的是 \([T, T+\Delta)\)，墙钟 **\(T+\Delta\)（收盘）** 才可知。

在开盘决策时读 **同一根已完成行** = 未来数据。网格对 bar 内最高最低极敏感：同一套脚本，open-bar 年化曾印成 50–70%；闭棒后干净段约 −35%～−43%。那些数字 **作废**。

```text
错: 墙钟 T = 本根开盘 → feats[T]（已含本根全路径）
对: 墙钟 T 决策 → feats[T − Δ]（上一根已收盘）
```

标签（`forward_rr*`）可以看未来；**禁止**当入场特征。不要为了「数字好看」改回同根。

细则：[math.md](math.md) §1。规则：`.cursor/rules/backtest-no-future-data.mdc`。

---

## 熔断关着比 edge {#kill-switch}

比策略 / 参数时 **关掉账户熔断**。熔断开着，变体在不同时刻停开仓，后面样本不是同一条路，会捧出假赢家、藏掉真回撤。

比保护机制时另跑一次熔断开，那张表不拿来给 edge 排名。

`DECISION.md` 必须写明 `kill_switch: false`。准则：[LAYER_PROMOTION_CRITERIA](../config/experiments/LAYER_PROMOTION_CRITERIA.md)。

---

## 五项 KPI，不要合计 R {#kpi}

口头和结案表只用：年化 / Calmar / 胜率 / MaxDD / Sharpe。  
`pnl_r` 的分母随权益变，长持肥尾会被摊薄；用合计 R 比叠仓会误判。

| 段 | 年化 | Calmar | WR | MaxDD | Sharpe |
|---|---|---|---|---|---|
| 熊 | … | … | … | … | … |

不要出现「Total R = …」当头条。

---

## 三段，不要只看近窗 {#segments}

Promote 看 `config/market_segment.yaml` 的 **熊 / 牛 / 近窗** 三段同时不坏。  
`recent_6m_oos` 是子集，**不能单独**结案。合成一段会长平均掉坏年。

粗到能讲清的合同，跨行情同号已经够用，不必再切一块「显得科学」的神秘 holdout。开始在同一段上反复调参救曲线，才升级外推。

公开 dummy 用 B 层日历，不要和日频横截面那套日期混用。

---

## 人出句，机器不改题 {#hypothesis}

假设由人定。IC / 特征组搜索 / 自动挖因子只许当 Phase 1 探照灯。分数涨了不能结案，不能改 locked YAML。

AI 不得把你的句子偷换成附近一条现成规则。回测否了的句子，人手也不许做。

主路径：[hypothesis.md](hypothesis.md)。对照：[design/2026-08-23_human_hypothesis_vs_auto_mine_CN.md](design/2026-08-23_human_hypothesis_vs_auto_mine_CN.md)。Court vs CV：[agent/rd_qa.yaml](agent/rd_qa.yaml)。

---

## 先分类再拧旋钮 {#classify}

| 类型 | 钱从哪来 | 不要 |
|---|---|---|
| Alpha | 条件期望更好 | 用牛市右尾证明 alpha |
| 肥尾收割 | 少数极端路径 | 为提高胜率砍右尾 |
| Beta 增强 | 点名因子上的暴露 | 用短探测器「保护」长 beta |
| 无用 | 跨段≤0 | 继续扫阈值 |

去 Top-3 / 胜率是 **alpha** 的尺子。趋势 / beta / 肥尾袖套样本少时，去掉最大几笔为负是算术，不是「没 edge」。长文：[design/alpha_vs_fattail_vs_beta_CN.md](design/alpha_vs_fattail_vs_beta_CN.md) · [design/ex_top3_capacity_confound_CN.md](design/ex_top3_capacity_confound_CN.md)。

---

## 对口评测机 {#harness}

harness 是考试卷（时钟、成交、成本、宇宙），不是 Cursor 循环。  
公开 dummy：`ma_cross` → `event_backtest`。`mlbot research harness` 先查。  
`DECISION.md` 的 `harness:` 必须和登记表一致。用错卷子的数字不算结案。

本树不再用滚仓 / 网格 / A 股专用 harness 给那些已删家族排名。

---

## 特征来自 FeatureStore {#features}

回测路径禁止现场 `compute_*`。缺列先登记 `feature_dependencies.yaml` 再 backfill。  
特征是测量；进 / 止 / 加 / 出是合同，写在 YAML 或手做清单，不要叠成一个分数当更强假设。

---

## 均值回归看刺穿后收回 {#mean-reversion}

扫 RSI / 箱沿 / 偏离时：不要把 `rsi >= 80` 当天当空信号。要极值刺穿再 **收回越过阈值**，从收回那根起算前瞻。把「当天在极值」当对照，不当主结果。长文：[design/mean_reversion_event_definition_CN.md](design/mean_reversion_event_definition_CN.md)。

---

## 网格已下线 {#grid}

震荡网格靠 open-bar 数字看起来像圣杯。闭棒后为负。产品不在本树。不要再发明「同一账本兼容趋势和网格两种圣杯」。见 [RETIRED.md](RETIRED.md)。

---

主入口：[README_CN.md](../README_CN.md) · [哲学](philosophy.md) · [数学](math.md)
