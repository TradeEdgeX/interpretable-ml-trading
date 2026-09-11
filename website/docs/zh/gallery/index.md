# 已量展厅

按现象分组，不按日期。数字来自本机法庭（[framework §4](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/framework.md) 与各例子原文）；**判决由人写**。失败案子照实写，不是策略推荐。

每张卡片：人话 → 分类 → 粒度 / 日历 → 特征族 → 三段年化 → 原文链接。

---

## 趋势 / beta

### 均线金叉

- **人话：** 价在 50 上方且刚上穿 200 做多，跌破 50 走。
- **分类：** beta / 趋势暴露
- **粒度 / 日历：** 成交 → 2h · 币圈三段
- **特征族：** 收盘技术量（EMA 交叉）
- **年化：** 熊 +3.7% / 牛 +3.0% / 近 **−3.1%**
- [README 完整例子](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/README_CN.md)

---

## 拥挤回吐

### 资金费率极端就反手

- **人话：** 费率 z 到 ±1.5 反手，回到 0 走。
- **分类：** 拥挤回吐（偏均值修复）
- **粒度 / 日历：** 费率序列 · 币圈三段
- **特征族：** 拥挤与持仓（`funding_rate_zscore_50`）
- **年化：** +2.7% / +3.0% / **−3.4%**
- [原文](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260910_funding_fade_CN.md)

---

## 日历

### A 股周一跌、后四天涨

- **人话：** 周一收跌后做多沪深 300，持有四个交易日。
- **分类：** 日历 alpha（很弱）
- **粒度 / 日历：** A 股日线 · A 股三段
- **特征族：** 收盘技术量（星期 / `monday_down`）
- **年化：** +0.61% / +0.46% / +0.06%
- [原文](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260911_ashare_monday_rebound_CN.md)

---

## 跨品种 beta

### BTC 大涨后 AI 山寨跟涨

- **人话：** BTC 上一根大涨后，做多主题山寨持有几根 2h。
- **分类：** beta（山寨对 BTC）
- **粒度 / 日历：** 成交 → 2h · 币圈三段；**2022 熊无样本**
- **特征族：** 收盘技术量（宿主棒上的 BTC 列）
- **年化：** 无样本 / +2.81% / **−1.52%**
- [原文](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260911_btc_lead_ai_alts_CN.md)

---

## 肥尾右尾

### P99 大单 + 布林上轨追涨

- **人话：** 根内成交额到 P99 且价在布林上沿，追涨。
- **分类：** 动量 / 肥尾右尾
- **粒度 / 日历：** **tick** · 币圈三段
- **特征族：** 订单流 + 收盘技术量（P99 + `bb_position`）
- **年化：** +0.24% / **−0.25%** / +0.28%
- [原文](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260911_p99_bb_break_chase_CN.md)

---

## 横截面（能力说明 · 已 reject）

框架能跑每天对一篮子打分；**能力不是 edge**。见 [三种评测机](../framework/which-court.md) · [cs_panel](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/cs_panel_CN.md)。

### 涨得多又热 → 相对等权继续涨？

- **分类：** 先当 beta；相对等权失败
- **特征族：** 横截面列（`mom_20` + `amount_z_20`）
- **结果：** 三段高分组都跑不赢等权；已 reject
- [原文](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260911_ashare_cs_mom_amount_CN.md)

### 热板块相对等权？

- **分类：** 板块轮动 / beta；相对等权无稳定超额
- **特征族：** 横截面列（板块均值）
- **年化（扣 10bp）：** 等权 +6.4 / **+74.6** / **+30.5**；热板块 +9.4 / +34.1 / +5.8 — 牛与震荡输对照；已 reject
- [原文](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260911_ashare_cs_sector_cost_CN.md)

---

## 不能量的故事

### 「下一家十倍股是谁」

没有成交、没有闭棒列 → **不能**当假设。只能量已发生的规则（例如入场日小市值、固定拿满年数）；那句 cohort 已 reject。

- [原文](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260911_tenbagger_smallcap_CN.md)

---

读表时记住：[闭棒](../quant/closed-bar.md) · [五项 KPI](../quant/five-kpis.md) · [先分类](../quant/classify.md) · [特征五族](../features/families.md)。
