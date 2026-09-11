# 三种评测机

**一句话：** 句子的时钟形状决定评测机。能力不是 edge。

## 怎么选

| 句子长什么样 | 评测机 | 例子 |
|---|---|---|
| 一根品种、一条时间轴上的事件 | **事件回测** `event_backtest` | 金叉、费率 fade、周一 |
| 每天给一篮子打分，买前百分之几 | **横截面** `cs_panel` | 动量+成交额（已 reject） |
| 入场日之后拿满 N 年 | **cohort** | 入场日小市值长持 |

「下一家十倍股是谁」还没有闭棒列，**不能**当假设去量。只能量已经发生过的入场规则。

公开命令 `mlbot research run` 默认只派事件回测。横截面 / cohort 走各自脚本。

## 错 vs 对

| 错 | 对 |
|---|---|
| 框架能跑横截面 → 这招能赚钱。 | 已量横截面句是 reject——那是能力说明，不是推荐策略。 |
| 把「每天打分」塞进单品种事件回测。 | 换对的评测机，或改写成事件句。 |

## 还想看细则

- [cs_panel](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/cs_panel_CN.md)
- [十倍股例子](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260911_tenbagger_smallcap_CN.md)
