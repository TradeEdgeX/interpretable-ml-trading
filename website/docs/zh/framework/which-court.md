# 三种评测机

**一句话：** 评测机是锁死的回测脚本，只负责印五项 KPI 分窗表。仓库里有三种：事件回测、横截面回测、组合回测。

## 三种评测机

| 评测机 | 量什么 | 什么时候用 |
|---|---|---|
| 事件回测 | 单品种、事件轴（一根棒一根棒走） | 均线交叉、费率反转、大单追涨 |
| 横截面回测 | 一篮子、每天打分排序 | 动量排名、成交额排名 |
| 组合回测 | 多策略、多品种组合 | 看几个策略放在一起的效果 |

## 锁死意味着什么

评测机的撮合逻辑（手续费、滑点、闭棒）是锁死的，不随策略变。这样所有策略用同一把尺子，表才能对齐。你不能为了让某句话好看，去改评测机的撮合参数。

## 本仓库怎么用

- 事件回测：`scripts/event_backtest.py`
- 横截面回测：`scripts/cross_section_backtest.py`
- 组合回测：`scripts/portfolio_backtest.py`

## 还想看细则

- [docs/framework.md §4 评测机](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/framework.md)
