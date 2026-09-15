# 三种回测

回测脚本是事先定好的比法，只负责印出三段、五项数字。这个仓库里常见三种比法：一根品种进出场、每天给全市场打分、多句话放在一起看。

## 三种比法

| 比法 | 量什么 | 什么时候用 |
|---|---|---|
| 一根品种进出场 | 一个名字，一根棒一根棒走 | 均线交叉、费率反手、特别大的单追涨 |
| 每天给全市场打分 | 一篮子名字，每天排名 | 买最热的股票、买最热的板块 |
| 多句话放在一起 | 几个策略、几个品种合在一起 | 看它们放在一起长什么样 |

## 「事先定好」是什么意思

手续费怎么扣、会不会滑点、开盘能不能偷看当根已经走完的数字，这些规则不随策略变。这样所有策略用同一套标准，表才能对齐。你不能为了让某句话好看，去改这些撮合规则。

## 在这个仓库里怎么用

- 一根品种进出场：`scripts/event_backtest.py`
- 每天给全市场打分：`scripts/research/cs_panel.py`
- 先给板块打分：`scripts/research/cs_sector.py`

## 还想看细则

- [docs/framework.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/framework.md)
- [买最热的股票，能跑赢「每人买一点」吗](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/cs_panel_CN.md)
