# 填模板

**一句话：** 把一句交易想法写成可验证的合同，从填 `docs/hypothesis_template.md` 的五格开始。

## 五格

| 格 | 写什么 | 例子（均线金叉） |
|---|---|---|
| 机制 | 这句话为什么能赚钱 | 趋势延续：涨过的继续涨 |
| 预期市况 | 哪段行情应该赚、哪段应该亏 | 趋势段赚、震荡段亏 |
| 合同 | 进 / 向 / 出写死 | 收盘在 50 日线上方且刚上穿 200 日线开多；跌破 50 日线走 |
| 证伪 | 哪段哪个 KPI 坏了就作废 | 任一段年化为负，或震荡段回撤深于趋势段 |
| 落地 | 用哪台评测机、哪个特征库 | 事件回测 + `features_ma_cross_120T_<hash>` |

## 填完之后

把填好的模板交给 AI，让它帮你翻译成 YAML、跑回测、印表。你读表，对照证伪线，自己写下结论。

## 还想看细则

- [docs/hypothesis_template.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/hypothesis_template.md)
- [docs/hypothesis.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/hypothesis.md)
