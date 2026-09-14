# 填模板

**一句话：** 把一句交易想法写成可验证的合同，从填 `docs/hypothesis_template.md` 的五格开始。已量例子都是同一条故事：网上的 AI 会讲什么 → 本仓库必须先填模板、查谱系、等人说「测一下」才下载 → 程序出表 → 人宣判。填完之后再读七段：设计、数据、特征、IC、验证、结论、报告解读。

## 五格（写进 `DECISION.md`）

| 格 | 写什么 | 例子（均线金叉） |
|---|---|---|
| 机制 | 这句话为什么能赚钱、发生了什么才允许做 | 趋势延续：收盘在 50 日线上方且刚上穿 200 日线才许开多 |
| 预期市况 | 哪段行情应该赚、哪段应该亏或失效 | 趋势段赚、震荡段回撤不该更深、年化不该变负 |
| 合同 | 进 / 向 / 出写死；加不加仓；熔断开不开 | 跌破 50 日线走；不加仓；熔断关；闭棒 |
| 证伪 | 哪段哪个五项 KPI 坏了就作废 | 任一段年化为负，或震荡段回撤深于趋势段 |
| 落地 | 用哪台评测机、机器和人手是不是同一句 | `event_backtest` + 层 `features_ma_cross_120T_<hash>`；人手同一句 |

空五格看起来像策略说明书。还要先写清这是哪种社会 / 数学 / 统计现象，以及用哪段数据判死。完整格子在 [假设模板](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/hypothesis_template.md)。

## 量完之后，报告按七段读

| 段 | 你要能回答 |
|---|---|
| 实验怎么设计的 | 评测机是 `event_backtest`、`cs_panel`、`cohort_hold` 还是 Phase 1？五格和证伪线事先写了吗？ |
| 数据 | 品种、周期、来源、分窗起止。缺年写了「无样本」吗？有没有套错市场的日历？ |
| 特征 | 列从 FeatureStore 来吗？开盘是不是只读上一根？有没有在回测里 `compute_*`？ |
| IC | 有表的话，它只是探照灯。没有表的话，是不是单品种 0/1 时钟、本来就不该有？ |
| 验证 | 对照是现金、等权，还是同窗买入持有？相对差打中证伪线了吗？ |
| 结论 | 先分类（alpha / 肥尾 / beta / 无用）。人 `--declare`，不手写 `verdict:`。 |
| 报告解读 | 年化不是 ΣR；胜率不是「跟上了牛市」；浅回撤可以是保险；近窗不能单独 promote。 |

对照一张已经写完的七段：[买 ETF 会错过美股牛市吗](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260914_eq_us_spy_qqq_beta_CN.md)。全部已量句在 [已量展厅](../gallery/index.md)。

## 填完之后

把填好的模板交给 AI，让它帮你翻译成 YAML、跑回测、印表。你读表，对照证伪线，自己写下结论。没说「测一下」之前，AI 只许验模板、查谱系、建目录，不许下载、不许回测。

## 还想看细则

- [docs/hypothesis_template.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/hypothesis_template.md)
- [docs/hypothesis.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/hypothesis.md)
- [docs/examples/](https://github.com/TradeEdgeX/interpretable-ml-trading/tree/main/docs/examples)
