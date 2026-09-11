# 策略是合同，不是感觉

**一句话：** 策略必须写清四件事——什么时候允许做、做错了怎么走、这一笔亏在哪、什么时候算结束。只有「金叉做多」这四个字，还不算一份策略。

## 错 vs 对

| 错 | 对 |
|---|---|
| 「均线金叉就买，感觉趋势能赚钱。」 | 「收盘在 50 日线上方、且刚上穿 200 日线，才允许开多；跌破 50 日线就走；任一段年化为负，或震荡段回撤深于趋势段，这句话就作废。」 |

感觉没法复查，也没法证伪。合同可以：它把「做」和「不做」的边界写死，把「错了怎么办」写死，这样回测才能告诉你这句话在熊、牛、近窗三段上分别长什么样。

## 本仓库怎么用

均线金叉的练习句写满了这些格：机制、预期市况、合同、证伪、落地。机器 YAML 和人手清单是**同一句**。回测没有过关时，阅读报告后用同一句结论，而不是当晚另做一版「这回不一样」。

见 [五格](../design/five-boxes.md) · 仓库例子在 [README_CN.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/README_CN.md)。

## 还想看细则

- [假设模板](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/hypothesis_template.md)
- [哲学：图是合同](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/philosophy.md)
