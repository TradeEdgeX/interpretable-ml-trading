# AI 帮你量

在这个仓库里，AI 的角色是：帮你把模板翻译成程序能跑的规则、跑脚本、把表复述成文字。它不下结论。结论由你读表之后自己写。

## AI 能做什么

- 把你填的五格，写成 `config/strategies/<archetype>/` 下面的规则文件。
- 跑一根品种进出场的回测，或每天给全市场打分的脚本。
- 把三段、五项数字复述成一段文字，方便你读。

## AI 不能做什么

- 替你下结论（「这句话成立」或「这句话不成立」）。
- 改回测怎么扣手续费、怎么滑点、开盘能不能偷看当根已经走完的数字。
- 在回测脚本里现场另算特征（必须从特征库读）。

## 为什么这样分工

AI 擅长翻译和复述，但不擅长判断「这句话在最近这段亏钱，是不是因为市场变了」。这个判断需要你对策略和市况的理解，所以结论由你写。

## 还想看细则

- [跟 AI 说话](../use/talk-to-the-ai.md)
- [谁写哪一格](../tech/who-writes-what.md)
- [docs/agent/rd_playbook.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/agent/rd_playbook.md)
