# AI 帮你量

**一句话：** AI 在这个仓库里的角色是「帮你把模板翻译成 YAML、跑脚本、把表复述成文字」。它不下结论，结论由你读表后自己写。

## AI 能做什么

- 把你填的五格翻译成 `config/strategies/<archetype>/*.yaml`。
- 跑 `scripts/event_backtest.py` 或横截面脚本。
- 把三段五项 KPI 表复述成一段文字，方便你读。

## AI 不能做什么

- 替你下结论（「这句话成立 / 不成立」）。
- 改评测机的撮合逻辑（手续费、滑点、闭棒）。
- 在回测脚本里现算特征（必须走特征库）。

## 为什么这样分工

AI 擅长翻译和复述，但不擅长判断「这句话在近窗亏钱是不是因为市场变了」。这个判断需要你对策略和市况的理解，所以结论由你写。

## 还想看细则

- [跟 AI 说话](../use/talk-to-the-ai.md)
- [谁写哪格](../tech/who-writes-what.md)
- [docs/agent/rd_playbook.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/agent/rd_playbook.md)
