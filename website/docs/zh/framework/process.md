# 流程

**一句话：** 从一句想法到一张 KPI 表，走五步：填模板 → 选评测机 → 写 YAML → 跑回测 → 读表下结论。

## 五步

1. **填模板**：把想法写进 `docs/hypothesis_template.md` 的五格（机制、预期市况、合同、证伪、落地）。
2. **选评测机**：根据「单品种 vs 一篮子」选事件回测或横截面回测。
3. **写 YAML**：把合同翻译成 `config/strategies/<archetype>/*.yaml`。
4. **跑回测**：`scripts/event_backtest.py` 或横截面脚本，印三段五项 KPI 表。
5. **读表下结论**：对照事先写的证伪线，自己写下「这句话在近窗成立 / 不成立」。

## 谁干什么

- **你**：填模板、读表、下结论。
- **AI**：帮你把模板翻译成 YAML、跑脚本、把表复述成文字。
- **程序**：算特征、撮合、印表。

## 还想看细则

- [填模板](../design/write-the-sentence.md)
- [跟 AI 说话](../use/talk-to-the-ai.md)
- [谁写哪格](../tech/who-writes-what.md)
