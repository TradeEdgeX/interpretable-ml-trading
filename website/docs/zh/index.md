<div class="tx-hero" markdown>

<span class="tx-kicker">假设验证器</span>

# interpretable-ml-trading

你有一句交易想法，拿不准它到底有没有用。这个仓库帮你把这句话写成一份**可验证的合同**：什么时候允许做、错了怎么走、亏在哪、用哪段行情量。然后它用**你机器上下载的成交**、**按月算好的特征库**和**同一把锁死的尺子**去跑，最后给你一张表：熊市、牛市、近窗三段，各报年化、Calmar、胜率、最大回撤、Sharpe。仓库自带一套基础因子库（均线、交叉、费率、订单流等），欢迎你基于这些列提出更多可验证的想法。程序只负责出数字；你阅读报告和策略后，自己写下结论。

</div>

同一句均线金叉，问法不同，拿到的答案完全不同：

| | 网页 AI | 这里 |
|---|---|---|
| 给你什么 | 「经典趋势，震荡市假信号多」——一段无法复查的故事 | 熊 / 牛 / 近窗三张五项 KPI 表 |
| 金叉近窗年化（本机） | — | **−3.1%**（数字来源 [README](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/README_CN.md)） |

证据来自你机器上的成交和特征库，换一个对话模型，表还是这张。AI 帮你整理模板、编排测量，并用五项 KPI 复述数字。结论由你理解报告之后写下。

## 五个入口

<div class="grid cards" markdown>

-   **[量化常识](quant/what-is-a-strategy.md)**

    合同、闭棒、三段、五项 KPI、先分类

-   **[特征与因子](features/what-is-a-feature.md)**

    测量、五族、特征库、数学不作门

-   **[框架](framework/four-layers.md)**

    四层、怎么选、流程、三种评测机

-   **[技术](tech/why-local.md)**

    为何本机、目录、YAML、命令、谁写哪格

-   **[怎么用](use/talk-to-the-ai.md)**

    开口方式与安装

-   **[已量展厅](gallery/index.md)**

    按现象看已量过的句子（含 reject）

</div>

## 从哪读起

1. [策略是合同](quant/what-is-a-strategy.md) · [闭棒](quant/closed-bar.md) · [五项 KPI](quant/five-kpis.md) · [先分类](quant/classify.md)
2. [什么是特征](features/what-is-a-feature.md) · [特征五族](features/families.md)
3. [本机四层](framework/four-layers.md) · [怎么选](framework/how-to-choose.md)
4. [填模板](design/write-the-sentence.md) · [跟 AI 说话](use/talk-to-the-ai.md)
5. [数据流与目录](tech/stack.md) · [命令地图](tech/commands-map.md)

装一次：[安装与第一次](use/install-and-first-run.md) · 跳读：[词汇表](glossary.md)

仓库里的 `docs/` 给 AI 和想自己翻细则的人。本站是教学地图。
