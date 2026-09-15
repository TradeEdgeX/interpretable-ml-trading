<div class="tx-hero" markdown>

<span class="tx-kicker">给 AI 用的工具库</span>

# interpretable-ml-trading

你提出一句交易假设，并写清什么时候做、做错了怎么办。这个框架是给电脑上的 AI 助手用的工具库。它帮你验证这句话成不成立，免得你对自己的想法过于乐观。它会写出详细报告，帮你改策略、收策略。排除一百种不成立的假设之后，你也许能找到一句用得上的。你怎么探索的、每次实验得出什么结论，框架也会帮你保管——那才是最值钱的知识。

</div>

同一句均线金叉，问法不同，拿到的答案完全不同：

| | 网页 AI | 这里 |
|---|---|---|
| 给你什么 | 「经典趋势，震荡市假信号多」——一段无法复查的故事 | 熊市、牛市、最近这段，各一张五项数字的表 |
| 金叉近窗年化（本机） | — | **−3.1%**（数字来源 [README](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/README_CN.md)） |

证据来自你机器上的成交和事先算好的测量，换一个对话模型，表还是这张。AI 帮你整理假设、跑测量、写出报告。结论由你读完报告之后自己写。

## 入口

<div class="grid cards" markdown>

-   **[量化常识](quant/what-is-a-strategy.md)**

    策略要事先写清楚、开盘只看已经收完的数字、至少看三段日子、表上只报五项数字、先说钱从哪来

-   **[特征与因子](features/what-is-a-feature.md)**

    什么叫测量、五族各量什么、先读算好的表、数学不能改许可

-   **[框架](framework/four-layers.md)**

    四层文件、怎么选回测、五步流程、三种比法

-   **[技术](tech/why-local.md)**

    为什么要在你自己电脑上跑、四个目录、规则文件、常用命令、谁写哪一格

-   **[怎么用](use/talk-to-the-ai.md)**

    怎么对本仓库里的 AI 说话，以及怎么装一次

-   **[Q & A](qa/index.md)**

    规则会不会骗人、为什么要分三段、相关为什么不能结案

-   **[已量展厅](gallery/index.md)**

    按现象看已经量过的句子（也包括已经判定不成立的）

</div>

## 从哪读起

1. [策略要事先写清楚](quant/what-is-a-strategy.md) · [开盘只看已经收完的数字](quant/closed-bar.md) · [五项数字](quant/five-kpis.md) · [先分类](quant/classify.md)
2. [什么是特征](features/what-is-a-feature.md) · [特征五族](features/families.md)
3. [本机四层](framework/four-layers.md) · [怎么选](framework/how-to-choose.md)
4. [填模板](design/write-the-sentence.md) · [跟 AI 说话](use/talk-to-the-ai.md)
5. [数据流与目录](tech/stack.md) · [命令地图](tech/commands-map.md)

装一次：[安装与第一次](use/install-and-first-run.md) · 跳读：[Q & A](qa/index.md) · [词汇表](glossary.md)

仓库里的 `docs/` 给 AI 和想自己翻细则的人。本站是教学地图。
