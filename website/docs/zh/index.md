<div class="tx-hero" markdown>

<span class="tx-kicker">假设验证器</span>

# interpretable-ml-trading

你有一句交易想法，拿不准它到底有没有用。这个仓库帮你把这句话写成一份**能验证对错的合同**：什么时候允许做、做错了怎么走、这一笔亏在哪、用哪段行情来量。然后它用**你机器上下载的成交**、**按月算好的特征库**和**同一把事先定好的尺子**去跑，最后给你一张表：熊市、牛市、最近这段日子，各报折合成一年、Calmar、胜率、最大回撤、Sharpe。仓库自带一套基础测量（均线、交叉、费率、订单流等），欢迎你基于这些列提出更多能验证对错的想法。程序只负责算出数字；你阅读报告和策略之后，自己写下结论。

</div>

同一句均线金叉，问法不同，拿到的答案完全不同：

| | 网页 AI | 这里 |
|---|---|---|
| 给你什么 | 「经典趋势，震荡市假信号多」——一段无法复查的故事 | 熊市、牛市、最近这段，各一张五项数字的表 |
| 金叉近窗年化（本机） | — | **−3.1%**（数字来源 [README](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/README_CN.md)） |

证据来自你机器上的成交和特征库，换一个对话模型，表还是这张。AI 帮你整理模板、编排测量，并用五项 KPI 复述数字。结论由你理解报告之后写下。

## 入口

<div class="grid cards" markdown>

-   **[量化常识](quant/what-is-a-strategy.md)**

    策略要写成合同、收盘才知道的数字第二天才能用、至少看三段日子、表上只报五项数字、先说钱从哪来

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

1. [策略是合同](quant/what-is-a-strategy.md) · [闭棒](quant/closed-bar.md) · [五项 KPI](quant/five-kpis.md) · [先分类](quant/classify.md)
2. [什么是特征](features/what-is-a-feature.md) · [特征五族](features/families.md)
3. [本机四层](framework/four-layers.md) · [怎么选](framework/how-to-choose.md)
4. [填模板](design/write-the-sentence.md) · [跟 AI 说话](use/talk-to-the-ai.md)
5. [数据流与目录](tech/stack.md) · [命令地图](tech/commands-map.md)

装一次：[安装与第一次](use/install-and-first-run.md) · 跳读：[Q & A](qa/index.md) · [词汇表](glossary.md)

仓库里的 `docs/` 给 AI 和想自己翻细则的人。本站是教学地图。
