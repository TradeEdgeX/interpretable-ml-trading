<div class="tx-hero" markdown>

<span class="tx-kicker">Hypothesis validator</span>

# interpretable-ml-trading

You have a trading idea and you are not sure it actually works. This repo helps you write that sentence as a **testable contract**: when trading is allowed, how you exit when wrong, what you risk, and which stretch of market measures it. Then it runs on **trades you downloaded**, a **monthly feature store**, and **one locked ruler**, and hands you a table: bear, bull, and recent windows, each with CAGR, Calmar, win rate, max drawdown, and Sharpe. The repo ships a base feature library (moving averages, crosses, funding, order flow, and more); you are invited to propose more testable ideas on top of those columns. The program prints the numbers; you read the report and the strategy, then write your own conclusion.

</div>

Same MA golden-cross sentence, two very different answers:

| | Web AI | Here |
|---|---|---|
| Gives you | “Classic trend; fake signals in ranges” — a story you cannot re-run | Bear / bull / recent tables of five KPIs |
| Recent-window CAGR (local) | — | **−3.1%** (source [README](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/README.md)) |

Evidence comes from trades and features on your machine; switch the chat model and the table stays the same. The AI helps fold the template, run the measurement, and retell the five KPIs. You write the conclusion after you understand the report.

## Five doors

<div class="grid cards" markdown>

-   **[Quant basics](quant/what-is-a-strategy.md)**

    Contract, closed bar, three windows, five KPIs, classify

-   **[Features and factors](features/what-is-a-feature.md)**

    Measurement, families, store-first, math is not a gate

-   **[Framework](framework/four-layers.md)**

    Four layers, how to choose, process, which court

-   **[Tech](tech/why-local.md)**

    Why local, paths, YAML, commands, who writes what

-   **[How to use](use/talk-to-the-ai.md)**

    How to speak and install

-   **[Gallery](gallery/index.md)**

    Measured sentences by phenomenon (including rejects)

</div>

## Where to start

1. [A strategy is a contract](quant/what-is-a-strategy.md) · [Closed bar](quant/closed-bar.md) · [Five KPIs](quant/five-kpis.md) · [Classify first](quant/classify.md)
2. [What is a feature](features/what-is-a-feature.md) · [Five families](features/families.md)
3. [Four local layers](framework/four-layers.md) · [How to choose](framework/how-to-choose.md)
4. [Fill the template](design/write-the-sentence.md) · [Talk to the AI](use/talk-to-the-ai.md)
5. [Stack and paths](tech/stack.md) · [Commands map](tech/commands-map.md)

Install once: [Install and first run](use/install-and-first-run.md) · Skim: [Glossary](glossary.md)

Repo `docs/` is for the AI and for people who want the fine print. This site is the teaching map.
