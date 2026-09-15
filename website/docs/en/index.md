<div class="tx-hero" markdown>

<span class="tx-kicker">A toolkit for an AI</span>

# interpretable-ml-trading

You propose a trading idea and write down when you would trade and how you would get out if you are wrong. This project is a toolkit that an AI assistant on your machine can use. It helps you check whether that idea actually holds, so you do not stay too optimistic. It writes a detailed report to help you shape a strategy. After you rule out a hundred ideas that do not hold, you might find one you can use. How you explored, and what each experiment concluded, are also kept here — that is the knowledge worth keeping.

</div>

Same MA golden-cross sentence, two very different answers:

| | Web AI | Here |
|---|---|---|
| Gives you | “Classic trend; fake signals in ranges” — a story you cannot re-run | Bear / bull / recent tables of five KPIs |
| Recent-window CAGR (local) | — | **−3.1%** (source [README](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/README.md)) |

Evidence comes from trades and measurements on your machine; switch the chat model and the table stays the same. The AI helps organize the idea, run the measurement, and write the report. You write the conclusion after you read it.

## Doors

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

-   **[Q & A](qa/index.md)**

    Can a rule cheat, why three windows, why correlation cannot close

-   **[Gallery](gallery/index.md)**

    Measured sentences by phenomenon (including rejects)

</div>

## Where to start

1. [A strategy is a contract](quant/what-is-a-strategy.md) · [Closed bar](quant/closed-bar.md) · [Five KPIs](quant/five-kpis.md) · [Classify first](quant/classify.md)
2. [What is a feature](features/what-is-a-feature.md) · [Five families](features/families.md)
3. [Four local layers](framework/four-layers.md) · [How to choose](framework/how-to-choose.md)
4. [Fill the template](design/write-the-sentence.md) · [Talk to the AI](use/talk-to-the-ai.md)
5. [Stack and paths](tech/stack.md) · [Commands map](tech/commands-map.md)

Install once: [Install and first run](use/install-and-first-run.md) · Skim: [Q & A](qa/index.md) · [Glossary](glossary.md)

Repo `docs/` is for the AI and for people who want the fine print. This site is the teaching map.
