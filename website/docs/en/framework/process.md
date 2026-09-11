# Process

**One line:** Human writes the sentence → template must pass before measuring → human declares. No “measure this” → no download, no backtest.

## Flow

```mermaid
flowchart TD
  A[Speak a trading idea] --> B[Fill the template]
  B --> C{Complete and fits the ruler?}
  C -->|no| B
  C -->|yes| D{Already judged in lineage?}
  D -->|yes| E[Retell the close; stop]
  D -->|no| F{Human asked to measure?}
  F -->|no| G[Init directory only; stop]
  F -->|yes| H[Data → features → court]
  H --> I[Program writes numbers]
  I --> J[Human declares]
```

## Wrong vs right

| Wrong | Right |
|---|---|
| AI downloads, sweeps params, and says “it works” for you. | Validate the template first; don’t rescan a judged sentence; program writes numbers, human writes the verdict. |

## Deeper docs

- [Talk to the AI](../use/talk-to-the-ai.md)
- [Who writes what](../tech/who-writes-what.md)
- [Architecture](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/ARCHITECTURE.en.md)
