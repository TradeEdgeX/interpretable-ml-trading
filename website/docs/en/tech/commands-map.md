# Commands map

**One line:** Commands are the only ruler the AI may use — not a cheat-sheet for humans. Talking is enough day to day.

## Four groups

| Group | Role | When |
|---|---|---|
| `mlbot research …` | Paper: validate / index / init / close | Template and close; `--trusted` finds judged sentences only |
| `mlbot data …` | Download trades / funding / A-shares | After “measure this”; no hand-written klines |
| `mlbot feature-store …` | Build / backfill the store | Only when columns are missing; same-layer incremental |
| Court | `research run` / `event_backtest` | After measure; windows on; kill-switch off |

There is no `mlbot train`. Full switches live in repo docs — this site does not copy the encyclopedia.

## Deeper docs

- [Usage](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/usage.en.md)
- [Architecture · commands](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/ARCHITECTURE.en.md)
