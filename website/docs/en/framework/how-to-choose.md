# How to choose

**One line:** Whatever the math names, download that and use that timeframe. Do not switch layers to look scientific.

## Ask yourself

1. On which bar is the decision knowable? (closed bar)
2. How long is “one bar” in the contract? Four trading days ≠ six 2h bars.
3. Is the symbol listed and on disk in this window? If not → “no sample”; recent alone cannot pass.
4. Is the calendar **this market’s** bear / bull / recent?
5. Are the columns in the feature store? Missing → register + same-layer incremental build.

## Gallery (plain)

| Sentence | Granularity | Calendar |
|---|---|---|
| MA cross | Trades → 2h | Crypto three windows |
| Funding fade | Funding series | Crypto three windows |
| Monday rebound | A-share daily | A-share three windows |
| P99 chase | **ticks** | Crypto three windows |
| Cross-section score | Daily universe | A-share · another court |

## Wrong vs right

| Wrong | Right |
|---|---|
| Explain 2h numbers with a website daily chart; paste crypto 2022 onto CSI 300. | Lock timeframe and calendar file in the template; missing years = no sample. |

## Deeper docs

- [Framework §3](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/framework.en.md)
- [Gallery](../gallery/index.md)
