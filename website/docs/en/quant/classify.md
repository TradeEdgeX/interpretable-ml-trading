# Classify first

**One line:** Say where the money comes from before you pick the ruler. Do not kill a trend or fat-tail sleeve with win rate or “drop the top three” alone.

## Four classes (plain)

| Class | Money from | Don’t |
|---|---|---|
| **Alpha** | Better conditional expectation on ordinary days | Prove alpha with bull-market right tails alone |
| **Fat-tail harvest** | A few extreme paths pay | Cut the right tail to raise win rate |
| **Beta** | Exposure to a named factor (trend, market, …) | “Protect” long beta with short filters |
| **Useless** | Non-positive across windows | Keep sweeping thresholds to rescue the curve |

Factor = shared pay source; beta = your exposure to it; alpha = what remains after those exposures. Most “looks like alpha” lives in beta.

## Wrong vs right

| Wrong | Right |
|---|---|
| MA cross WR 35%, negative after dropping top 3 → “no edge”. | Treat the cross as **beta / trend exposure** first. Negative after top-3 is often arithmetic shape, not a verdict. |

## In this repo

- MA golden cross → beta first
- P99 notional chase → fat-tail right tail
- Monday rebound → calendar alpha (weak still uses the alpha ruler across windows)

## Deeper docs

- [Alpha vs fat-tail vs beta](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/design/alpha_vs_fattail_vs_beta_CN.md)
- [Lessons · classify](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/lessons.md#classify)
