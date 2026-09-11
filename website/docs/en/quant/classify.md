# Classify first

**One-liner:** Say where the money comes from first, then pick the ruler. Don’t kill a trend or fat-tail sleeve using win rate or “drop the biggest few trades” alone.

## Four types (in plain language)

| Type | Where the money comes from | Don’t |
|---|---|---|
| **Alpha** | Conditional expectation is better even on ordinary days | Prove alpha using only a big bull win |
| **Fat-tail harvest** | A few extreme paths pay | Cut the right tail to raise win rate |
| **Beta** | Standing on a known exposure (trend / market) | “Protect” long beta with a short filter |
| **Useless** | Loses across windows | Keep sweeping thresholds to save the curve |

Mnemonic: factor = a common paying source; beta = your exposure to it; alpha = the money left after those are paid out. Most of what looks like alpha lives in beta.

## Wrong vs right

| Wrong | Right |
|---|---|
| Golden-cross win rate 35%, dropping the biggest three trades is negative → “no edge”. | Treat the golden cross as **beta / trend exposure** first. Negative-after-Top-3 is often arithmetic shape, not a verdict. |

## How this repo uses it

- MA golden cross → treat as beta first
- P99 big-order chase → fat-tail right tail
- Monday rebound → calendar alpha (even if weak, judge it with the alpha ruler across windows)

## Fine print

- [alpha / fat-tail / beta](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/design/alpha_vs_fattail_vs_beta_CN.md)
- [Lessons · classify first](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/lessons.md#classify)
