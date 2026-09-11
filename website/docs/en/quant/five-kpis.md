# Five KPIs

**One line:** Speak and close cases with CAGR, Calmar, win rate, max drawdown, and Sharpe. Do not headline “Total R”.

## Five words (plain)

| KPI | Plain meaning |
|---|---|
| CAGR | How fast the account would grow (or shrink) if you followed the sentence |
| Calmar | Growth versus pain (CAGR ÷ drawdown depth) |
| Win rate | How often you are right — **you can make money with a low win rate** |
| Max drawdown | Worst peak-to-trough drop |
| Sharpe | How noisy the return path is |

“Total R” has a denominator that moves with equity; fat-tail holds get diluted. Using it to rank strategies misleads.

## Wrong vs right

| Wrong | Right |
|---|---|
| “Total R = 120 — strong.” | Report CAGR / Calmar / WR / MaxDD / Sharpe on each window. |

## In this repo

MA cross (BTC · 2h · local feature store; source [README](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/README.md)):

| Window | CAGR | Calmar | WR | MaxDD | Sharpe |
|---|---|---|---|---|---|
| Bear 2022 | +3.7% | 1.51 | 38.3% | −2.4% | 0.15 |
| Bull 2023–2024 | +3.0% | 0.97 | 33.3% | −3.1% | 0.15 |
| Recent | −3.1% | −0.68 | 24.3% | −4.5% | −0.36 |

## Deeper docs

- [Lessons · KPI](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/lessons.md#kpi)
