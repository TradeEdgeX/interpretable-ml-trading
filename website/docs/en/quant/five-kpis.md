# Five KPIs

**One-liner:** Verbal and closing tables only use CAGR, Calmar, win rate, max drawdown, and Sharpe. Don’t headline “total R”.

## The five words (in plain language)

| KPI | Plain language |
|---|---|
| CAGR | If you follow this sentence, how fast does the account grow (or shrink) per year |
| Calmar | How fast it grows relative to how deep it hurts (CAGR ÷ drawdown depth) |
| Win rate | How often you are right — **making money does not require a high win rate** |
| Max drawdown | From the peak, how far did it worst-case fall |
| Sharpe | How smooth the returns are (how volatile) |

The denominator of “total R / Total R” changes with equity, and long-held fat tails get diluted — using it to compare strategies is easy to misread.

## Wrong vs right

| Wrong | Right |
|---|---|
| “Total R = 120, very strong.” | Three windows each report CAGR / Calmar / win rate / MaxDD / Sharpe. |

## How this repo uses it

Golden cross (BTC · 2h · local feature store; source [README_CN](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/README_CN.md)):

| Window | CAGR | Calmar | Win rate | MaxDD | Sharpe |
|---|---|---|---|---|---|
| Bear 2022 | +3.7% | 1.51 | 38.3% | −2.4% | 0.15 |
| Bull 2023–2024 | +3.0% | 0.97 | 33.3% | −3.1% | 0.15 |
| Recent | −3.1% | −0.68 | 24.3% | −4.5% | −0.36 |

## Fine print

- [Lessons · five KPIs](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/lessons.md#kpi)
