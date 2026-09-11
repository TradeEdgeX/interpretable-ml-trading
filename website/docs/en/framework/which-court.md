# Three courts

**One-liner:** A court is a locked backtest script that only prints the five-KPI-by-window table. This repo has three: event backtest, cross-section backtest, portfolio backtest.

## The three courts

| Court | What it measures | When to use |
|---|---|---|
| Event backtest | Single symbol, event axis (bar by bar) | MA cross, funding fade, big-order chase |
| Cross-section backtest | A basket, scored and ranked daily | Momentum rank, amount rank |
| Portfolio backtest | Multi-strategy, multi-symbol portfolio | See how a few strategies work together |

## What “locked” means

The court’s fill logic (fees, slippage, closed bar) is locked and does not change with the strategy. That way all strategies use the same ruler and tables line up. You cannot make a sentence look better by tweaking the court’s fill parameters.

## How this repo uses it

- Event backtest: `scripts/event_backtest.py`
- Cross-section backtest: `scripts/cross_section_backtest.py`
- Portfolio backtest: `scripts/portfolio_backtest.py`

## Fine print

- [docs/framework.md §4 courts](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/framework.md)
