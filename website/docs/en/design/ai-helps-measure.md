# The AI helps you measure

**One-liner:** The AI’s role in this repo is “translate your template into YAML, run the scripts, retell the table as prose”. It does not conclude — you write the conclusion after reading the table.

## What the AI can do

- Translate your five boxes into `config/strategies/<archetype>/*.yaml`.
- Run `scripts/event_backtest.py` or the cross-section script.
- Retell the three-window five-KPI table as a paragraph so it’s easier to read.

## What the AI cannot do

- Conclude for you (“this sentence holds / doesn’t hold”).
- Change the court’s fill logic (fees, slippage, closed bar).
- Compute features on the fly inside a backtest script (must go through the feature store).

## Why this split

The AI is good at translating and retelling, but not at judging “is this sentence losing in the recent window because the market changed”. That judgment needs your understanding of the strategy and the regime, so the conclusion is yours to write.

## Fine print

- [Talk to the AI](../use/talk-to-the-ai.md)
- [Who writes what](../tech/who-writes-what.md)
- [docs/agent/rd_playbook.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/agent/rd_playbook.md)
