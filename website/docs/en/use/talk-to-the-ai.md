# Talk to the AI

**One-liner:** You don’t need to write code — you only need to write your idea into the five-box template. The AI translates it into YAML, runs the backtest, and retells the table.

## How to speak

| You say | The AI does |
|---|---|
| “I want to test whether the MA golden cross works on BTC 2h.” | Helps you fill the five boxes, translates into YAML, runs the event backtest, retells the three-window five-KPI table. |
| “I want to test the Monday rebound.” | Helps you fill the template, translates into YAML, runs the event backtest, retells the table. |
| “This result loses in the recent window — did the market change?” | Helps you retell the recent vs bear/bull comparison, but the conclusion is yours. |

## What the AI won’t do

- Conclude for you (“this sentence holds / doesn’t hold”).
- Change the court’s fill logic.
- Compute features on the fly inside a backtest script.

## A full example conversation

1. You: “I want to test whether the MA golden cross works on BTC 2h.”
2. AI: helps you fill the five boxes, translates into YAML, runs the event backtest.
3. AI: “Here is the three-window five-KPI table: bear +3.7%, bull +3.0%, recent −3.1%. Against the falsification line you wrote in advance — ‘any window with negative CAGR voids it’ — please read this table and the strategy, then write your own conclusion.”
4. You: read the table, write the conclusion.

## Fine print

- [Fill the template](../design/write-the-sentence.md)
- [The AI helps you measure](../design/ai-helps-measure.md)
- [Who writes what](../tech/who-writes-what.md)
