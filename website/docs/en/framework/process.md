# Process

**One-liner:** From an idea to a KPI table, five steps: fill the template → pick the court → write the YAML → run the backtest → read the table and conclude.

## The five steps

1. **Fill the template**: write the idea into the five boxes of `docs/hypothesis_template.md` (mechanism, expected regime, contract, falsification, landing).
2. **Pick the court**: event backtest or cross-section backtest, based on “one symbol vs a basket”.
3. **Write the YAML**: translate the contract into `config/strategies/<archetype>/*.yaml`.
4. **Run the backtest**: `scripts/event_backtest.py` or the cross-section script, printing the three-window five-KPI table.
5. **Read the table and conclude**: against the falsification line you wrote in advance, write your own “this sentence holds / doesn’t hold in the recent window”.

## Who does what

- **You**: fill the template, read the table, write the conclusion.
- **The AI**: helps translate the template into YAML, runs the scripts, retells the table as prose.
- **The program**: computes features, fills trades, prints the table.

## Fine print

- [Fill the template](../design/write-the-sentence.md)
- [Talk to the AI](../use/talk-to-the-ai.md)
- [Who writes what](../tech/who-writes-what.md)
