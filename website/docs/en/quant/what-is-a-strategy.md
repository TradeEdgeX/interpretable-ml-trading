# A strategy is a contract, not a feeling

People say: “Buy the golden cross; trends feel like they make money.” Those words are not yet a strategy.

A strategy has to write down at least four things:

1. When you are allowed to trade
2. How you get out when you are wrong
3. What you can lose on this trade
4. When the trade is over

Only then can someone else re-check it with the same ruler, and only then can you test whether the sentence is true or false.

## Which wording is a strategy

| Not yet a strategy | This is a strategy |
|---|---|
| “Buy the golden cross; trends feel like they make money.” | “Only buy when the close is above the 50-day line *and* it just crossed above the 200-day line; exit when it breaks the 50-day. If any stretch compounds to a loss over a year, or the chop stretch draws down deeper than the trend stretch, the sentence does not hold.” |

A feeling cannot be re-checked, and you cannot test whether it is true or false. A contract is different: when you trade, when you don’t, and what you do when you are wrong are written first. A backtest can then tell you what the sentence looks like in a bear, a bull, and the recent stretch.

## How this repo uses it

We use the moving-average golden cross as practice. The paper has to say: what you do, in which markets it should pay, the entry and exit rules, how the sentence fails, and whether the machine and a human are running the same sentence.

The rules in the program and the checklist you write by hand must be the same sentence. If the backtest does not pass, you read the report and close that same sentence — you do not make a “this time is different” version that night.

How to fill the boxes: [Five boxes](../design/five-boxes.md). Repo example: [README.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/README.md).

## Fine print

- [Hypothesis template](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/hypothesis_template.md)
- [Philosophy: charts are contracts](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/philosophy.md)
