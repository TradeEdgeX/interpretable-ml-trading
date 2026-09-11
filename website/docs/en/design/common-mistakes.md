# Common mistakes

**One line:** These errors make examples unreadable and numbers falsely pretty.

## Checklist

| Mistake | Why it hurts | Read instead |
|---|---|---|
| Add filters to rescue the curve | Post-hoc knobs, not a prior contract | [Fill the template](write-the-sentence.md) |
| Kill trend / fat-tail with WR or drop-top-3 | Wrong class of ruler | [Classify first](../quant/classify.md) |
| Recent window only | Recent alone cannot promote | [Three windows](../quant/three-windows.md) |
| Stuff cross-section into single-name backtest | Wrong court shape | [Which court](../framework/which-court.md) |
| Switch unrelated data to look scientific | Granularity must match math | [How to choose](../framework/how-to-choose.md) |
| Compute features inside the backtest | Clock misalignment; unauditable | [Feature store first](../features/store-first.md) |
| Slow math as a hard gate | Hindsight filter | [Math is not a gate](../features/math-is-not-a-gate.md) |
| Headline Total R | Moving denominator; misranks holds | [Five KPIs](../quant/five-kpis.md) |
| Read same-bar features at the open | Lookahead | [Closed bar](../quant/closed-bar.md) |

## Deeper docs

- [Lessons](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/lessons.md)
