# How to choose

**One-liner:** Given a sentence, ask three things first: is it measuring one symbol or a basket? Does it earn from trend, reversal, or calendar? How does it exit when wrong? Answer those and you know which court to use and what contract to write.

## The three questions

1. **One symbol or a basket?**
   - One symbol (this coin on one bar) → event backtest
   - A basket (score dozens of coins every day) → cross-section backtest

2. **Trend, reversal, or calendar?**
   - Trend (follow MAs, momentum) → treat as beta first
   - Reversal (buy dips, sell rips) → alpha or fat-tail
   - Calendar (Monday, month-end, specific dates) → alpha

3. **How does it exit when wrong?**
   - Breaks a line → trend stop
   - Fixed number of days → time stop
   - Opposite signal → signal stop

## An example

“When BTC leads, AI alts follow”:

1. Measures cross-symbol lead-lag (BTC → AI alts) → event backtest, but needs features from two symbols.
2. Earns from trend (lead-follow) → treat as beta first.
3. Exit when wrong → BTC breaks a line, or the AI alt breaks its own line.

## Fine print

- [Classify first](../quant/classify.md)
- [Three courts](which-court.md)
