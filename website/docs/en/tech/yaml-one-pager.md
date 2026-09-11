# YAML in one page

**One-liner:** A strategy YAML is the machine-readable contract, translating plain language into “when trading is allowed, how you exit when wrong, which feature store to use”.

## What a strategy YAML looks like

```yaml
name: ma_cross
archetype: trend_follow
enabled: true

entry:
  condition: ema_50_200_cross_side == 1
  direction: long

exit:
  condition: close < ema_50
  stop_loss: atr_14 * 2

feature_store:
  layer: features_ma_cross_120T_<hash>
  timeframe: 120T

symbols: [BTCUSDT, BNBUSDT, SOLUSDT]
```

## What each section does

- `entry`: when trading is allowed (here: the 50-day line just crossed above the 200-day).
- `exit`: how you exit when wrong (here: break of the 50-day line, or a 2×ATR stop).
- `feature_store`: which feature store and timeframe to use.
- `symbols`: which symbols to run on.

## Fine print

- [Fill the template](../design/write-the-sentence.md)
- [docs/usage.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/usage.md)
