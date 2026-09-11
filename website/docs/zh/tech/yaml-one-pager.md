# YAML 一页

**一句话：** 策略 YAML 是机器可读的合同，把人话翻译成「什么时候允许做、错了怎么走、用哪个特征库」。

## 一个策略 YAML 长什么样

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

## 各段干什么

- `entry`：什么时候允许做（这里：50 日线刚上穿 200 日线）。
- `exit`：错了怎么走（这里：跌破 50 日线或 2 倍 ATR 止损）。
- `feature_store`：用哪个特征库、哪个时间框架。
- `symbols`：在哪些品种上跑。

## 还想看细则

- [填模板](../design/write-the-sentence.md)
- [docs/usage.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/usage.md)
