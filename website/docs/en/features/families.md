# Five feature families

**One line:** Learn the family (what it answers / does not answer) before naming columns. Do not memorize 100+ names.

## Families

| Family | Answers | Does not answer | Worked example |
|---|---|---|---|
| **Close tech** (EMA / ATR / Bollinger / weekday) | Close-known location, crosses, calendar | Who pushed inside the bar | MA cross, Monday rebound |
| **Order flow** (VPIN / CVD / footprint / P99) | Who pushed; extreme prints | Fake delta from closes | P99 + Bollinger chase |
| **Crowding / positioning** (funding, OI) | Whether leverage crowding is extreme | Not order-flow ticks | Funding fade |
| **Slow math** (Hurst / spectrum / Hilbert) | How careful to be now | Whether this is a breakout | (Rare in public demos; see next page) |
| **Cross-section** (mom, amount z, sector) | Who is hot vs the basket today | Single-name event clock | Mom+amount (rejected) |

Close tech needs OHLC only. Order flow needs ticks. Funding is another file. Cross-section uses another court.

## Wrong vs right

| Wrong | Right |
|---|---|
| Add VPIN to a daily calendar sentence to look scientific. | Download the granularity the math names. |

## Deeper docs

- [Features · order flow / math](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/features.en.md)
- [Gallery](../gallery/index.md)
- [Math is not a gate](math-is-not-a-gate.md)
