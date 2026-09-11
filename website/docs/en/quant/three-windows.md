# Three windows

**One-liner:** Don’t sentence the whole sentence using only the last six months. Look at at least one down stretch, one up stretch, and one recent window, reported separately.

## Wrong vs right

| Wrong | Right |
|---|---|
| “Last six months CAGR is positive, strategy works.” | Bear, bull, and recent windows each report five KPIs; if any window breaks the falsification line, the whole sentence is void. A pretty recent window alone cannot close the case. |

Averaging three years into one CAGR washes out the bad years. Don’t copy the crypto 2022 bear dates onto A-shares — each market uses its own regime calendar.

## How this repo uses it

- Crypto default: `bear_2022` / `bull_2023_2024` / `recent_range_to_bear`
- A-shares: `bear_2021` / `bull_924` / `chop_recent` (a separate calendar file)

The golden cross’s recent-window CAGR is about −3.1%. Against the falsification line you wrote in advance, please read this table and the strategy, then write your own conclusion.

## Fine print

- [Framework · how to choose calendars](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/framework.md)
- [Lessons · three windows](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/lessons.md#segments)
