# Do the hottest stocks beat “buy a little of everyone”?

**中文:** [cs_panel_CN.md](cs_panel_CN.md)

This page assumes no trading, backtest, or statistics background. One idea: people say “names that already rose a lot and are trading hot will keep beating the market.” We turn that into a sentence that can be true or false, print a table on three stretches of A-share history, then read the table.

---

## 1. The idea

You hear two versions:

1. **Names:** the stocks that rose the most over the last 20 sessions, and whose amount is hotter than *their own* recent past, will keep beating “the market.”
2. **Sectors:** if the name-level sentence dies, rank industries first, then buy the names inside the hot ones; after a small fee, they should still beat “buy a little of everyone.”

“The market” here is not the Shanghai index. It is: every name that can be scored that day, **the same amount of money in each**. Almost no turnover. Below, that book is “buy a little of everyone.”

“Hottest” is locked before the table. It is not swapped afterwards:

- how much the name rose over the last 20 sessions
- how hot its amount is versus **itself** over those 20 sessions

Each gets half the score. The score is known only at the close. You buy the top 20% at the **next open**. One day’s gain or loss runs from that open to the open after that.

This is not “being good at picking which name will rise.” It is buying the already-hot sleeve itself.

---

## 2. How the sentence loses

Written first: in the three stretches below, if **any one** of them has the hot sleeve compounding slower than “buy a little of everyone,” the sentence dies.

| Stretch | Roughly what the market was doing |
|---|---|
| 2021-07 → 2022-10 | Bear: many names falling |
| 2024-09-24 → 2025-05 | The fast bull after 24 September 2024 |
| 2025-06 → 2026-09 | Recent digestion / range |

“Compounded to a year” means: if the pace of that stretch lasted a full year, about how much the account would rise or fall. Compare only to “buy a little of everyone” on the **same dates and the same names**. Do not score the hot sleeve’s own green number against cash — that is a different sentence.

The full walk (who pays, which open, which exit) is in the two examples:

- [Hot names](examples/20260911_ashare_cs_mom_amount.en.md)
- [Hot sectors vs buy-a-little-of-everyone](examples/20260911_ashare_cs_sector_cost.en.md)

---

## 3. The report: buy the hottest names

| Stretch | Hottest 20%, compounded to a year | Buy a little of everyone, compounded to a year | Hottest minus everyone |
|---|---:|---:|---:|
| Bear | **−8.65%** | +4.70% | **13.35 points worse** |
| Fast bull | +24.13% | **+76.57%** | **52.44 points worse** |
| Recent range | +13.84% | +22.75% | **8.91 points worse** |

All three: hottest is worse. Against the ruler written first, the sentence is false. A human already judged it that way.

A side look: names with a high score **tend to weaken** over the next 20 sessions, not keep strengthening (same sign in all three stretches). That can warn you the “continuation” story may be backwards. It is not a report card by itself, and it is not a license to short the hottest names — that would be a new sentence.

---

## 4. Read that table in plain language

**Compare the two numbers on the same row before you trust a green cell.**

In the fast bull, “buy a little of everyone” compounds at about +76% a year. The small-name market itself was rising. The same dates, buying the hottest 20% each day, only about +24%. The sentence was “hot names keep beating the market.” The gap is 52 points. The green number sits on “everyone,” not on this sentence.

The bear is blunter: everyone still ekes out a small plus; the hottest sleeve loses money. Recently both are green; everyone is still faster.

Win rates sit near half on both sides. What you lose is size: when it works it works less, because the crowded hot names give away the move.

Do not:

- see +24% on the hot sleeve and say “it makes money” — the comparison is everyone, not cash
- flip the paper into “short the hot names” because high scores weaken later — write that as a new sentence and measure it
- hand-buy the hottest quintile at night after the table said no

---

## 5. Second sentence: rank hot sectors first?

The name-level sentence already lost. Someone says: that is because you did not rank industries, and a fee would make it fair.

A new control and a new ranking unit are a **new sentence**. Both sides pay 0.10% of the traded amount (10 basis points) up front. Industry is a coarse 20-bucket snapshot in this repo, not the exchange’s official first-level sectors, and not “the map on the wall that day.”

### The report

| Stretch | Names in the hottest 20% sectors (fee already charged), compounded to a year | Same names, same fee, buy a little of everyone | Hot sectors minus everyone |
|---|---:|---:|---:|
| Bear | +9.41% | +6.42% | 2.99 points better (this stretch passes) |
| Fast bull | +34.05% | **+74.59%** | **40.54 points worse** |
| Recent range | +5.79% | +30.48% | **24.69 points worse** |

The hot-sector book is green in all three stretches (about +9% / +34% / +6%). That is not the pass line. The sentence asks whether, after the **same** fee, hot sectors still beat “buy a little of everyone.”

In the bull and the range, hot sectors are slower **even before the fee**. Cost did not kill the sentence by itself. The hot book turns over about 40% of the position most days, so 0.10% eats about 10 points a year; everyone almost never trades, so the fee is noise. The bear is a small plus for hot sectors; the ruler still says any one stretch can kill the whole sentence.

A human already judged this sentence false too.

---

## 6. Summary

Both sentences buy the already-hot sleeve. They do not show that you can pick stocks. Measured: no stable edge versus “buy a little of everyone.”

Buying hot sectors weekly versus **cash** can be green in all three stretches — the control is cash, a different sentence, and it cannot overturn the table above. Long hot sectors / short cold ones was measured separately and is also false.

Fine print, fills, and the full numbers are in the two examples. This page is the same report in ordinary language.

---

## 7. Reproduce on this machine

You can tell the AI in this repo to “test this strategy” or “run this experiment.” Only then should it run the commands below. Daily bars default to `data/ashare/daily/` (not in git).

```bash
PYTHONPATH=src python scripts/research/cs_panel.py
PYTHONPATH=src python scripts/research/cs_sector.py
```
