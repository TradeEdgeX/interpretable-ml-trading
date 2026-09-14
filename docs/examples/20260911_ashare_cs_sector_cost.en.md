# Are hot sectors stock-picking alpha versus same-universe equal-weight?

Paper: [config/experiments/20260911_ashare_cs_sector_cost/](../../config/experiments/20260911_ashare_cs_sector_cost/)  
Class first as **sector rotation**: money moves between sectors, not “being good at picking the name inside a sector.” A human already judged the sentence false.

> Rank sectors first, then charge rebalance cost. Versus the *same* universe and the *same* 10bp, hot sectors should be better.

The name-level two-factor paper ([momentum + amount](20260911_ashare_cs_mom_amount.en.md)) already died versus equal-weight. This sentence changes both the ranking unit and the control. It is not a rename of the old table, and it is not “subtract 10bp on the old paper and declare a pass.”

中文：[20260911_ashare_cs_sector_cost_CN.md](20260911_ashare_cs_sector_cost_CN.md)

---

## How this sentence walks the repo

You ask a web AI: “The name-level book lost to equal-weight. Maybe we should rank sectors first, and subtract a fee so it’s fair?”

Adding industry and cost on the old paper, then declaring a pass, is not allowed here. A new control and a new ranking unit are a **new sentence**: new folder, lock “same universe, same 10bp” first. Industry is a 20-bucket coarse snapshot, not PIT 申万.

After “measure this,” print a table that scores sectors first, then buys the names inside the hottest ones. The hot-sector book is green in all three windows (+9% / +34% / +6%). A web AI stops there: “sector rotation works.” This repo compares to **the same names, same fee, one share each**: bull −40 points, chop −25 points. Daily turnover ≈ 0.38 eats about 10 points a year at 10bp — but bull and chop lose even before the fee. Cost did not kill the sentence by itself.

A human already judged it false. Weekly hot sectors versus cash can be green — different sentence, different ruler. Changing the control changes the sentence. Absolute green can still die versus one-share-each. After the judgment, do not “read the hot sector first” and pretend the fee was never measured.

The seven sections below unpack cost and the control.

---

## Design

```text
Human sentence (rank sectors, assume a rebalance fee)
  → write down: same names, same fee; the score is a sector score
  → check: this sentence was not already closed
  → average the two columns inside each sector, rank sectors, buy names in the hottest 20%
  → charge 10bp each side on both books
  → human writes whether it holds
```

The old zero-cost all-A EW control is retired for this close. Industry is a 20-bucket coarse snapshot, not PIT 申万.

| Box | This sentence |
|---|---|
| Mechanism | Equal-weight mean of `mom_20` / `amount_z_20` inside each sector, z across sectors, buy names in the top 20% sectors equal-weight. |
| Regimes | After stripping mixed small-name noise, the relative sign versus the same-cost control should hold. |
| Contract | Daily rebalance; 10bp each side on *both* books; closed-bar (score at close, next open). |
| Falsifier | Any window: hot-sector minus same-cost EW CAGR ≤ 0. The hot book being green on its own is **not** a pass. |
| Landing | Hands: read the sector first, then the name; cost the turnover as expensive, not zero. |

Clock matches the name-level sentence: T close score, T+1 open in, T+2 open marks one day. Only “who is in the top 20%” and the 10bp charge change.

---

## Data

```bash
PYTHONPATH=src python scripts/research/cs_sector.py
```

| Item | This machine |
|---|---|
| Industry map | [`config/industry_map_ashare.yaml`](../../config/industry_map_ashare.yaml), 20 coarse buckets, **3633** names; `综合` 772 |
| PIT revisions | `industry_pit: false` (latest table pasted onto the entry date) |
| Panel | 5175 aligned daily names; 7,417,456 industry rows |
| Cost | **10bp** each side, locked, not swept |
| Calendar | [`config/market_segment_ashare.yaml`](../../config/market_segment_ashare.yaml) |
| Kill switch | Off |

Related papers that do **not** close this one:

| Folder | What it measured | Relation |
|---|---|---|
| `20260911_ashare_cs_sector_alpha` | Sector long–short residual | Measured; bull / chop negative; reject. Cannot keep this table alive. |
| `20260911_ashare_cs_sector_beta` | Weekly hot sectors vs **cash** | Cash-control CAGRs green, MaxDD above −40%; different paper. Do not reuse this table’s *absolute* CAGR as a pass. |

---

## Features

| Column | Construction | Closed-bar use |
|---|---|---|
| Name `mom_20` / `amount_z_20` | Same locked pair as the name-level paper | Known at T close |
| Sector score | Equal-weight mean inside the sector, then z across sectors | Today’s closed members only |
| Composite | Combine the two sector z’s | Buy the top 20% **sectors**, then EW the names inside |
| Industry | 20-bucket snapshot on the symbol | Not 申万-I; not PIT |
| Turnover | Daily holdings difference | 10bp × turnover into the book |

Do not blend this book with the name-level top 20%. That book already died.

---

## IC

**This sentence does not print a separate sector IC table.**

The name-level Phase 1 already showed `mom_20` / `amount_z_20` / `score` daily IC negative in all three windows. This sentence uses the same pair, averaged inside the sector first. IC cannot change the sentence. The close is hot-sector versus same-cost EW relative CAGR.

A “sector score versus next-20-day sector return” IC is a new folder.

---

## Validation

Ruler, written first: any window hot-sector minus **same-universe, same 10bp** EW CAGR **≤ 0**.

### Books (kill switch off · 10bp each side)

| Book | `bear_2021` | `bull_924` | `chop_recent` |
|---|---:|---:|---:|
| Same-universe EW (10bp) | +6.42% | **+74.59%** | **+30.48%** |
| Hot-sector top 20% (10bp) | **+9.41%** | +34.05% | +5.79% |
| Hot-sector gross (0 cost) | +20.52% | +47.57% | +16.29% |

| Window | vs same-cost EW | Gross vs EW | Falsifier |
|---|---:|---:|---|
| `bear_2021` | **+2.99pp** | Gross also beats EW | This window passes |
| `bull_924` | **−40.54pp** | Gross +47.57% still loses to EW +74.59% | Hit |
| `chop_recent` | **−24.69pp** | Gross +16.29% still loses to EW +30.48% | Hit |

Hot-sector CAGRs are all green (+9.41% / +34.05% / +5.79%). That is **not** this sentence’s gate.

### Where the fee comes from

| Book | Daily one-way turnover | 10bp × turnover × 252 |
|---|---:|---|
| Hot-sector top 20% | about **0.38** | about **9–10** points a year |
| Same-universe EW | about **0.001** | almost nothing |

Bear gross +20.52% becomes +9.41% after cost. EW barely turns over. Bull and chop lose to EW *even gross*: cost did not kill the sentence by itself.

---

## Conclusion

Class: **sector-rotation / beta claim, no stable excess versus equal-weight.** Not sector-picking alpha.

- Bear relative +2.99pp passes that window’s ruler.
- Bull / chop relative −40pp / −25pp hit the falsifier. The sentence cannot be used as money.
- Gross books also lose in bull / chop: death is “hot sectors keep running,” not “10bp was too harsh.”
- Long–short residual and weekly-versus-cash are other papers, already closed on their own rulers.

A human already judged it false. The robot does not run it; a human should not “read the hot sector first” versus one-share-each.

---

## How to read the report

1. **Control first, then the green number.** Hot +34% looks like “it makes money”; EW is +75% in the same window.
2. **The gross row is diagnosis, not a second verdict.** It blocks “if cost were lower we would have passed.”
3. **Turnover 0.38 is the mechanism.** About 40% of the book moves every day. EW is almost free under “same cost.”
4. **The map is not PIT 申万.** 3633 names, 20 buckets, `industry_pit: false`.
5. **Do not revive this with weekly-versus-cash green.** Different control, different falsifier.
6. **Do not splice name-level EW +76.57% onto this +74.59%.** Universe (industry coverage) and cost differ.
7. **Bear +2.99pp cannot promote.** The ruler says “any window.”
8. **reject binds hands.** Do not “read the hot sector first” and pretend 10bp was never measured.

The paper is [DECISION.md](../../config/experiments/20260911_ashare_cs_sector_cost/DECISION.md).
