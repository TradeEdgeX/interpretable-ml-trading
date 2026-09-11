# Gallery of measured sentences

Grouped by phenomenon, not by date. Numbers come from the local court ([framework §4](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/framework.en.md) and each example); **humans write the verdict**. Failed cases are shown as-is — not strategy tips.

Each card: plain sentence → class → granularity / calendar → feature family → three-window CAGR → source.

---

## Trend / beta

### MA golden cross

- **Plain:** Long when price is above the 50 and just crossed the 200; exit if it breaks the 50.
- **Class:** beta / trend exposure
- **Granularity / calendar:** trades → 2h · crypto three windows
- **Feature family:** close tech (EMA cross)
- **CAGR:** bear +3.7% / bull +3.0% / recent **−3.1%**
- [README full example](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/README.md)

---

## Crowding fade

### Extreme funding → fade

- **Plain:** Fade when funding z hits ±1.5; exit when z returns to 0.
- **Class:** crowding fade (mean-reversion-ish)
- **Granularity / calendar:** funding series · crypto three windows
- **Feature family:** crowding / positioning (`funding_rate_zscore_50`)
- **CAGR:** +2.7% / +3.0% / **−3.4%**
- [Source](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260910_funding_fade.en.md)

---

## Calendar

### A-share Monday down → hold four days

- **Plain:** After a Monday down close, long CSI 300 for four trading days.
- **Class:** calendar alpha (weak)
- **Granularity / calendar:** A-share daily · A-share three windows
- **Feature family:** close tech (weekday / `monday_down`)
- **CAGR:** +0.61% / +0.46% / +0.06%
- [Source](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260911_ashare_monday_rebound.en.md)

---

## Cross-asset beta

### BTC surge → AI alts follow

- **Plain:** After a large prior BTC bar, long theme alts for a few 2h bars.
- **Class:** beta (alts vs BTC)
- **Granularity / calendar:** trades → 2h · crypto three windows; **no sample in 2022 bear**
- **Feature family:** close tech (BTC column on host bars)
- **CAGR:** no sample / +2.81% / **−1.52%**
- [Source](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260911_btc_lead_ai_alts.en.md)

---

## Fat-tail right tail

### P99 notional + Bollinger chase

- **Plain:** Chase when bar max notional hits P99 and price sits on the upper band.
- **Class:** momentum / fat-tail right tail
- **Granularity / calendar:** **ticks** · crypto three windows
- **Feature family:** order flow + close tech (P99 + `bb_position`)
- **CAGR:** +0.24% / **−0.25%** / +0.28%
- [Source](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260911_p99_bb_break_chase.en.md)

---

## Cross-section (capability · rejected)

The framework can score a universe daily; **capability is not edge**. See [Which court](../framework/which-court.md) · [cs_panel](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/cs_panel.en.md).

### Hot + strong mom → beat equal-weight?

- **Class:** beta claim first; loses to equal-weight
- **Feature family:** cross-section (`mom_20` + `amount_z_20`)
- **Result:** top book underperforms EW in all three windows; rejected
- [Source](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260911_ashare_cs_mom_amount_CN.md)

### Hot sectors vs equal-weight?

- **Class:** sector rotation / beta; no stable excess vs EW
- **Feature family:** cross-section (sector means)
- **CAGR (10bp):** EW +6.4 / **+74.6** / **+30.5**; hot sectors +9.4 / +34.1 / +5.8 — loses in bull and chop; rejected
- [Source](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260911_ashare_cs_sector_cost_CN.md)

---

## Stories you cannot measure

### “Who is the next tenbagger?”

No trades, no closed-bar column → **cannot** be a hypothesis. You can only measure rules that already happened (e.g. small-cap on entry day, fixed hold years); that cohort sentence was rejected.

- [Source](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260911_tenbagger_smallcap.en.md)

---

When reading cards: [Closed bar](../quant/closed-bar.md) · [Five KPIs](../quant/five-kpis.md) · [Classify first](../quant/classify.md) · [Feature families](../features/families.md).
