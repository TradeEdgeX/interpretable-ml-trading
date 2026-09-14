# AI financing days vs BTC funding and returns

Paper: [config/experiments/20260914_ai_financing_btc/](../../config/experiments/20260914_ai_financing_btc/)  
Class: **beta** (risk-on spillover), not point-selection alpha.  
Not [BTC leads AI alts](20260911_btc_lead_ai_alts.en.md). Not [funding fade](20260910_funding_fade.en.md).

中文：[20260914_ai_financing_btc_CN.md](20260914_ai_financing_btc_CN.md)

---

## Rule

> After a public mega AI-lab financing announcement (UTC date D closed), long BTCUSDT for five UTC days. None of the three windows should blow the book.

The locked calendar is `config/research/ai_financing_events.yaml` (≥ $300M). The FeatureStore flag `ai_financing_event` is 1 on the first 2h bar of D+1. Hold 60 two-hour bars. Kill switch off. Closed-bar.

---

## Phase 1 (does not close)

21 events. Spearman IC of the 5-day dummy vs next-day BTC return: **−0.021** (p = 0.40).

| Window | n | Event 5d mean | Median | Any-5d baseline |
|---|---:|---:|---:|---:|
| bear_2022 | 7 | +1.12% | +0.97% | +0.01% |
| bull_2023_2024 | 9 | +0.04% | −0.24% | +1.24% |
| recent_range_to_bear | 7 | +0.25% | −0.76% | −0.15% |

Bull underperforms buy-and-hold BTC. The recent mean is one OpenAI print on 2026-02-27 (+8.5%); the median is negative. Post-event 8h funding is only slightly higher than pre-event. AI-basket funding z (lag 1) vs BTC funding z: IC **0.098** (p = 0.003) — crowding moves together, and does not forecast next-day BTC.

Two-hour court KPIs are still empty. Human `--declare` only after `mlbot research run` / `close`.
