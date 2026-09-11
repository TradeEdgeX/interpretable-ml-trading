# Example: BTC surge → AI alts follow

Landing: [README.md](../../README.md).  
Experiment: [config/experiments/20260911_btc_lead_ai_alts/](../../config/experiments/20260911_btc_lead_ai_alts/).

Claim class: **beta** (alts vs BTC), not point-entry alpha.  
Feature: `btc_prior_bar_return` on alt bars (FeatureStore). **Not** `btc_roc90` inject.

Contract: BTC prior closed 2h return ≥ +3% → long NEAR/FET/RENDER; hold 6 bars; kill switch off; closed-bar.

---

## Local numbers

| Window | CAGR | Calmar | WR | MaxDD | Sharpe(R) | n |
|---|---:|---:|---:|---:|---:|---:|
| bear_2022 | no sample | — | — | — | — | 0 |
| bull_2023_2024 | +2.81% | 2.29 | 50.0% | −1.22% | 0.26 | 46 |
| recent_range_to_bear | −1.52% | −0.54 | 30.8% | −2.82% | −0.24 | 26 |

Local ticks start ~2023-01 for these alts — bear window has no listing sample. Recent CAGR negative and MaxDD deeper than bull. **You declare.** Leave `verdict:` empty.

中文：[20260911_btc_lead_ai_alts_CN.md](20260911_btc_lead_ai_alts_CN.md)
