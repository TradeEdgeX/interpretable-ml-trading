# Example: P99 large trade + Bollinger upper chase

Landing: [README.md](../../README.md).  
Experiment: [config/experiments/20260911_p99_bb_break_chase/](../../config/experiments/20260911_p99_bb_break_chase/).

Claim class: momentum / fat-tail right tail. Top-3 removal is classification only — not a solo kill.

Contract: `bar_max_notional_ge_p99 ≥ 1` and `bb_position ≥ 1` → long BTC 2h; exit when back inside band (`bb_position < 1`) or after 12 bars; kill switch off; closed-bar.

---

## Local numbers

| Window | CAGR | Calmar | WR | MaxDD | Sharpe(R) | n |
|---|---:|---:|---:|---:|---:|---:|
| bear_2022 | +0.24% | 0.32 | 45.5% | −0.74% | 0.09 | 22 |
| bull_2023_2024 | −0.25% | −0.26 | 36.4% | −0.95% | −0.10 | 22 |
| recent_range_to_bear | +0.28% | 0.92 | 57.1% | −0.30% | 0.16 | 21 |

Almost all exits are `structural_exit_bb_position_lt1`. Bull CAGR negative hits the falsifier. **You declare.** Leave `verdict:` empty.

中文：[20260911_p99_bb_break_chase_CN.md](20260911_p99_bb_break_chase_CN.md)
