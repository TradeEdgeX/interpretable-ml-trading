# When AI chip spend accelerates, is BTC in a bear?

Paper: [config/experiments/20260914_ai_chip_spend_btc_regime/](../../config/experiments/20260914_ai_chip_spend_btc_regime/)  
Class: **beta / regime coexistence**. Harness is **phase1_scan_only**. Y is closed-bar MA200, not the segment labels. Verdict still empty.

中文：[20260914_ai_chip_spend_btc_regime_CN.md](20260914_ai_chip_spend_btc_regime_CN.md)

---

## Rule

> After a completed Epoch AI chip-sales quarter, if QoQ cost is above its expanding median, BTC daily closes should sit below the 200-day more often than on low-QoQ days. If they do not, the sentence dies.

The tape is pinned at `config/research/ai_chip_sales_quarterly.csv` (Epoch AI, CC BY). Do not use the dollar *level* (it rises almost every quarter). Readable only after quarter-end. Do not use `market_segment.yaml` as Y.

```bash
mlbot research validate 20260914_ai_chip_spend_btc_regime
PYTHONPATH=src python scripts/research/ai_chip_spend_regime_scan.py
```

## Phase 1 (does not close)

Pooled: high-QoQ days below MA200 **13.4%** vs low-QoQ **50.5%** (−37pp). Bull and recent windows have the same sign. The 2023 doubling of chip bills sat inside a BTC slow bull. `bear_2022` has no low-QoQ days. Human `--declare` only.
