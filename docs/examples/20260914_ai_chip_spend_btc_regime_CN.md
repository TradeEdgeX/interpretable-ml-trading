# AI 算力开支过多时，BTC 是不是熊

实验：[config/experiments/20260914_ai_chip_spend_btc_regime/](../../config/experiments/20260914_ai_chip_spend_btc_regime/)  
先当 **beta / 市况共存**。评测机是 **phase1_scan_only**。Y 是闭棒 MA200，不是三段标签。判决仍空着。

人说的原句：AI 融资 / 算力开支过多时，BTC 处于熊或震荡，而不是随后几天涨不涨。

不是 [AI 融资公告后做多 BTC](20260914_ai_financing_btc_CN.md)，也不是 [BTC 领涨 AI 山寨](20260911_btc_lead_ai_alts_CN.md)。

---

## 规则

> 上一完整季 Epoch 芯片销售金额环比高于自身扩窗中位数时，BTC 日线收盘更常低于 200 日均线。若高强度日非牛比例不高于低强度日，这句话死。

| 格 | 内容 |
|---|---|
| 机制 | 季频芯片账单环比过高 → 闭棒非牛。 |
| 预期 | 开支加速的季对应 BTC 非牛。 |
| 合同 | 高强度时不许按牛市模板做多。不是开空。 |
| 证伪 | 高强度非牛比例 ≤ 低强度，或任一分列段反向。 |
| 落地 | 人手同一句。还没有 2h 进出 YAML。 |

X 钉在 [`config/research/ai_chip_sales_quarterly.csv`](../../config/research/ai_chip_sales_quarterly.csv)（Epoch AI chip sales，CC BY）。不用金额水平（几乎单调上升）。季末次日才可读。不用 `market_segment.yaml` 当 Y。

```bash
mlbot research validate 20260914_ai_chip_spend_btc_regime
mlbot research index --trusted --query ai-chip-spend
PYTHONPATH=src python scripts/research/ai_chip_spend_regime_scan.py
```

---

## Phase 1（不能结案）

| 范围 | 高强度非牛 | 低强度非牛 | 差 | n 高 / n 低 |
|---|---:|---:|---:|---:|
| 合并 | 13.4% | 50.5% | −37.1pp | 546 / 610 |
| bear_2022 | 27.7% | 无样本 | — | 213 / 0 |
| bull_2023_2024 | 14.9% | 39.7% | −24.8pp | 396 / 184 |
| recent_range_to_bear | 15.6% | 55.1% | −39.5pp | 90 / 425 |

方向和纸上声称相反：2023 年芯片账单翻倍时，BTC 更常站在 MA200 上。`bear_2022` 没有低强度日。判决用 `--declare`，不要手写 `verdict:`。

产物：`config/experiments/20260914_ai_chip_spend_btc_regime/quick_scan/chip_spend_regime.json`

English: [20260914_ai_chip_spend_btc_regime.en.md](20260914_ai_chip_spend_btc_regime.en.md)
