# BTC 大涨后 AI 山寨跟涨

实验：[config/experiments/20260911_btc_lead_ai_alts/](../../config/experiments/20260911_btc_lead_ai_alts/)  
先当 **beta**（山寨对 BTC），不是选点 alpha。

---

## 规则

| 格 | 内容 |
|---|---|
| 机制 | BTC 前一根闭棒 2h 收益 ≥ +3%（FS 列 `btc_prior_bar_return`）做多 AI 山寨。 |
| 预期 | 风险偏好传导；三段不该做穿。 |
| 合同 | 持有 6 根 2h；不加仓；熔断关；闭棒。 |
| 证伪 | 任一段年化 < 0，或近窗回撤深于趋势段。 |
| 品种 | `NEARUSDT` / `FETUSDT` / `RENDERUSDT`（+ BTC 建领涨列）。`TAOUSDT` 若缺熊段不作主证。 |

领涨列用 BTC **前一根已收盘**的收益，不要拿 `btc_roc90` 当山寨入场特征。

---

## 命令摘要

```bash
PYTHONPATH=src python -m cli.main research validate 20260911_btc_lead_ai_alts
# 数据：mlbot data download/convert（本机可复用已有 parquet）
PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/experiments/20260911_btc_lead_ai_alts/strategies/btc_lead_alts \
  --symbols NEARUSDT,FETUSDT,RENDERUSDT \
  --timeframe 120T --root feature_store --layer features_btc_lead_alts_120T \
  --data-path data/parquet_data --start-date 2022-01-01 --end-date 2026-06-01 \
  --allow-partial --no-reuse
PYTHONPATH=src python -m scripts.event_backtest --variant-grid \
  config/experiments/20260911_btc_lead_ai_alts/btc_lead_alts_grid.yaml
```

---

## 本机数字（2h · 熔断关）

| 段 | 年化 | Calmar | 胜率 | 最大回撤 | Sharpe(R) | 笔数 |
|---|---|---|---|---|---|---|
| bear_2022 | 无样本 | — | — | — | — | 0 |
| bull_2023_2024 | +2.81% | 2.29 | 50.0% | −1.22% | 0.26 | 46 |
| recent_range_to_bear | −1.52% | −0.54 | 30.8% | −2.82% | −0.24 | 26 |

熊段无上市样本，不能靠近窗单独 promote。近窗年化为负且回撤深于牛段。判决用 `--declare`。

English: [20260911_btc_lead_ai_alts.en.md](20260911_btc_lead_ai_alts.en.md)
