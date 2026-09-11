# 例子：BTC 大涨后 AI 山寨跟涨

主入口：[README_CN.md](../../README_CN.md)。  
实验：[config/experiments/20260911_btc_lead_ai_alts/](../../config/experiments/20260911_btc_lead_ai_alts/)。  
公开家族仍是 `ma_cross`。**不要改**默认包。

---

## 网上的 AI vs 本仓库

网上的 AI 会讲「BTC 定价风险偏好、山寨跟涨」。  
本仓库要：模板 → `validate` → 谱系 → 人说测一下 → 下载/特征库 → 三段 `event_backtest` → 五项 KPI。**不写 verdict。**

统计类是 **beta**（山寨对 BTC），不是选点 alpha。

---

## 合同

| 格 | 内容 |
|---|---|
| 机制 | BTC 前一根闭棒 2h 收益 ≥ +3%（FS 列 `btc_prior_bar_return`）做多 AI 山寨。 |
| 预期 | 风险偏好传导；三段不该做穿。 |
| 合同 | 持有 6 根 2h；不加仓；熔断关；闭棒。 |
| 证伪 | 任一段年化 < 0，或近窗回撤深于趋势段。 |
| 品种 | `NEARUSDT` / `FETUSDT` / `RENDERUSDT`（+ BTC 建领涨列）。`TAOUSDT` 若缺熊段不作主证。 |

**禁止**把 `btc_roc90` 注入当山寨入场特征。

---

## 命令摘要

```bash
PYTHONPATH=src python -m cli.main research validate 20260911_btc_lead_ai_alts
# 数据：mlbot data download/convert（本机可复用已有 parquet）
PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/experiments/20260911_btc_lead_ai_alts/strategies/ma_cross \
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

熊段无上市样本 → 不能靠近窗单独 promote。近窗年化为负且回撤深于牛段。**判决你来写。**

English: [20260911_btc_lead_ai_alts.en.md](20260911_btc_lead_ai_alts.en.md)
