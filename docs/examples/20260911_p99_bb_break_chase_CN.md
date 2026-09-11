# 例子：P99 大单 + 布林上轨突破追涨

主入口：[README_CN.md](../../README_CN.md)。  
实验：[config/experiments/20260911_p99_bb_break_chase/](../../config/experiments/20260911_p99_bb_break_chase/)。  
公开家族仍是 `ma_cross`。

---

## 网上的 AI vs 本仓库

网上的 AI 会讲「大单吃货 + 突破上轨追涨」。  
本仓库要：模板 → validate → FeatureStore（tick P99 + `bb_position`）→ 三段法庭。统计类是动量 / 肥尾右尾；去 Top-3 只作分类，不单独否。

---

## 合同

| 格 | 内容 |
|---|---|
| 机制 | `bar_max_notional_ge_p99 ≥ 1` 且 `bb_position ≥ 1` 做多。 |
| 合同 | 回到带内（`bb_position < 1`）或 12 根 2h 时间出场；熔断关；闭棒。 |
| 证伪 | 任一段年化 < 0，或近窗回撤深于趋势段。 |
| 品种 | `BTCUSDT` · 2h · 币圈三段。 |

缺 tick 的月份写 NaN，不编数字。

---

## 命令摘要

```bash
PYTHONPATH=src python -m cli.main research validate 20260911_p99_bb_break_chase
PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/experiments/20260911_p99_bb_break_chase/strategies/ma_cross \
  --symbols BTCUSDT --timeframe 120T --root feature_store \
  --layer features_p99_bb_chase_120T --data-path data/parquet_data \
  --start-date 2022-01-01 --end-date 2026-05-31 --no-reuse
PYTHONPATH=src python -m scripts.event_backtest --variant-grid \
  config/experiments/20260911_p99_bb_break_chase/p99_bb_chase_grid.yaml
```

---

## 本机数字（BTC · 2h · 熔断关）

| 段 | 年化 | Calmar | 胜率 | 最大回撤 | Sharpe(R) | 笔数 |
|---|---|---|---|---|---|---|
| bear_2022 | +0.24% | 0.32 | 45.5% | −0.74% | 0.09 | 22 |
| bull_2023_2024 | −0.25% | −0.26 | 36.4% | −0.95% | −0.10 | 22 |
| recent_range_to_bear | +0.28% | 0.92 | 57.1% | −0.30% | 0.16 | 21 |

出场几乎全是回到带内。牛段年化为负 → 证伪线打中。**判决你来写。**

English: [20260911_p99_bb_break_chase.en.md](20260911_p99_bb_break_chase.en.md)
