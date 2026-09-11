# A 股周一跌、后四天涨

实验：[config/experiments/20260911_ashare_monday_rebound/](../../config/experiments/20260911_ashare_monday_rebound/)  
家族是 `ashare_monday`，结果在 `results/ashare_monday/…`。

> 周一收盘下跌后做多沪深300，持有四个交易日。用 A 股自己的三段日历测。

---

## 规则

| 格 | 内容 |
|---|---|
| 机制 | 周一收盘下跌（`monday_down=1`）后做多指数；持有 4 根日线。 |
| 预期市况 | 情绪出清后收复；三段都不该把账户做穿。 |
| 合同 | 闭棒：周一收盘才知道跌了，最早周二开盘进；四根日线时间出场；不加仓。 |
| 证伪 | 任一段年化 < 0，或近窗回撤深于两段趋势。 |
| 落地 | 机器和人手同一句。 |

统计先当日历 alpha。日历用 [`config/market_segment_ashare.yaml`](../../config/market_segment_ashare.yaml)，**不**套币圈 `bear_2022`。

---

## 查过没有

```bash
PYTHONPATH=src python -m cli.main research index --trusted --query monday
```

公开仓库当时 **0 条**已结案。

---

## 模板

```bash
PYTHONPATH=src python -m cli.main research validate 20260911_ashare_monday_rebound
```

模板不过只改 `DECISION.md`，不动数据。

---

## 日线与特征

| 用什么 | 从哪来 |
|---|---|
| 日线 | `mlbot data download-ashare --symbols 000300.SH` → `data/ashare/daily/` |
| 特征 | `weekday` / `monday_return` / `monday_down`（`weekday_monday_f`） |
| 时钟 | 开盘决策只读上一根已收盘特征 |
| 尺子 | `bear_2021` / `bull_924` / `chop_recent`；熔断关；五项 KPI |

```bash
PYTHONPATH=src python -m cli.main data download-ashare --symbols 000300.SH --start-date 2019-01-01
```

辅助盘仍退役；日线下载只为法庭例子。见 [RETIRED.md](../RETIRED.md)。

---

## 实验包

目录：`config/experiments/20260911_ashare_monday_rebound/strategies/ashare_monday/`

| 文件 | 写什么 |
|---|---|
| `features.yaml` | `atr_f` + `weekday_monday_f` |
| `prefilter.yaml` | `monday_down ≥ 1` 且 `weekday == 0` |
| `direction.yaml` | `fixed_direction: long` |
| `execution.yaml` | `time_stop_bars: 4` |
| `meta.yaml` | `timeframe: 1D`，层 `features_ashare_monday_1D` |

网格：`monday_rebound_grid.yaml`（`market_segment_path: config/market_segment_ashare.yaml`）。

---

## 特征库

```bash
PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/experiments/20260911_ashare_monday_rebound/strategies/ashare_monday \
  --symbols 000300.SH \
  --timeframe 1D \
  --root feature_store \
  --layer features_ashare_monday_1D \
  --data-path data/ashare/daily \
  --start-date 2021-01-01 --end-date 2026-09-10 \
  --no-reuse
```

---

## 三段（熔断关）

```bash
PYTHONPATH=src python -m scripts.event_backtest --variant-grid \
  config/experiments/20260911_ashare_monday_rebound/monday_rebound_grid.yaml
```

---

## 结果（000300.SH · 日线）

| 段 | 年化 | Calmar | 胜率 | 最大回撤 | Sharpe(R) | 笔数 |
|---|---|---|---|---|---|---|
| bear_2021 | +0.61% | 1.16 | 52.9% | −0.53% | 0.12 | 34 |
| bull_924 | +0.46% | 1.52 | 50.0% | −0.30% | 0.12 | 16 |
| chop_recent | +0.06% | 0.14 | 58.8% | −0.45% | 0.03 | 17 |

三段年化均为正；近窗 Calmar 明显弱于两段趋势。判决用 `mlbot research close --declare`，不要手写 `verdict:`。

产物：

- `results/ashare_monday/experiments/20260911_ashare_monday_rebound/monday_down/bear_2021`
- `results/ashare_monday/experiments/20260911_ashare_monday_rebound/monday_down/bull_924`
- `results/ashare_monday/experiments/20260911_ashare_monday_rebound/monday_down/chop_recent`

English: [20260911_ashare_monday_rebound.en.md](20260911_ashare_monday_rebound.en.md)
