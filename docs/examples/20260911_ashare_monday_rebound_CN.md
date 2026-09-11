# 例子：A 股周一跌、后四天涨

主入口仍是仓库根目录的 [README_CN.md](../../README_CN.md)。  
这一页把「周一跌了后四天会涨」从一句话走到五格、日线下载、特征库、YAML、A 股三段法庭、数字。判决仍由人写。

实验目录：[config/experiments/20260911_ashare_monday_rebound/](../../config/experiments/20260911_ashare_monday_rebound/)  
公开练习家族仍是 `ma_cross`。**不要改**默认包 `config/strategies/ma_cross/`。

---

## 你对网上的 AI 说

> A 股周一跌了，后面四个交易日是不是容易涨回来？

它通常会讲周末消息、周一恐慌盘、统计套利。没有 A 股自己的熊/牛/近窗数字，也没有「错了算谁输」。

---

## 你对本仓库的 AI 说同一句

> A 股周一收盘下跌后做多沪深300，持有四个交易日。我觉得是情绪出清。用 A 股自己的三段日历测一下。

差别在 AI **必须走下面这条路**，不许用网页文章当证据。

---

## 1. 先写成可验证的几条

| 格 | 这一句怎么写 |
|---|---|
| 机制 | 周一收盘下跌（`monday_down=1`）后做多指数；持有 4 根日线。 |
| 预期市况 | 情绪出清后收复；三段都不该把账户做穿。 |
| 合同 | 闭棒：周一收盘才知道跌了，最早周二开盘进；四根日线时间出场；不加仓。 |
| 证伪 | 任一段年化 < 0，或近窗回撤深于两段趋势。 |
| 落地 | 机器和人手同一句。 |

统计先当日历 alpha。日历用 [`config/market_segment_ashare.yaml`](../../config/market_segment_ashare.yaml)，**不**套币圈 `bear_2022`。

---

## 2. 先查有没有判过

```bash
PYTHONPATH=src python -m cli.main research index --trusted --query monday
```

公开仓库当时 **0 条**已结案。

---

## 3. 模板与 validate

```bash
PYTHONPATH=src python -m cli.main research validate 20260911_ashare_monday_rebound
```

模板不过只改 `DECISION.md`，不动数据。

---

## 4. 用的不是网页，是你机器上的日线

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

## 5. 规则写成实验包

目录：`config/experiments/20260911_ashare_monday_rebound/strategies/ma_cross/`

| 文件 | 写什么 |
|---|---|
| `features.yaml` | `atr_f` + `weekday_monday_f` |
| `prefilter.yaml` | `monday_down ≥ 1` 且 `weekday == 0` |
| `direction.yaml` | `fixed_direction: long` |
| `execution.yaml` | `time_stop_bars: 4` |
| `meta.yaml` | `timeframe: 1D`，层 `features_ashare_monday_1D` |

网格：`monday_rebound_grid.yaml`（`market_segment_path: config/market_segment_ashare.yaml`）。

---

## 6. 补特征库

```bash
PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/experiments/20260911_ashare_monday_rebound/strategies/ma_cross \
  --symbols 000300.SH \
  --timeframe 1D \
  --root feature_store \
  --layer features_ashare_monday_1D \
  --data-path data/ashare/daily \
  --start-date 2021-01-01 --end-date 2026-09-10 \
  --no-reuse
```

---

## 7. 三段法庭（熔断关）

```bash
PYTHONPATH=src python -m scripts.event_backtest --variant-grid \
  config/experiments/20260911_ashare_monday_rebound/monday_rebound_grid.yaml
```

---

## 8. 本机跑出来的结论（000300.SH · 日线）

| 段 | 年化 | Calmar | 胜率 | 最大回撤 | Sharpe(R) | 笔数 |
|---|---|---|---|---|---|---|
| bear_2021 | +0.61% | 1.16 | 52.9% | −0.53% | 0.12 | 34 |
| bull_924 | +0.46% | 1.52 | 50.0% | −0.30% | 0.12 | 16 |
| chop_recent | +0.06% | 0.14 | 58.8% | −0.45% | 0.03 | 17 |

三段年化均为正；近窗 Calmar 明显弱于两段趋势。**判决你来写。** `verdict:` 留空。

产物：

- `results/ma_cross/experiments/20260911_ashare_monday_rebound/monday_down/bear_2021`
- `results/ma_cross/experiments/20260911_ashare_monday_rebound/monday_down/bull_924`
- `results/ma_cross/experiments/20260911_ashare_monday_rebound/monday_down/chop_recent`

English: [20260911_ashare_monday_rebound.en.md](20260911_ashare_monday_rebound.en.md)
