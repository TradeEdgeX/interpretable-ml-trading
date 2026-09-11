# 多因子横截面（cs_panel）

**English:** [cs_panel.en.md](cs_panel.en.md)

本仓库不只能量金叉那种「一根品种、一条时间轴」。  
**横截面多因子**（每天对一篮子名字打分、排序、做多一档）走另一张卷子：`cs_panel`。  
`mlbot research run` 只派 `event_backtest`。这类句子要自己跑登记过的入口。

能力 ≠ 有 edge。下面的已量句都已 `--declare reject`。框架能考，不保证考得过。

---

## 能验什么

事先锁死几列（不要过夜扫因子），每个交易日对宇宙里的股票算分，买高分一侧（或空低分一侧），对照同一宇宙的等权或现金。分窗只报年化 / Calmar / 胜率 / MaxDD / Sharpe。

| 句子类型 | 评测机入口 | 已量例子 |
|---|---|---|
| 个股多因子排序 vs 等权 | `python scripts/research/cs_panel.py` | [热票相对等权](examples/20260911_ashare_cs_mom_amount_CN.md) · reject |
| 板块先打分再买名字 | `python scripts/research/cs_sector.py` | [热板块相对等权](examples/20260911_ashare_cs_sector_cost_CN.md) · reject |
| 板块多空残差 | `python scripts/research/cs_sector.py --mode ls` | `20260911_ashare_cs_sector_alpha` · reject |
| 周频板块暴露 vs 现金 | `python scripts/research/cs_sector.py --mode weekly` | `20260911_ashare_cs_sector_beta` · reject |

查卷子：

```bash
PYTHONPATH=src python -m cli.main research harness ashare_cs_mom_amount
```

日历用 [`config/market_segment_ashare.yaml`](../config/market_segment_ashare.yaml) 的 `bear_2021` / `bull_924` / `chop_recent`，不要套币圈日期。

---

## 时钟（闭棒）

收盘才知道当天的因子。决定发生在 **T 收盘**，第一笔成交是 **下一根开盘**，一天的书收益是「下一根开盘 → 再下一根开盘」。

不许用当天已完成的 FeatureStore 行在开盘做决定。标签（未来 20 日收益）可以看未来；进场列不可以。

---

## 算法（锁死以后不要改题）

公开练习句锁了两列，权重各一半：

1. `mom_20`：收盘 / 20 根前收盘 − 1  
2. `amount_z_20`：成交额相对自己过去 20 根的 z  

每个交易日对两列分别做**截面 z**，`score = 0.5 * cs_z(mom) + 0.5 * cs_z(amount_z)`。  
书：买 `score` 前 20% 等权。对照：同一天全部可打分名字等权。

Phase 1 可以算当日 score 与未来 20 日开盘收益的 Spearman IC。IC 只探照灯，**不能结案、不能回头换列**。

板块句用同一对因子：先在板块内等权均值，再在板块间 z，买热板块里的名字（或空冷板块）。行业列是 `config/industry_map_ashare.yaml` 的 20 档粗分快照，不是申万一级、不是时点修订。

新句子要另写 `DECISION.md`，事先锁列和对照。不要在回测里 `compute_*` 新因子；缺列先登记 FeatureStore 再 backfill。

---

## 人怎么走

```text
人出句（列事先锁死）
  → 模板：社会 / 数学 / 统计 / 尺子 / A 股三段 / 五格
  → mlbot research validate <id>
  → mlbot research index --trusted --query cross-section
  → 人说「测」才跑 cs_panel / cs_sector
  → 人 --declare
```

```bash
PYTHONPATH=src python -m cli.main research validate 20260911_ashare_cs_mom_amount
PYTHONPATH=src python scripts/research/cs_panel.py
PYTHONPATH=src python scripts/research/cs_sector.py
PYTHONPATH=src python scripts/research/cs_sector.py --mode ls \
  --out results/ashare_cs_sector/experiments/20260911_ashare_cs_sector_alpha
PYTHONPATH=src python scripts/research/cs_sector.py --mode weekly \
  --out results/ashare_cs_sector/experiments/20260911_ashare_cs_sector_beta
```

日线默认 `data/ashare/daily/`，名单 `data/ashare/stock_basic/stock_basic.parquet`。`data/` 不进 git。

---

## 已量结论（不是推荐策略）

相对**同宇宙等权**（闭眼一人一份、几乎不换仓），锁死的热票 / 热板块 / 反转 / 板块多空，都没有稳定更好。  
周频热板块对**现金**三段年化为正，但对等权在牛、震荡更差。人已全部 `--declare reject`。

等权自己在这三段年化是绿的：那是小票市场 beta，不是选股逻辑。细则见两张例子。
