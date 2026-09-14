# A 股周一跌、后四天涨

实验：[config/experiments/20260911_ashare_monday_rebound/](../../config/experiments/20260911_ashare_monday_rebound/)  
分类先当 **日历 alpha 声称**，不是大盘择时，也不是选股。  
评测机是 **event_backtest**（沪深300 · 日线）。纸面已出表；判决留给人 `--declare`，不要手写 `verdict:`。

> 周一收盘下跌后做多沪深300，持有四个交易日。用 A 股自己的三段日历测。

这句话量的是「周末坏消息在周一被定价、后四天情绪出清」能不能在 A 股三段上都活下来。赢了原句还活着；任一段年化为负，或近窗回撤深于两段趋势，原句死。它不是「周一开盘买、周五收盘卖」——许可发生在**周一收盘之后**，最早周二开盘进。

English: [20260911_ashare_monday_rebound.en.md](20260911_ashare_monday_rebound.en.md)

---

## 这一句在本仓库里怎么走完

你对网上的 AI 说：「A 股周一跌了，后四天会涨，有没有周一效应？」

它经常讲周末消息、讲「开盘买周五卖」。日历用的可能是美股，进场可能是周一开盘——那已经换题了。没有沪深300自己的熊 / 924 牛 / 近窗，也没有「年化接近零算不算成立」。

你对本仓库说同一句。AI 先收模板：付费方是周一的恐慌盘；`monday_down` 周一**收盘**才可知，最早周二开盘进；持有 4 根日线，不是周五必须走；日历必须是 [`market_segment_ashare.yaml`](../../config/market_segment_ashare.yaml)，不许套币圈 2022。你说「测一下」之后，才 `download-ashare`、建日线特征层、跑事件回测。

表印出来：三段年化都是很小的正数，近窗 +0.06%，Calmar 从 1 掉到 0.14。证伪线的两条硬门槛没有被年化打穿，但日历 alpha 付不起当一句话来做的成本。网上的 AI 会说「还是正的」。本仓库要你对照事先写好的尺子自己 `--declare`：正的几个基点不是可落地的合同。回测看起来「没亏」，也不许每个周一手做同一句。

粒度跟测量对象走：这句话是日线日历，不是 2 小时 tick。辅助盘已退役，日线下载只为法庭。下面七段把闭棒、A 股三段和表怎么读写完。

---

## 实验怎么设计的

```text
人出句（周一跌了，后四个交易日会涨）
  → 模板：社会 / 数学 / 统计 / 尺子 / A 股三段 / 五格
  → validate + 谱系（公开库此前无已结案同一句）
  → 人说「测一下」才下载日线、建 FeatureStore
  → 日线法院：monday_down=1 做多，持有 4 根日线
  → 人 --declare
```

走 `event_backtest`，网格里写 `market_segment_path: config/market_segment_ashare.yaml`。**不要**套币圈 `bear_2022`。辅助盘已退役；日线下载只为法庭例子，见 [RETIRED.md](../RETIRED.md)。

| 格 | 这一句 |
|---|---|
| 机制 | 周一收盘下跌（`monday_down=1`）后做多沪深300；持有 4 根日线时间出场。 |
| 预期市况 | 情绪出清后收复；三段都不该把账户做穿。 |
| 合同 | 周一收盘才知道跌了，最早周二开盘进；四根日线时间出场；不加仓；熔断关；闭棒。 |
| 证伪 | 任一段年化 < 0，或近窗 MaxDD 深于两段趋势。 |
| 落地 | 机器：本目录 `strategies/ashare_monday`。人手同一句。 |

| 模板格 | 这一句怎么写 |
|---|---|
| 社会学 | 周末坏消息在周一开盘被定价；恐慌盘付钱。若后四天只是情绪出清，接盘的人收复。付费方是周一的恐慌盘，不是「市场」。 |
| 数学 | 日线收益。`monday_down` 周一收盘可知。持有 4 个交易日。 |
| 统计学 | 先当日历 alpha。抽样用 A 股自己的政权窗，不套币圈日期。 |
| 验证标准 | 任一段年化 < 0，或近窗回撤更深。 |
| 数据范围 | `000300.SH` 日线 · `market_segment_ashare.yaml` · 层 `features_ashare_monday_1D`。 |

对照时钟（信号在收盘 t 可知，仓位从 **t+1** 起）：

| 时刻 | 发生什么 |
|---|---|
| 周一收盘 | 才知道 `monday_return < 0`，`monday_down=1` |
| 周二开盘 | 最早进场 |
| 之后四根日线 | 时间出场（大约到下周初，不是「周五收盘必须走」） |
| 现金日 | 周一没跌，或已经出完，收益记 0 |

只报五项 KPI：年化 / Calmar / 胜率 / MaxDD / Sharpe。不要合计 R。

---

## 数据

```bash
mlbot research validate 20260911_ashare_monday_rebound
mlbot research index --trusted --query monday
mlbot data download-ashare --symbols 000300.SH --start-date 2019-01-01
```

| 项 | 本机事实 |
|---|---|
| 品种 | 沪深300指数 `000300.SH` |
| 周期 | 日线（`1D`） |
| 来源 | `mlbot data download-ashare` |
| 落盘 | `data/ashare/daily/` |
| 特征层 | `features_ashare_monday_1D` |
| 日历 | [`config/market_segment_ashare.yaml`](../../config/market_segment_ashare.yaml) |
| 熔断 | 关 |

分窗（A 股自己的三段，不是币圈日期）：

| 段 | 起止 | 用途 |
|---|---|---|
| `bear_2021` | 2021-07-01 → 2022-10-31 | 沪深300见顶后的熊。日历 alpha 不该把账户做穿。 |
| `bull_924` | 2024-09-24 → 2025-05-31 | 924 政策组合拳后的快牛。 |
| `chop_recent` | 2025-06-01 → 2026-09-10 | 消化期近窗。不能单独 promote。 |

`crash_2015` / `bear_2018` / `covid_2020` 写在同一份日历里，是给十倍股队列用的入场日分桶，**不是**本句的三段。

---

## 特征

| 列 | 怎么来 | 闭棒用法 |
|---|---|---|
| `weekday` | 该日线是星期几（0 = 周一） | 开盘只读上一根；和 `monday_down` 一起过滤 |
| `monday_return` | 周一这根日线的收益 | 周一收盘才可知 |
| `monday_down` | `monday_return < 0` 则为 1 | 最早周二开盘才许进 |
| `atr_f` | 仓位用的波动 | 开盘只读上一根 |

实验包：`config/experiments/20260911_ashare_monday_rebound/strategies/ashare_monday/`

| 文件 | 写什么 |
|---|---|
| `features.yaml` | `atr_f` + `weekday_monday_f` |
| `prefilter.yaml` | `monday_down ≥ 1` 且 `weekday == 0` |
| `direction.yaml` | `fixed_direction: long` |
| `execution.yaml` | `time_stop_bars: 4` |
| `meta.yaml` | `timeframe: 1D`，层 `features_ashare_monday_1D` |
| 网格 | `monday_rebound_grid.yaml`（`market_segment_path` 指向 A 股日历） |

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

不要在回测里现场算「今天是不是周一」。缺列先建层。

---

## IC

**这条没有横截面 IC 表，也不该有。**

本句是日历 0/1 许可：周一收盘跌了才进。没有宇宙、没有连续分数。若有人把「周一跌多少」当连续分数去挖后四日 IC，那是另一句，要另开目录。

探照灯不能改题。结案只看 A 股三段五项 KPI。

---

## 验证

```bash
PYTHONPATH=src python -m scripts.event_backtest --variant-grid \
  config/experiments/20260911_ashare_monday_rebound/monday_rebound_grid.yaml
mlbot research close 20260911_ashare_monday_rebound
```

尺子事先写在纸上：任一段年化 **< 0**，或近窗 MaxDD **深于**两段趋势，原句就死。

### 三段结果（000300.SH · 日线 · 熔断关）

| 段 | 年化 | Calmar | 胜率 | 最大回撤 | Sharpe(R) | 笔数 |
|---|---:|---:|---:|---:|---:|---:|
| `bear_2021` | +0.61% | 1.16 | 52.9% | −0.53% | 0.12 | 34 |
| `bull_924` | +0.46% | 1.52 | 50.0% | −0.30% | 0.12 | 16 |
| `chop_recent` | +0.06% | 0.14 | 58.8% | −0.45% | 0.03 | 17 |

| 事先写的证伪线 | 数字有没有打中 |
|---|---|
| 任一段年化 < 0 | 没打中。三段都是很小的正数。 |
| 近窗回撤深于两段趋势 | 没打中。近窗 −0.45% 浅于熊段 −0.53%，深于牛段 −0.30%，但不是「深于两段」。 |

年化都在零附近。近窗 Calmar 从趋势段的 1.16 / 1.52 掉到 0.14：同样浅的回撤，几乎赚不到钱。日历 alpha 很弱，不是「日线就能当账户」。判决仍留空，等人对照尺子自己 `--declare`。

---

## 结论

分类：**日历 alpha 声称，量出来接近无用。**

- 三段年化为正，证伪线的两条硬门槛没有被年化打穿。
- 近窗年化 +0.06%、Sharpe 0.03：情绪出清即使存在，也付不起当一句话来做的成本。
- 笔数少（16–34）是周一下跌本身的密度，不是「再放宽到周二也算」。
- 不要把本表读成「周一效应成立」。正的几个基点不是可落地的合同。

判决用 `--declare`。不要手写 `verdict:`。不要回测看起来「没亏」就手做每个周一。

产物：

| 路径 | 是什么 |
|---|---|
| `results/ashare_monday/experiments/20260911_ashare_monday_rebound/monday_down/bear_2021` | 熊段报告 |
| `results/ashare_monday/experiments/20260911_ashare_monday_rebound/monday_down/bull_924` | 924 牛报告 |
| `results/ashare_monday/experiments/20260911_ashare_monday_rebound/monday_down/chop_recent` | 近窗报告 |
| [DECISION.md](../../config/experiments/20260911_ashare_monday_rebound/DECISION.md) | 纸面原文 |

---

## 报告解读

1. **先对日历，再看数字。** 这是 A 股 `bear_2021` / `bull_924` / `chop_recent`。不要拿币圈 2022 熊来解释沪深300。
2. **标题是年化，不是胜率。** 近窗胜率 58.8% 最高，年化却最矮。赢的是小波动，不是「后四天收复了很多」。
3. **Calmar 掉下来是机制。** 回撤都不到 1%，年化也从 0.61% 掉到 0.06%。浅回撤乘上几乎为零的收益，Calmar 会从 1 掉到 0.1。
4. **进场不是周一开盘。** 展厅或口头若说「周一开盘买、周五卖」，那是另一句，本表解释不了。
5. **4 根日线 ≠ 周五。** 周二进、持有四根，出场大约落在下下周，不是本周五收盘。
6. **笔数 16 的 924 牛不能单独 promote。** 快牛段周一下跌的次数本来就少。
7. **不要合计四天的 ΣR。** 尺子只看年化 / Calmar / 胜率 / MaxDD / Sharpe。
8. **辅助盘退役。** 本句只用指数日线。不要把旧的盘口字段塞回这张纸。

纸面原文在 [DECISION.md](../../config/experiments/20260911_ashare_monday_rebound/DECISION.md)。
