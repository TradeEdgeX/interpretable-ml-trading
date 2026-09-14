# 资金费率极端拥挤就反手

实验：[config/experiments/20260910_funding_fade/](../../config/experiments/20260910_funding_fade/)  
先当 **拥挤回吐**：挤在一边的人该把钱吐出来，不是「会选点」，也不是趋势跟随。表已经出来；结不成立由你对照尺子写。

> 资金费率特别高就空、特别低就多，回到正常就走。费率极端拥挤的一边该回吐；熊 / 牛 / 近窗都不该把账户做穿。

这句话量的是「永续资金费率自己的拥挤」能不能在三段上都活下来。赢了原句还活着；任一段年化为负，或近窗回撤深于两个趋势段，原句就不成立。它不是「再加一层持仓量过滤」的起点。

English: [20260910_funding_fade.en.md](20260910_funding_fade.en.md)

---

## 这一句在本仓库里怎么走完

你对网上的 AI 说：「资金费率特别高就空、特别低就多，回到正常就走，能赚钱吗？」

它通常会说：这是经典拥挤反转，再配合持仓量更好。没有熊 / 牛 / 近窗的数字，也没有「近窗亏了算谁输」。换一个模型，它会建议再加一层过滤。

你对本仓库说同一句，并加上「测试一下这个策略」。AI 必须先把原句收进[假设模板](../hypothesis_template.md)：付钱的是挤在一边的永续杠杆盘；量的是费率**自己多久结算一次**上的拥挤程度（大约 8 小时一次，不是「最近 50 根 2 小时 K 线」）；先当拥挤回吐，不是趋势跟随；任一段全年涨速为负，或最近一段回撤深于前面两段，这句话就不成立。模板过了、以前没有判过同一句，你说了「测试一下这个策略」，才下载成交和费率，再按 2 小时进出场算出一张表。

拥挤程度仓库里本来就会算。缺的是「回到正常就走」这条出场规则，所以补的是规则，不是在算出表的时候现场另算一列。程序印出三段：趋势两年小赚，最近一段 −3.4%，回撤也更深。两条输赢线都打中。结不成立由你写。不成立就机器不跑，也不要手盘「这回费率不一样」。不要在同一句上再加「还要看持仓量」，那是另一句话。

这就是本仓库的哲学：先有一句人写的规矩，再用同一把尺子验证对错；相关扫描不能当成绩单；回测已经说明不成立的句子，手做也不许。下面七段是同一条路的展开。

---

## 实验怎么设计的

```text
人出句（费率极端就反手，回到 0 就走）
  → 先写成：谁付钱、量什么、哪段日子、怎样算输
  → 查过：以前有没有判过同一句
  → 你说「测试一下这个策略」之后，才下载成交和费率、事先算好拥挤程度
  → 按 2 小时进出场：极端就反手，回到正常就走
  → 人写下结不成立
```

这句按比特币 2 小时棒进出场。拥挤程度事先算好、按月存着；缺的是「回到正常就走」这条出场规则。缺一个事先算好的数字，就先补进表里；缺一条进出规则，才改规则。不要在算出这张表的时候现场另算一列拥挤程度。

| 格 | 这一句 |
|---|---|
| 机制 | 资金费率最近 50 次观察的稳健 z 分数 ≥ 1.5 开空，≤ −1.5 开多。方向取 z 分数的反号（`negate_sign`）。 |
| 预期市况 | 极端拥挤该回吐；单边趋势年不该把反手账户做穿；近窗若拥挤不再均值回复，年化会翻负。 |
| 合同 | z 回到 0 就走（多头 z≥0，空头 z≤0）。不加仓、不摊、不跟踪止盈。熔断关。闭棒。 |
| 怎样算不成立 | 任一段年化为负，或近窗最大回撤深于两个趋势段。近窗子窗不能单独过关。 |
| 落地 | 机器：本目录 `strategies/funding_fade`。人手同一句。不要为了再加持仓量过滤，在同一句上继续拧。 |

| 模板格 | 这一句怎么写 |
|---|---|
| 社会学 | 付钱的是挤在拥挤一边的永续杠杆盘。费率是协调仪式：大家都在同一侧付钱时，后到的人把费率推得更极端。 |
| 数学 | 测量对象是费率自己的节奏上的稳健 z 分数，不是 50 根 2 小时 K 线。出场是 z 穿过 0。 |
| 统计学 | 先当拥挤回吐。三段分列五项 KPI，不合成一条年化。 |
| 验证标准 | 任一段年化 < 0，或近窗 MaxDD 深于熊段和牛段。 |
| 数据范围 | `BTCUSDT` · 2 小时 · `bear_2022` / `bull_2023_2024` / `recent_range_to_bear`。 |

对照时钟（信号在收盘 t 可知，仓位从 **t+1** 起）：

| 时刻 | 发生什么 |
|---|---|
| 进场 | 上一根已收盘的 `funding_rate_zscore_50` 绝对值 ≥ 1.5 |
| 方向 | z 为正做空，z 为负做多 |
| 出场 | 约 94% 的笔是 z 穿过 0；其余是宽初始止损，不是跟踪止盈 |
| 现金日 | 不在市，收益记 0 |

只报五项 KPI：年化 / Calmar / 胜率 / MaxDD / Sharpe。不要合计 R。

---

## 数据

```bash
mlbot research validate 20260910_funding_fade
mlbot research index --trusted --query funding
mlbot data download --symbols BTCUSDT \
  --start-year 2021 --start-month 6 --end-year 2026 --end-month 6
mlbot data convert --symbols BTCUSDT
mlbot data download-funding-rate --symbols BTCUSDT \
  --start-year 2020 --start-month 1 --end-year 2026 --end-month 6
```

| 项 | 本机事实 |
|---|---|
| 品种 | `BTCUSDT` 永续 |
| 周期 | 2 小时棒（`120T`） |
| 成交来源 | `mlbot data download` + `convert` → `data/parquet_data` |
| 费率来源 | `mlbot data download-funding-rate` → `data/funding_rate/parquet` |
| 费率节奏 | 大约每 8 小时一次，50 次观察大约 17 个自然日 |
| 特征层 | `features_funding_fade_120T`（61 个月；每月只要 ATR 和费率，层会瘦） |
| 抽查 | 2023-05：`funding_rate_zscore_50` 无空值；约 18% 的棒 `|z| ≥ 1.5` |
| 日历 | [`config/market_segment.yaml`](../../config/market_segment.yaml)，不要套 A 股 `bull_924` 或美股窗 |
| 熔断 | 关。比的是这句话本身，不是账户保护 |

分窗：

| 段 | 起止 | 用途 |
|---|---|---|
| `bear_2022` | 2022-01-01 → 2023-11-01 | 熊市基线。反手不该在单边里被趋势碾穿。 |
| `bull_2023_2024` | 2023-06-01 → 2025-01-01 | 牛市主段。多头拥挤该回吐，不该把空头账户做穿。 |
| `recent_range_to_bear` | 2025-01-01 → 2026-05-31 | 近窗。不能单独过关。 |
| `recent_6m_oos` | 2025-12-01 → 2026-05-31 | 更短的参考窗。本句不拿它结案。 |

纸面写的就是这三条 canonical 窗。费率文件从 2020 起下，是为了让 2022 年初的 z 分数有预热，不是另开一条「2020 全期」判决。

---

## 特征

合同用到的拥挤程度必须事先算好再读。不要算出表的时候现场另算。

| 列 | 怎么来 | 闭棒用法 |
|---|---|---|
| `funding_rate` | 币安永续资金费率，按费率自己的时间对齐到 2h 棒 | 只作节点输入，不直接当方向 |
| `funding_rate_zscore_50` | 最近 50 次费率观察的中位数 / 绝对中位差稳健 z | `z[t]` 最早下一根才开仓 |
| `atr_f` | 仓位和宽止损用的波动 | 开盘只读上一根 |
| 出口 | 引擎合同 `structural_exit: funding_zscore0` | 多头看到 z≥0、空头看到 z≤0 才走；当根收盘才知道，下一根平 |

实验包：`config/experiments/20260910_funding_fade/strategies/funding_fade/`

| 文件 | 写什么 |
|---|---|
| `features.yaml` | `atr_f`、`funding_rate_features_f` |
| `archetypes/prefilter.yaml` | `\|funding_rate_zscore_50\| ≥ 1.5` 才开 |
| `archetypes/direction.yaml` | 特征 `funding_rate_zscore_50`，`negate_sign` |
| `archetypes/execution.yaml` | `structural_exit: funding_zscore0`；关掉跟踪止盈、加仓 |
| `meta.yaml` | 特征库层 `features_funding_fade_120T` |
| 网格 | `funding_fade_grid.yaml`（`strategies_root` 指到上面这个包） |

```bash
PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/experiments/20260910_funding_fade/strategies/funding_fade \
  --symbols BTCUSDT \
  --timeframe 120T \
  --root feature_store \
  --layer features_funding_fade_120T \
  --data-path data/parquet_data \
  --start-date 2021-06-01 \
  --end-date 2026-06-01 \
  --no-reuse
```

建在你自己的 `feature_store` 里。缺预热窗的日子对应空仓，不是用未来费率填 z。

---

## IC

**这条没有横截面 IC 表，也不该有。**

IC（按日 Spearman：分数 vs 未来收益）是给「每天对一篮子名字打分、买前 N 名」用的探照灯，见 [涨得多又热的票](20260911_ashare_cs_mom_amount_CN.md)。本句是单品种 0/1 时钟：费率 z 过阈值就反手，没有宇宙、没有「更拥挤的日子该不该第二天涨」这道题。

若有人把 `funding_rate_zscore_50` 当连续分数去挖 overnight IC，那是另一句，要另开目录。探照灯不能改题、不能当判决。结案只看三段五项 KPI。

---

## 验证

```bash
mlbot research run 20260910_funding_fade
mlbot research close 20260910_funding_fade
```

尺子事先写在纸上：任一段年化 **< 0**，或近窗 MaxDD **深于**熊段和牛段，原句就不成立。熔断关。只报五项 KPI。

### 三段结果（BTCUSDT · 2 小时 · 熔断关）

进场是 `|z| ≥ 1.5` 的反手。约 94% 的出场是 z 穿过 0。

| 段 | 年化 | Calmar | 胜率 | 最大回撤 | Sharpe(R) | 笔数 |
|---|---:|---:|---:|---:|---:|---:|
| `bear_2022` | +2.7% | 0.93 | 47.7% | −2.9% | 0.08 | 128 |
| `bull_2023_2024` | +3.0% | 0.60 | 50.7% | −4.9% | 0.13 | 75 |
| `recent_range_to_bear` | **−3.4%** | −0.50 | 45.9% | **−6.9%** | −0.11 | 109 |

| 事先写好的标准 | 数字有没有打中 |
|---|---|
| 任一段年化 < 0 | 打中。近窗 −3.4%。 |
| 近窗回撤深于两个趋势段 | 打中。−6.9% 深于熊段 −2.9% 和牛段 −4.9%。 |

近窗两条线都打中。趋势两段年化为正，不能拿来给近窗续命：尺子写的是「三段都不该做穿」，不是「趋势段绿了就算过」。

---

## 结论

分类：**拥挤回吐声称**。费率 z 是拥挤的测量，不是趋势因子。

- 熊段、牛段年化为小正，说明极端拥挤在趋势年里有时会回吐，但 Sharpe 只有 0.08 / 0.13，不是能当账户的 alpha。
- 近窗年化变负，回撤也更深：拥挤不再按纸上的方向回吐，或者回吐不够付摩擦和止损。
- 去 Top-3 单笔只作分类，不能单独用来否或救这条袖套。
- 不要在同一句上再拧「再加持仓量 / 再加费率水平」。那是另一张纸。

判决用 `mlbot research close 20260910_funding_fade --declare …`，不要手写 `verdict:`。不成立就机器不跑，也不要手盘「这回费率不一样」。

产物：

| 路径 | 是什么 |
|---|---|
| `results/funding_fade/experiments/20260910_funding_fade/funding_z15/bear_2022` | 熊段报告 |
| `results/funding_fade/experiments/20260910_funding_fade/funding_z15/bull_2023_2024` | 牛段报告 |
| `results/funding_fade/experiments/20260910_funding_fade/funding_z15/recent_range_to_bear` | 近窗报告 |
| [DECISION.md](../../config/experiments/20260910_funding_fade/DECISION.md) | 纸面原文 |

查谱系：

```bash
mlbot research index --trusted --query funding
```

---

## 报告解读

读三段表和 `DECISION.md` 时，按这几条，不要自己另立尺子。

1. **先对窗，再比年化。** 每一段只和自己的事先写好的标准比。不要拿熊段 +2.7% 去解释近窗 −3.4%，也不要把三段加成一条「全期还行」。
2. **标题是年化，不是 ΣR。** `pnl_r` / 合计 R 不是五项 KPI。Calmar = 年化 / |MaxDD|。Sharpe(R) 是按这笔交易的 R 倍数算的，不要把它读成日收益 Sharpe。
3. **胜率是交易笔的方向对不对，不是「跟上了牛市」。** 近窗胜率 45.9%，和熊段 47.7% 差不多，但年化已经翻负：亏的那几笔更深，或者回吐不够快。
4. **笔数是覆盖，不是置信徽章。** 近窗 109 笔并不比熊段 128 笔「更有资格推翻结论」。尺子看的是年化和回撤，不是样本够不够再扫一次阈值。
5. **约 94% 的出场是 z 回 0。** 这是合同，不是彩蛋。如果大多数出场其实是止损，那纸面上的「回到正常就走」就没被执行到，表就不能这么读。
6. **z 的 50 次是费率自己的节奏。** 不要把「50」读成 50 根 2 小时 K 线。窗口长度错了，拥挤的定义就换了，那是另一句。
7. **熔断关着比 edge。** 近窗亏了不能靠「要是当时熔断开着就不会亏」推翻结论。熔断是账户保护，不是这句话的一部分。
8. **正的趋势段不能单独过关。** 尺子写的是「任一段」。近窗已经死，整句就不能当钱用。

纸面原文在 [DECISION.md](../../config/experiments/20260910_funding_fade/DECISION.md)。
