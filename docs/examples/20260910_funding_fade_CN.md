# 第二个例子：资金费率极端拥挤就反手

主入口仍是仓库根目录的 [README_CN.md](../../README_CN.md)。  
那一页用均线金叉把「跟网上的 AI 说话」和「跟本仓库的 AI 说话」对上。  
这一页把**另一句、完整能复查的过程**写下来：从一句话到五格、数据、特征库、YAML、三段法庭、数字。判决仍由人写。

实验目录：[config/experiments/20260910_funding_fade/](../../config/experiments/20260910_funding_fade/)  
公开练习家族仍是 `ma_cross`。**不要改**默认包 `config/strategies/ma_cross/`（那是远离均线死区 + ATR 跟踪，不是这句话）。

---

## 0. 为什么是这一句，不是链上仪表盘

有人会拿 Nansen / Glassnode / Dune 的截图当第一句。那些要密钥、不能在这个公开仓库里复现。

资金费率这个仓库**已经能下**：`mlbot data download-funding-rate`。特征库里已经登记了 `funding_rate_features_f` → `funding_rate_zscore_50`。所以第二句选它。

---

## 你对网上的 AI 说

> 资金费率特别高就空、特别低就多，回到正常就走。能赚钱吗？

它通常会说：拥挤回吐是经典逻辑，牛市费率长期为正所以空头容易挨打，最好再加点持仓量或价格过滤。  
没有熊市 / 牛市 / 近三年的数字，也没有「错了算谁输」。换一个模型，答案会变。

---

## 你对本仓库的 AI 说同一句

> 资金费率 50 次观察的 z 分数到 1.5 就做空，到 −1.5 就做多；回到 0 就走。  
> 我觉得拥挤该回吐，三段都不该把账户做穿。测一下。

你仍然是说话。差别在 AI **必须走下面这条路**，不许用网页文章当证据。

---

## 1. 先写成可验证的几条

| 格 | 这一句怎么写 |
|---|---|
| 机制 | 资金费率 50 次观察的稳健 z 分数 ≥ 1.5 开空；≤ −1.5 开多。方向取 z 分数的反号。 |
| 预期市况 | 费率极端拥挤的一段该回吐；熊 / 牛 / 近窗都不该把账户做穿。 |
| 合同 | z 分数回到 0 就走（多头 z≥0，空头 z≤0）。不加仓、不摊、不跟踪止盈。 |
| 证伪 | 任一段年化为负，或近窗回撤深于两个趋势段。 |
| 落地 | 机器和人手同一句。 |

50 次观察是**资金费率自己的节奏**（大约每 8 小时一次），大约 17 天，不是 50 根 2 小时棒。  
z 分数用中位数 / 绝对中位差，抗尖刺。这些已经写在特征节点里，回测不许再现场编一列。

网上的 AI 到这里往往停了，开始讲「再叠加持仓量」。本仓库到这里才开始量。

---

## 2. 先查有没有判过

```bash
mlbot research index --trusted --query funding
```

公开仓库当时 **0 条**已结案。没有命中，才允许建目录、再量。  
已经判过的同一句，复述结论，不要再扫一遍。

---

## 3. 建实验目录（还没下数据）

```bash
mlbot research init 20260910_funding_fade --strategy ma_cross
```

只建文件夹和 `DECISION.md` 壳。没让测，就停在这里。  
这次人说了「测一下」，才继续。

---

## 4. 用的不是网页，是你机器上的数据

| 用什么 | 从哪来 | 网上的 AI 没有的 |
|---|---|---|
| K 线 | `mlbot data download` + `convert`，成交转成 parquet | 不是某篇复盘截图 |
| 资金费率 | `mlbot data download-funding-rate`，落到 `data/funding_rate/parquet` | 不是仪表盘截图，也不要密钥 |
| 特征 | 特征库按月算好的 `funding_rate_zscore_50`、`atr` | 不是对话里随手算一个对不齐的 z |
| 时钟 | 开盘做决定时，只许读上一根已经收盘的特征 | 网页回测经常把当根最高最低算进去 |
| 尺子 | 熊市 2022-01→2023-11、牛市 2023-06→2025-01、近窗 2025-01→2026-05。熔断关掉。只报年化、Calmar、胜率、最大回撤、Sharpe | 不是「最近半年还行」 |

本机若已经有成交和资金费率，可以对着本仓库建一层**本地**特征库（不要写进别人的特征库目录）。公开用户按 [docs/usage.md](../usage.md) 自己下：

```bash
mlbot data download --symbols BTCUSDT \
  --start-year 2021 --start-month 6 --end-year 2026 --end-month 6
mlbot data convert --symbols BTCUSDT
mlbot data download-funding-rate --symbols BTCUSDT \
  --start-year 2020 --start-month 1 --end-year 2026 --end-month 6
```

---

## 5. 规则写成实验包，不改默认练习包

目录：`config/experiments/20260910_funding_fade/strategies/ma_cross/`

| 文件 | 写什么 |
|---|---|
| `features.yaml` | 只要 `atr_f` 和 `funding_rate_features_f` |
| `archetypes/prefilter.yaml` | `\|funding_rate_zscore_50\| ≥ 1.5` 才许开 |
| `archetypes/direction.yaml` | 特征 `funding_rate_zscore_50`，变换 `negate_sign`（高 z 做空） |
| `archetypes/execution.yaml` | `structural_exit: funding_zscore0`；关掉跟踪止盈、加仓、另设止盈 |
| `meta.yaml` | 钉死特征库层 `features_funding_fade_120T` |

默认 `config/strategies/ma_cross/` 原样不动。法庭网格用 `strategies_root` 指到实验包：

`config/experiments/20260910_funding_fade/funding_fade_grid.yaml`

---

## 6. 缺的不是特征列，是出场合同

`funding_rate_zscore_50` 特征库里早就有。  
这句话的合同是「回到 0 就走」，引擎原先没有这个出口。要量这一句，才在持仓逻辑里加上 `structural_exit: funding_zscore0`（多头 z≥0 走，空头 z≤0 走），回测把当根已收盘的 z 分数传进去。  
缺列时补特征库；缺合同才改引擎。不要在回测里 `compute_*` 现场编 z 分数。

---

## 7. 补特征库

```bash
PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/experiments/20260910_funding_fade/strategies/ma_cross \
  --symbols BTCUSDT \
  --timeframe 120T \
  --root feature_store \
  --layer features_funding_fade_120T \
  --data-path data/parquet_data \
  --start-date 2021-06-01 \
  --end-date 2026-06-01 \
  --no-reuse
```

本机结果：61 个月、每月 27 列（只要这两类特征，层会偏瘦，这是预期，不是坏了）。  
抽查 2023-05：`funding_rate_zscore_50` 无空值，约 18% 的棒 `|z| ≥ 1.5`。

---

## 8. 三段法庭（熔断关）

```bash
mlbot research run 20260910_funding_fade
mlbot research close 20260910_funding_fade
```

人再宣判时才加 `--declare`。程序和 AI 都不许手写 `verdict:`。

网格：BTC、2 小时、`--feature-store-strict --fast`、熔断关。

---

## 9. 本机跑出来的结论（BTC · 2 小时棒）

进场是 `|z| ≥ 1.5` 的反手。约 94% 的出场是 z 分数穿过 0。数字来自本机成交和特征库，不是模型编的。

| 段 | 年化 | Calmar | 胜率 | 最大回撤 | Sharpe(R) | 笔数 |
|---|---|---|---|---|---|---|
| 熊市 2022 | +2.7% | 0.93 | 47.7% | −2.9% | 0.08 | 128 |
| 牛市 2023–2024 | +3.0% | 0.60 | 50.7% | −4.9% | 0.13 | 75 |
| 近窗震荡到熊 | −3.4% | −0.50 | 45.9% | −6.9% | −0.11 | 109 |

近窗年化为负，且最大回撤 (−6.9%) 深于熊段 (−2.9%) 和牛段 (−4.9%)。  
按事先写好的证伪线，**两条都打中了**。

**判决你来写。** 不成立：机器不跑，你也不要晚上手盘「这回费率不一样」。

网上的 AI 给不了这张表：它没有你的成交，也没有这把尺子。

产物在：

- `results/ma_cross/experiments/20260910_funding_fade/funding_z15/bear_2022`
- `results/ma_cross/experiments/20260910_funding_fade/funding_z15/bull_2023_2024`
- `results/ma_cross/experiments/20260910_funding_fade/funding_z15/recent_range_to_bear`

---

## 10. 这一句量完了，还没宣判

`DECISION.md` 里的五项数字已经填上。`verdict` 仍空着。  
人若判不成立：`mlbot research close 20260910_funding_fade --declare reject`。  
不要为了「再加点持仓量过滤」在同一句上继续拧。要换句子，另开目录。
