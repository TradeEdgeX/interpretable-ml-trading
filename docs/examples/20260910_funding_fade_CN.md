# 资金费率极端拥挤就反手

实验：[config/experiments/20260910_funding_fade/](../../config/experiments/20260910_funding_fade/)  
家族是 `funding_fade`，结果在 `results/funding_fade/…`。不要改默认练习包 `config/strategies/ma_cross/`。

资金费率特别高就空、特别低就多，回到正常就走。费率本仓库能下（`mlbot data download-funding-rate`），特征库里已有 `funding_rate_zscore_50`，不必去买链上仪表盘。

---

## 规则

> 资金费率最近 50 次观察的 z 分数到 1.5 做空，到 −1.5 做多；回到 0 就走。熊 / 牛 / 近窗都不该把账户做穿。

| 格 | 内容 |
|---|---|
| 机制 | z ≥ 1.5 开空，z ≤ −1.5 开多（方向取反号）。 |
| 预期市况 | 极端拥挤该回吐；三段都不该做穿。 |
| 合同 | z 回到 0 就走（多头 z≥0，空头 z≤0）。不加仓、不摊、不跟踪止盈。 |
| 证伪 | 任一段年化为负，或近窗回撤深于两个趋势段。 |
| 落地 | 机器和人手同一句。 |

这 50 次是**费率自己的节奏**（大约每 8 小时一次，约 17 天），不是 50 根 2 小时 K 线。z 分数用中位数 / 绝对中位差，写在特征节点里，回测不要现场另算一列。

---

## 有没有判过

```bash
mlbot research index --trusted --query funding
```

公开仓库当时 0 条。没命中再开目录：

```bash
mlbot research init 20260910_funding_fade --strategy ma_cross
```

没说「测一下」就停在文件夹。这次人说了测，才往下走。

---

## 数据和特征

```bash
mlbot data download --symbols BTCUSDT \
  --start-year 2021 --start-month 6 --end-year 2026 --end-month 6
mlbot data convert --symbols BTCUSDT
mlbot data download-funding-rate --symbols BTCUSDT \
  --start-year 2020 --start-month 1 --end-year 2026 --end-month 6
```

费率落到 `data/funding_rate/parquet`。开盘决策只读上一根已收盘的特征。日历：熊 2022-01→2023-11、牛 2023-06→2025-01、近窗 2025-01→2026-05。熔断关。只报年化 / Calmar / 胜率 / 最大回撤 / Sharpe。

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

建在你自己的 `feature_store` 里。本机：61 个月、每月 27 列（只要 atr 和费率，层会瘦，正常）。2023-05 抽查：`funding_rate_zscore_50` 无空值，约 18% 的棒 `|z| ≥ 1.5`。

---

## 实验包

目录：`config/experiments/20260910_funding_fade/strategies/funding_fade/`

| 文件 | 写什么 |
|---|---|
| `features.yaml` | `atr_f`、`funding_rate_features_f` |
| `archetypes/prefilter.yaml` | `\|funding_rate_zscore_50\| ≥ 1.5` 才开 |
| `archetypes/direction.yaml` | 特征 `funding_rate_zscore_50`，`negate_sign` |
| `archetypes/execution.yaml` | `structural_exit: funding_zscore0`；关掉跟踪止盈、加仓 |
| `meta.yaml` | 特征库层 `features_funding_fade_120T` |

网格：`config/experiments/20260910_funding_fade/funding_fade_grid.yaml`（`strategies_root` 指到上面这个包）。

z 分数列本来就有。缺的是「回到 0 就走」这个出口，所以引擎加了 `structural_exit: funding_zscore0`。缺列补特征库，缺合同才改引擎。

---

## 三段结果（BTC · 2 小时 · 熔断关）

```bash
mlbot research run 20260910_funding_fade
mlbot research close 20260910_funding_fade
```

进场是 `|z| ≥ 1.5` 的反手。约 94% 的出场是 z 穿过 0。

| 段 | 年化 | Calmar | 胜率 | 最大回撤 | Sharpe(R) | 笔数 |
|---|---|---|---|---|---|---|
| 熊市 2022 | +2.7% | 0.93 | 47.7% | −2.9% | 0.08 | 128 |
| 牛市 2023–2024 | +3.0% | 0.60 | 50.7% | −4.9% | 0.13 | 75 |
| 近窗震荡到熊 | −3.4% | −0.50 | 45.9% | −6.9% | −0.11 | 109 |

近窗年化为负，回撤 (−6.9%) 也深于熊段和牛段。事先写的两条证伪线都打中了。

判决用 `mlbot research close 20260910_funding_fade --declare …`，不要手写 `verdict:`。不成立就机器不跑，也不要手盘「这回费率不一样」。不要为了再加持仓量过滤，在同一句上继续拧。

产物：

- `results/funding_fade/experiments/20260910_funding_fade/funding_z15/bear_2022`
- `results/funding_fade/experiments/20260910_funding_fade/funding_z15/bull_2023_2024`
- `results/funding_fade/experiments/20260910_funding_fade/funding_z15/recent_range_to_bear`
