# 特征计算

**English:** [features.en.md](features.en.md)  
**入口:** [../README_CN.md](../README_CN.md)

特征是**事先算好、按月落盘**的测量。回测和规则只读 FeatureStore，不在热路径现场 `compute_*`。  
闭棒：[lessons.md#closed-bar](lessons.md#closed-bar) · [math.md](math.md) §1。

---

## 这一面在量什么

| 名字 | 是什么 | 不是什么 |
|---|---|---|
| `*_f` | DAG 节点，写在 `config/feature_dependencies.yaml` | 规则里直接引用的列名 |
| 输出列 | 真正进 FeatureStore 的列（`output_columns`） | 常常比节点名少一个 `_f` |
| FeatureStore 层 | `feature_store/<layer>/<symbol>/<TF>/YYYY-MM.parquet` | 一次对话里手算的均线 |

例子：节点 `ema_50_200_cross_f` → 列 `ema_50_200_cross_side`。规则 YAML 写 `feature: ema_50_200_cross_side`。

```
成交 parquet（K 线；订单流还要 tick）
        │
        ▼
feature_dependencies.yaml（DAG：依赖、所需列、输出列）
        │
        ▼
月分区 FeatureStore（warmup 窗口 + 本月）
        │
        ▼
archetype YAML / event_backtest 只读已有列
```

缺列 → 先登记再 backfill，再跑尺子。不要从 `requested_features` 里删列来躲报错。

---

## 怎么用已经算好的列

### 1. 策略包点名节点

`config/strategies/<family>/features.yaml`：

```yaml
feature_pipeline:
  requested_features:
    - atr_f
    - ema_50_f
    - ema_200_value_f
    - ema_50_200_cross_f
```

公开练习包：`config/strategies/ma_cross/`。金叉方向读的是列 `ema_50_200_cross_side`。

### 2. 规则读输出列，不读节点名

```yaml
direction_rules:
  - id: ma_cross_ema50_200_side
    feature: ema_50_200_cross_side
    transform: sign
```

进 / 止 / 加 / 出写在 archetype YAML。FeatureStore 不签合同。

### 3. 建库

```bash
mlbot feature-store build \
  --config config/strategies/ma_cross \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 120T \
  --start-date 2022-01-01 --end-date 2026-06-01
```

指定已有层（加列时必带，见下文）：

```bash
PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/strategies/ma_cross \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 120T \
  --root feature_store \
  --layer features_ma_cross_120T_<hash> \
  --data-path data/parquet_data
```

### 4. 回测只读层

`event_backtest` / `mlbot research run` 走 `feature_store_strict=True`。层名在策略 `meta.yaml` 的 `feature_store_layer`。  
列不在盘上 → 停，去 backfill。不要在回测里补 `compute_*`。

### 5. 查有哪些计算函数

```bash
mlbot features list
mlbot features list --category orderflow
mlbot features list --search vpin
mlbot features count
```

权威清单是 DAG：`config/feature_dependencies.yaml`。注册表在 `src/features/registry.py`。

---

## 订单流

订单流用的是 **aggTrade tick**（`price` / `volume` / `side` + 时间索引），不是收盘价凑出来的假 delta。

| 你要 | 数据 | 节点例子 | 输出例子 |
|---|---|---|---|
| 知情交易概率 | tick | `vpin_features_f`、`vpin_base_aligned_features_f` | `vpin_*` |
| 买卖共识 | tick | `ofci_f`、`ofci_pct_f` | `ofci`、`ofci_pct` |
| 累积买卖差 | tick 或由其派生 | `cvd_basic_f`、`cvd_divergence_v2_f` | `cvd`、`cvd_change_*` |
| 单根足迹 | 根内 tick | `footprint_basic_f` | `fp_poc`、`fp_hvn`、`fp_delta_*` |
| 连续同向成交 | tick | trade clustering（`utils_order_flow_features`） | `trade_cluster_*` |

先拉成交再转 parquet：

```bash
mlbot data download --symbols BTCUSDT --start-year 2022 --start-month 1 \
  --end-year 2026 --end-month 6
mlbot data convert --symbols BTCUSDT
```

默认 tick 目录：`data/parquet_data/`。计算函数通过 `ticks_loader_json` / `ticks_dir` 按月加载，无 tick 的根写 NaN，不编造。

资金费率、持仓量是另一路 parquet（`mlbot data download-funding-rate` / `download-open-interest`），不是订单流，但同样先落盘再进 DAG。

订单流可以当 Gate（做 / 不做）。**不要**用慢数学量当硬否决主开关。分层：[math.md](math.md) §2。

---

## 数学特征

登记过的慢数学量（实现都在 `src/features/time_series/`）：

| 族 | 节点例子 | 回答的问题 | 不回答 |
|---|---|---|---|
| WPT | `wpt_price_fluctuation_f` | 去趋势后的波动成分 | 是不是突破 |
| Hilbert | `hilbert_features_f` | 包络、CVD/价格背离 | 开不开仓 |
| Hurst | `hurst_price_f`、`hurst_cvd_f` | 路径是趋势还是噪声 | 方向 |
| 频谱 | `spectrum_*_f` | 周期/能量 | 入场几何 |
| DTW | `dtw_features_f` | 和模板路径有多像 | 单独当 Gate |
| GARCH / EVT | `garch_features_f`、EVT 节点 | 波动聚集、尾部 | 合同几何 |

Hilbert 依赖 WPT 去趋势后的价格 / CVD 波动；Hurst 滚动窗口只用 `[t-W, t-1]`。  
原则：**数学不回答「是不是突破」，只回答「现在该有多小心」。** 适合 Execution（噪声惩罚、缩小暴露），不适合把历史失败压成 0 的 Gate。

收盘可知的技术量（ATR、RSI、EMA、ROC）也在同一张 DAG 里，只是便宜、不读 tick。公开 dummy 只用这一档。

---

## 怎么加快

特征级计算是**顺序**的。加速靠「少算」和「按月/按品种切开」，不靠把整表扔进进程池。

| 手段 | 做什么 | 什么时候 |
|---|---|---|
| 增量 merge | 已有月只算缺的 `*_f`，旧列原样留下 | **加一列的默认路径** |
| 同一 `--layer` | 不要为新列改层名 | 故意换层除外 |
| 跨层 donor | 缺月先从同品种、同时框的旧层拷过来，再只补新列 | 默认开；`--no-reuse` 关掉 |
| `--workers N` | 多品种并行（每品种一个子进程） | 2–8；不改变数值 |
| `FEATURE_MONTHLY_WORKERS` | 单品种内按月并行 | tick / DTW / Hilbert 很重时 |
| 月缓存 + warmup | 磁盘按月缓存；默认 `FEATURE_MONTHLY_WARMUP_MONTHS=3` | 避免月初 rolling 全是 NaN |
| 先收盘列 | EMA / ATR / 交叉只要 OHLC | 比 VPIN / footprint 快一个数量级 |
| 不要 `--force-rebuild` | 整层重算 | 只在层坏了、要扔掉 donor 时 |

```bash
# 多品种
mlbot feature-store build --config config/strategies/ma_cross \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT --timeframe 120T --workers 4

# 重特征：按月切开
FEATURE_MONTHLY_WORKERS=4 FEATURE_MONTHLY_WARMUP_MONTHS=3 \
  PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/strategies/ma_cross --symbols BTCUSDT --timeframe 120T \
  --layer features_ma_cross_120T_<hash>
```

`--workers` 和 `FEATURE_MONTHLY_WORKERS` 只切工作块，不改公式。

---

## 加一列，复用已经算好的列

不要为了新特征换层名。换层 = 整层重算。

1. 在 `src/features/` 写计算函数，用 `@register_feature` 登记（若还没有）。
2. 在 `config/feature_dependencies.yaml` 加 `*_f`：`compute_func`、`dependencies`、`required_columns`、`output_columns`。依赖列已经在层里就不用再算。
3. 把节点写进该策略 `features.yaml` 的 `requested_features`。
4. **带着原来的 `--layer` 再跑 build**（不要改名，不要 `--force-rebuild`）。

构建器按月看 meta 里已有列：

- 输出列都在 → 跳过这个月
- 缺某个 `*_f` 的输出 → `merge_mode`，只算缺的，写回同一份月 parquet（`+N features`）
- 这个层没有这一月、别的层有 → 先拷 donor，再只补缺口

DAG 会自动带上依赖。例如只点 `hilbert_features_f`，会先确保 `wpt_price_fluctuation` / `wpt_cvd_fluctuation` 在；它们已在层里就直接读。

```bash
# features.yaml 里刚加上 ema_50_200_cross_f
PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/strategies/ma_cross \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 120T \
  --root feature_store \
  --layer features_ma_cross_120T_<hash> \
  --data-path data/parquet_data
```

日志里应看到 `Incremental … +1 features (ema_50_200_cross_f)`，而不是整月重来。

收盘列（交叉、均线、ATR）增量很快。VPIN / footprint / DTW 仍然要读 tick 或长窗，但**旧的 OHLC 列不会重算**。

### 不要

- 为了躲 strict 读失败，从 `requested_features` 删列
- 回测热路径写 `_compute_*_column()`
- 没打算换数据集却改层名
- 用 `--force-rebuild` 当「加一列」
- 把标签（`forward_rr*`）当入场特征

---

## 相关

| 文 | 内容 |
|---|---|
| [usage.md](usage.md) | 下载、建库、回测命令 |
| [lessons.md#features](lessons.md#features) | FeatureStore-first |
| [math.md](math.md) | 闭棒；数学不作 Gate |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 尺子怎么接到回测 |
| `config/feature_dependencies.yaml` | DAG |
| `src/features/` | 计算函数 |
| `src/feature_store/` | 月分区读写 |

主入口：[README_CN.md](../README_CN.md)
