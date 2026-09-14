# AI 融资公告和 BTC 的资金、涨跌

实验：[config/experiments/20260914_ai_financing_btc/](../../config/experiments/20260914_ai_financing_btc/)  
分类先当 **beta**（风险偏好外溢），不是选点 alpha。  
评测机计划是 **event_backtest**（BTCUSDT · 2 小时）。**现在只跑完 Phase 1 日线扫描，2 小时五项 KPI 还是空的，不能 `--declare`。**

> 公开的大型 AI 公司融资公告日 D（UTC）结束后，次日做多 BTCUSDT，持有 5 个 UTC 日。熊 / 牛 / 近窗都不该把账户做穿。

人最初说的是「AI 的融资和 BTC 的资金和涨跌之间有某种关系」。模板把它收成上面这句可执行的合同：锁定日历、D+1 进、持有 5 日。关系本身不是假设；假设是「公告后做多 BTC 在三段上都不该做穿」。

不是 [BTC 大涨后 AI 山寨跟涨](20260911_btc_lead_ai_alts_CN.md)，也不是 [资金费率极端拥挤就反手](20260910_funding_fade_CN.md)，更不是 [AI 算力开支过多时 BTC 是不是熊](20260914_ai_chip_spend_btc_regime_CN.md)。

English: [20260914_ai_financing_btc.en.md](20260914_ai_financing_btc.en.md)

---

## 这一句在本仓库里怎么走完

你对网上的 AI 说：「AI 的融资和 BTC 的资金、涨跌之间，是不是有某种关系？」

「有某种关系」不是假设。网上的 AI 会开始讲风险偏好外溢，或者建议你去扫 Crunchbase。本仓库要先收成一句**可执行的合同**：锁定日历上的公开大额轮，公告日 D 收盘后，次日做多 BTC，持有 5 个 UTC 日；三段都不该做穿。关系本身不进法院。

模板过了、谱系没有同一句，先做 Phase 1：日线事件窗和 IC。窗口虚拟变量对次日收益 IC = −0.021（p = 0.40）——日频上没有条件期望。牛段窗内跑输随便拿着 BTC；近窗均值被一笔 OpenAI +8.5% 拉开，中位数是负的。AI 叙事币和 BTC 的费率拥挤同步（IC 0.098），同步带不来涨跌。

**故事在这里停了一拍。** 证伪线写的是 2 小时书的年化和回撤。书还是空的。本仓库的用法是：探照灯再亮、再暗，都不能 `--declare`，也不能改题去做空公告。人还没说「把 2h 跑完」，AI 就不许假装结案。不要手做「这回融资不一样」。

下面七段把日历怎么锁、IC 怎么读、空表是什么意思写完。

---

## 实验怎么设计的

```text
人出句（融资和 BTC 资金、涨跌有关系）
  → 模板收成：公告日 D 结束后 D+1 做多 BTC，持有 5 个 UTC 日
  → validate + 谱系（公开库此前无已结案同一句）
  → Phase 1：锁定日历 vs 日线收益 / 费率（探照灯，不结案）
  → 2h 法院还没跑：建层 → mlbot research run → 人 --declare
```

Phase 1 看见 IC 或事件窗均值，**不能**改题，也**不能**宣判。结案只能等 2 小时书的五项 KPI。现在书是空的，所以这一页只解释探照灯怎么读。

| 格 | 这一句 |
|---|---|
| 机制 | 锁定日历 `config/research/ai_financing_events.yaml`（公开大额轮，默认 ≥ 3 亿美元）。D+1 第一根 2h 棒 `ai_financing_event = 1` 时做多 BTC。 |
| 预期市况 | 外溢年该赚；公告其实是抽走现金去买算力的年该失效。三段不该做穿。 |
| 合同 | 持有 5 个 UTC 日（60 根 2h）；不加仓；熔断关；闭棒。公告日当天的路径不算入场信息。 |
| 证伪 | 任一段年化 < 0，或近窗回撤深于两个趋势段。Phase 1 的 IC / 窗内均值不能当这条线。 |
| 落地 | 机器：本目录策略包。人手同一句：公告次日做多 BTC，五日后走。 |

| 模板格 | 这一句怎么写 |
|---|---|
| 社会学 | 付钱的是后知后觉的风险偏好盘。大额融资公告是协调仪式。若公告其实抽走现金买算力，付钱的会变成 BTC 多头自己。 |
| 数学 | 测量对象是锁死的公告日历，不是 Crunchbase 全量。开盘只读上一根已收盘。 |
| 统计学 | 先当 beta：公告上的风险偏好外溢。三段分列。 |
| 验证标准 | 2h 书任一段年化 < 0，或近窗回撤更深。 |
| 数据范围 | `BTCUSDT` · 2h 法院；Phase 1 用日线对齐公告日。币圈三段。 |

对照时钟（2h 法院，还没跑）：

| 时刻 | 发生什么 |
|---|---|
| 日历 | UTC 日 D 是公告日，当天收盘后才可用 |
| 进场 | D+1 第一根已收盘许可出现后做多 |
| 出场 | 再持有 60 根 2h（5 个 UTC 日） |
| 现金日 | 不在窗口里，收益记 0 |

---

## 数据

```bash
mlbot research validate 20260914_ai_financing_btc
mlbot research index --trusted --query ai-financing
mlbot data download-funding-rate --symbols BTCUSDT \
  --start-year 2020 --start-month 1 --end-year 2026 --end-month 6
```

| 项 | 本机事实 |
|---|---|
| 日历 | [`config/research/ai_financing_events.yaml`](../../config/research/ai_financing_events.yaml)，纸面锁死的公开大额轮（OpenAI / Anthropic / xAI / Inflection），不是 Crunchbase 全量 |
| 事件数 | 21 笔 |
| Phase 1 价格 | Binance Vision 日线，写在 `data/klines_vision/`，**不进** tick parquet |
| 2h 价格 | 法院要用 `data/parquet_data` 的成交转 parquet |
| 费率 | `mlbot data download-funding-rate` → `data/funding_rate/parquet` |
| 日历分段 | [`config/market_segment.yaml`](../../config/market_segment.yaml) |
| 2h 层 | 计划 `features_ai_financing_btc_120T`，**还没建完就不许 run** |

分窗（Phase 1 和未来 2h 书共用同一把日历）：

| 段 | 起止 | Phase 1 事件数 | 怎么读 |
|---|---|---:|---|
| `bear_2022` | 2022-01-01 → 2023-11-01 | 7 | 外溢若存在，熊段不该把账户做穿。 |
| `bull_2023_2024` | 2023-06-01 → 2025-01-01 | 9 | 风险偏好年该亮。 |
| `recent_range_to_bear` | 2025-01-01 → 2026-05-31 | 7 | 近窗。不能单独 promote。 |

`market_segment.yaml` 里熊段收到 2023-11、牛段从 2023-06 起，所以 **2023-06 → 2023-11 重叠**。Inflection 2023-06-29 和 Anthropic 2023-09-25 进了两段。读事件数时不要把 7+9+7 当成 23 个互不重复的公告。

---

## 特征

Phase 1 扫描脚本读日历和日线。2h 法院必须把同一套列建进 FeatureStore，不要在回测里 `compute_*`。

| 列 | 是什么 | 闭棒用法 |
|---|---|---|
| `ai_financing_event` | D+1 第一根 2h 棒为 1 | 当根收盘才知道「今天是窗口第一根」；下一根才许开 |
| `ai_financing_in_window` | D+1 起 5 个 UTC 日 | 窗口虚拟变量；Phase 1 用它对齐日线 |
| `ai_financing_days_since` | 距最近一笔公告多少日 | 只作描述，不改这一句 |
| `ai_financing_log_usd` | 公告规模的对数 | 只作描述，不按规模加权（加权是另一句） |
| `funding_rate` / `funding_rate_zscore_50` | BTC 永续费率（已有节点） | 开盘只读上一根 |
| `ai_basket_funding_zscore` | FET / RENDER / NEAR / TAO 费率 z 等权 | **只作探照灯**，不改「做多 BTC」这一句 |

```bash
PYTHONPATH=src python scripts/research/ai_financing_scan.py
```

2h 法院还要建层再跑（现在五项 KPI 仍空）：

```bash
PYTHONPATH=src python scripts/build_feature_store_from_config.py \
  --config config/experiments/20260914_ai_financing_btc/strategies/ai_financing_btc \
  --symbols BTCUSDT --timeframe 120T \
  --root feature_store --layer features_ai_financing_btc_120T \
  --data-path data/parquet_data \
  --start-date 2022-01-01 --end-date 2026-06-01
mlbot research run 20260914_ai_financing_btc
```

---

## IC

Phase 1 **有** IC，但它只是探照灯，**不能结案、不能改题**。

这里的 IC 不是横截面「高分名字 vs 低分名字」，而是时间序列：窗口虚拟变量（这天是否落在某笔公告后的 5 日窗里）对 BTC **次日**收益的 Spearman。n = 1611 个有标签的交易日。

| 分数 | 标签 | Spearman IC | p | 怎么读 |
|---|---|---:|---:|---|
| 5 日窗口虚拟变量 | BTC 次日收益 | **−0.021** | 0.40 | 没有条件期望。公告窗并不预示第二天涨。 |
| AI 篮子费率 z（滞后 1） | BTC 费率 z | **0.098** | 0.003 | 两边拥挤同步。这是市况共存，不是交易信号。 |
| AI 篮子费率 z（滞后 1） | BTC 次日收益 | 0.022 | 0.43 | 同步的拥挤带不来涨跌。 |

IC 为负或接近零，只能说「日频上这句话没有亮」。它不能代替 2h 书，也不能让人改成「那我们做空公告」。做空是另一句。

---

## 验证

Phase 1 已经跑完扫描。2h 五项 KPI **还没跑**，所以验证只停在探照灯。

### Phase 1 事件窗（闭日历 D+1 → D+5，不能结案）

| 段 | 事件 | 窗内均值 | 中位数 | 日胜率 | 任意 5 日基线 | 费率前 / 后（8h 均值） |
|---|---:|---:|---:|---:|---:|---:|
| `bear_2022` | 7 | +1.12% | +0.97% | 71% | +0.01% | 6.3×10⁻⁵ / 6.8×10⁻⁵ |
| `bull_2023_2024` | 9 | +0.04% | −0.24% | 44% | +1.24% | 9.4×10⁻⁵ / 11.7×10⁻⁵ |
| `recent_range_to_bear` | 7 | +0.25% | −0.76% | 14% | −0.15% | 2.2×10⁻⁵ / 2.4×10⁻⁵ |

| 段 | 怎么读这一行 |
|---|---|
| 熊段 | 相对「随便拿 5 日」的基线有正差，中位数也是正的。探照灯在熊段没有反向，但 7 笔不能结案。 |
| 牛段 | 窗内均值几乎走平，中位数为负，**跑输随便拿着 BTC**（基线 +1.24%）。风险偏好年该亮，探照灯却灭了。 |
| 近窗 | 均值被 2026-02-27 OpenAI 一笔 **+8.5%** 拉开；中位数 −0.76%，7 笔里 6 笔窗内收益为负。均值在撒谎。 |

资金费率：三段里公告后 8 小时费率均值都略高于公告前，量级是 10⁻⁵，不是拥挤 fade 那种 z 分数故事。费率没有讲「公告把永续挤到极端」。

### 2 小时法院（空表，等 `mlbot research run`）

| 段 | 年化 | Calmar | 胜率 | MaxDD | Sharpe |
|---|---:|---:|---:|---:|---:|
| `bear_2022` | （未跑） | — | — | — | — |
| `bull_2023_2024` | （未跑） | — | — | — | — |
| `recent_range_to_bear` | （未跑） | — | — | — | — |

空表的意思是：人还没有说「把 2h 书跑完」。没有五项 KPI，就不能 `--declare`。

---

## 结论

分类仍是 **beta**。Phase 1 只能说：日频上，锁定日历并没有给出「公告后 BTC 该涨」的条件期望。

- 窗口虚拟变量对次日收益 IC ≈ 0，p = 0.40。
- 牛段跑输随便持有 BTC；近窗中位数为负，均值被单笔拉开。
- AI 叙事币和 BTC 的费率拥挤同步（IC 0.098），但同步带不来涨跌。
- **这些都不能宣判。** 证伪线写的是 2h 年化和回撤。书还是空的。

判决用 `mlbot research close 20260914_ai_financing_btc --declare …`，且必须等 2h 书出来。不要手写 `verdict:`。不要回测还没跑就手做「这回融资不一样」。

产物：

| 路径 | 是什么 |
|---|---|
| `config/experiments/20260914_ai_financing_btc/quick_scan/ai_financing_scan.json` | Phase 1 扫描 |
| [DECISION.md](../../config/experiments/20260914_ai_financing_btc/DECISION.md) | 纸面原文 |
| 2h `results/…` | **还没有** |

---

## 报告解读

1. **先问这张表是不是法院。** `ai_financing_scan.json` 是探照灯。空着的五项 KPI 表才是法院。不要把窗内均值当成年化。
2. **先看中位数，再看均值。** 近窗均值 +0.25%、中位数 −0.76%。差出来的那一截就是 2026-02-27 的 +8.5%。去 Top-3 只分类；这里连「先有稳定窗口」都没有。
3. **基线是「任意 5 日」，不是现金。** 牛段窗内 +0.04% 对基线 +1.24%，读成「公告后拿 BTC 还不如随便拿」。这已经在打原句的社会学：外溢年该赚。
4. **重叠日历不要重复数事件。** 两笔公告进了熊、牛两段。7+9+7 不是 23 个独立实验。
5. **费率前 / 后不是 fade。** 10⁻⁵ 量级的差值，说明公告没有把永续挤到 `|z| ≥ 1.5`。不要把本页和 [费率反手](20260910_funding_fade_CN.md) 并成一句。
6. **篮子费率 IC 0.098 是共存。** AI 币和 BTC 一起拥挤，不能推出「该做多 BTC」。同一列对次日收益 IC 是 0。
7. **IC 不能改题。** 探照灯指向「没有条件期望」，不能因此把原句翻成做空，也不能把日历改成 Crunchbase 全量再扫一轮。
8. **人还没 `--declare`。** 空 `verdict` 不是「还行」。它只表示 2h 书没跑完。

纸面原文在 [DECISION.md](../../config/experiments/20260914_ai_financing_btc/DECISION.md)。
