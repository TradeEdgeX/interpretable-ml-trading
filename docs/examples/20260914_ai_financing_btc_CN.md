# AI 融资公告和 BTC 的资金、涨跌

实验：[config/experiments/20260914_ai_financing_btc/](../../config/experiments/20260914_ai_financing_btc/)  
先当 **beta**（风险偏好外溢），不是选点 alpha。  
不是 [BTC 大涨后 AI 山寨跟涨](20260911_btc_lead_ai_alts_CN.md)，也不是 [资金费率极端拥挤就反手](20260910_funding_fade_CN.md)。

人说的原句：AI 的融资和 BTC 的资金和涨跌之间有某种关系。

---

## 规则

> 公开的大型 AI 公司融资公告日 D（UTC）结束后，次日做多 BTCUSDT，持有 5 个 UTC 日。熊 / 牛 / 近窗都不该把账户做穿。

| 格 | 内容 |
|---|---|
| 机制 | 锁定日历 `config/research/ai_financing_events.yaml`（≥ 3 亿美元）。D+1 第一根 2h 棒 `ai_financing_event = 1`。 |
| 预期 | 外溢年该赚；抽走现金买算力的年该失效。 |
| 合同 | 60 根 2h；不加仓；熔断关；闭棒。 |
| 证伪 | 任一段年化 &lt; 0，或近窗回撤深于两个趋势段。 |
| 落地 | 机器：本目录策略包。人手同一句。 |

公告日当天的路径不算入场信息。开盘决策只读上一根已收盘。

---

## 有没有判过

```bash
mlbot research validate 20260914_ai_financing_btc
mlbot research index --trusted --query ai-financing
```

模板已过。公开仓库 0 条已结案同一句。

---

## 数据和特征

日历是纸面锁死的公开大额轮（OpenAI / Anthropic / xAI / Inflection），不是 Crunchbase 全量。费率走 `mlbot data download-funding-rate`。Phase 1 日线来自 Binance Vision，写在 `data/klines_vision/`，不进 tick parquet。

| 列 | 是什么 |
|---|---|
| `ai_financing_event` | D+1 第一根棒为 1 |
| `ai_financing_in_window` | D+1 起 5 个 UTC 日 |
| `ai_financing_days_since` / `ai_financing_log_usd` | 距离与规模 |
| `funding_rate` / `funding_rate_zscore_50` | BTC 永续费率（已有节点） |
| `ai_basket_funding_zscore` | FET / RENDER / NEAR / TAO 费率 z 等权；只作探照灯 |

```bash
PYTHONPATH=src python scripts/research/ai_financing_scan.py
```

2h 法院还要建层再 `mlbot research run`：

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

## Phase 1 看见了什么（不能结案）

21 笔公告。窗口虚拟变量对 BTC 次日收益 Spearman IC = **−0.021**（p = 0.40）。日频上这句话没有条件期望。

| 段 | 事件 | 窗内均值 | 中位数 | 任意 5 日基线 | 怎么读 |
|---|---:|---:|---:|---:|---|
| bear_2022 | 7 | +1.12% | +0.97% | +0.01% | 相对基线有正差 |
| bull_2023_2024 | 9 | +0.04% | −0.24% | +1.24% | 跑输随便拿着 BTC |
| recent_range_to_bear | 7 | +0.25% | −0.76% | −0.15% | 均值被 2026-02-27 OpenAI +8.5% 拉开 |

`market_segment.yaml` 里熊段收到 2023-11、牛段从 2023-06 起，Inflection 2023-06-29 和 Anthropic 2023-09-25 进了两段。近窗 7 笔里 6 笔窗内收益为负。

资金费率：三段里公告后 8 小时费率均值都略高于公告前，量级是 10⁻⁵，不是拥挤 fade 那种 z 分数故事。

AI 叙事币篮子费率 z（滞后 1）对 BTC 费率 z：IC = **0.098**（p = 0.003）。两边拥挤同步。同一列对 BTC 次日收益 IC = 0.022（p = 0.43），带不来涨跌。

分类仍是 **beta**。Phase 1 不能 `--declare`。2h 五项 KPI 还没跑。

判决用 `mlbot research close 20260914_ai_financing_btc --declare …`，不要手写 `verdict:`。不要回测否了再手做「这回融资不一样」。

English: [20260914_ai_financing_btc.en.md](20260914_ai_financing_btc.en.md)
