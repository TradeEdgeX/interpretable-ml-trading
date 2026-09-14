# 买 ETF 会错过美股牛市吗

实验：[config/experiments/20260914_eq_us_spy_qqq_beta/](../../config/experiments/20260914_eq_us_spy_qqq_beta/)  
先当 **beta**。评测机是 **eq_us_daily**，不是 2h `event_backtest`。人已 `--declare reject`。

人说的原句：买 ETF（SPY / QQQ）会错过美股牛市；要跟上爆发，得选股，或者等指数超跌再买。

---

## 规则

> 买入持有无杠杆 SPY 或 QQQ。对照是等 RSI / −20% / MA200 再进场。若择时全期年化高于同窗买入持有且回撤不更深，原句才还活着。

日历用 [`config/market_segment_us.yaml`](../../config/market_segment_us.yaml)。闭棒：信号在收盘 t 可知，仓位从 t+1 起。熔断关。

```bash
mlbot research validate 20260914_eq_us_spy_qqq_beta
mlbot data download-us --symbols SPY,QQQ --start-date 2013-01-01
PYTHONPATH=src python scripts/research/eq_us_spy_qqq.py
mlbot research close 20260914_eq_us_spy_qqq_beta --declare reject
```

本机带子是 Nasdaq 日线 2016-09-12 → 2026-09-11；法院锁到 2026-08-17（2,496 根）。Yahoo / Stooq 被拦。拆分红后收盘价，不是含分红 total return。标普点时选股未重跑（无宇宙）。

---

## 本机五项 KPI（熔断关）

买入持有：SPY 年化 **13.7%** / MaxDD **−34.1%** / Sharpe 0.80；QQQ 年化 **20.4%** / MaxDD **−35.6%** / Sharpe 0.93。

同窗择时（QQQ）：RSI≤30 持有 40 日 **10.6%**（在市 21%）；拿到回 252 日前高 **10.7%**；首次 −20% 持有 60 日 **8.4%**；MA200 **16.0%** / MaxDD −21.9%。SPY 四条择时年化都在 5.0%–7.8%，买入持有是 13.7%。

分窗买入持有：`us_bear_2022` SPY −31.0% / QQQ −42.5%；`us_bull_2023_2024` +24.2% / +39.3%；`us_recent` +19.0% / +25.0%。`us_covid_2020` 只有 24 根，年化不能当结案。

分类：**beta**。超跌择时年化全部低于同窗买入持有。MA200 是保险，不是 alpha。选股本机未重跑。

已 `--declare reject`。产物：`results/eq_us_spy_qqq/experiments/20260914_eq_us_spy_qqq_beta`

English: [20260914_eq_us_spy_qqq_beta.en.md](20260914_eq_us_spy_qqq_beta.en.md)
