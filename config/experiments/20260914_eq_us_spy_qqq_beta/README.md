# 20260914_eq_us_spy_qqq_beta

买 ETF（SPY / QQQ）会错过美股牛市？先当 **beta**。评测机 **eq_us_daily**。对照是指数超跌择时。已 `--declare reject`。

过程（设计 / 数据 / 特征 / IC / 验证 / 结论 / 报告解读）：

- [docs/examples/20260914_eq_us_spy_qqq_beta_CN.md](../../../docs/examples/20260914_eq_us_spy_qqq_beta_CN.md)
- [docs/examples/20260914_eq_us_spy_qqq_beta.en.md](../../../docs/examples/20260914_eq_us_spy_qqq_beta.en.md)

纸面：[DECISION.md](DECISION.md) · 数字副本：[result.json](result.json)

```bash
mlbot research validate 20260914_eq_us_spy_qqq_beta
mlbot data download-us --symbols SPY,QQQ --start-date 2013-01-01
PYTHONPATH=src python scripts/research/eq_us_spy_qqq.py
mlbot research close 20260914_eq_us_spy_qqq_beta --declare reject
```
