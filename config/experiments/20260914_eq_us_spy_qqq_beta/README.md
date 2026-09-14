# 20260914_eq_us_spy_qqq_beta

买 ETF（SPY / QQQ）会错过美股牛市？先当 **beta**。对照是指数超跌择时。已 `--declare reject`。

过程：[docs/examples/20260914_eq_us_spy_qqq_beta_CN.md](../../../docs/examples/20260914_eq_us_spy_qqq_beta_CN.md)

```bash
mlbot research validate 20260914_eq_us_spy_qqq_beta
mlbot data download-us --symbols SPY,QQQ --start-date 2013-01-01
PYTHONPATH=src python scripts/research/eq_us_spy_qqq.py
mlbot research close 20260914_eq_us_spy_qqq_beta --declare reject
```
