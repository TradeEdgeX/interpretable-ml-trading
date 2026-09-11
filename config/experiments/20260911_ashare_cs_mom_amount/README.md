# 20260911_ashare_cs_mom_amount

过去 20 日涨得多、成交额也热的股票，相对全市场等权继续涨。

评测机是 **cs_panel**（面板 IC + 前 20% 等权书），不是 `event_backtest`。

```bash
PYTHONPATH=src python -m cli.main research validate 20260911_ashare_cs_mom_amount
PYTHONPATH=src python scripts/research/cs_panel.py
```
