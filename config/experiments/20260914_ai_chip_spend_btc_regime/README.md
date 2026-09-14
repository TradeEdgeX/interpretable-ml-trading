# 20260914_ai_chip_spend_btc_regime

Epoch 芯片销售季频环比过高时，BTC 日线闭棒更常低于 MA200。  
先当 **beta / 市况**，不是选点。Y 不是 `market_segment.yaml` 标签。已 `--declare reject`。

过程：[docs/examples/20260914_ai_chip_spend_btc_regime_CN.md](../../../docs/examples/20260914_ai_chip_spend_btc_regime_CN.md)

```bash
mlbot research validate 20260914_ai_chip_spend_btc_regime
mlbot research index --trusted --query ai-chip-spend
PYTHONPATH=src python scripts/research/ai_chip_spend_regime_scan.py
PYTHONPATH=src python scripts/research/close.py 20260914_ai_chip_spend_btc_regime --declare reject
```
