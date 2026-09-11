# 本机四层

**一句话：** 从原始成交到一张 KPI 表，中间经过四层：数据 → 特征库 → 策略 YAML → 评测机。每一层都有锁死的格式，层与层之间只认文件。

## 四层速览

| 层 | 干什么 | 锁死的格式 |
|---|---|---|
| 数据 | 下载、清洗、按月 parquet | `data/parquet_data/<SYMBOL>/<TF>/<YYYY-MM>.parquet` |
| 特征库 | 按月算好所有测量列 | `feature_store/features_<arch>_<TF>_<hash>/<SYMBOL>/<YYYY-MM>.parquet` |
| 策略 YAML | 把合同写成机器可读的规则 | `config/strategies/<archetype>/*.yaml` |
| 评测机 | 跑回测，印五项 KPI 分窗表 | `scripts/event_backtest.py` 等 |

## 为什么要分层

层与层之间只认文件，不认内存。这样：

- 换一台机器，只要文件在，结果就能复现。
- 改策略不用重算特征；改特征不用重下数据。
- AI 帮你改 YAML 时，不会顺手改掉数据或特征。

## 本仓库怎么用

你大多数时候只碰「策略 YAML」这一层。数据和特征库是基础设施，评测机是锁死的尺子。

## 还想看细则

- [数据流与目录](../tech/stack.md)
- [docs/framework.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/framework.md)
