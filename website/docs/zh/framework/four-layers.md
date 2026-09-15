# 本机四层

从原始成交到一张有五项数字的表，中间经过四层：数据 → 特征库 → 策略规则文件 → 回测脚本。每一层都有事先定好的格式。层与层之间只认文件，不认内存里的临时结果。

## 四层分别干什么

| 层 | 干什么 | 文件长什么样 |
|---|---|---|
| 数据 | 下载、清洗、按月存成文件 | `data/parquet_data/<SYMBOL>/<TF>/<YYYY-MM>.parquet` |
| 特征库 | 按月算好所有测量列 | `feature_store/features_<arch>_<TF>_<hash>/<SYMBOL>/<YYYY-MM>.parquet` |
| 策略规则 | 把进出场规则写成机器能读的文件 | `config/strategies/<archetype>/*.yaml` |
| 回测脚本 | 跑回测，印出三段、五项数字 | `scripts/event_backtest.py` 等 |

## 为什么要分层

层与层之间只认文件，不认内存。这样：

- 换一台机器，只要文件还在，结果就能复现。
- 改策略不用重算特征；改特征不用重新下载数据。
- AI 帮你改规则文件时，不会顺手改掉数据或特征。

## 在这个仓库里怎么用

你大多数时候只碰「策略规则」这一层。数据和特征库是基础设施。回测脚本是事先定好的比法，不要为了让某句话好看去改它。

## 还想看细则

- [数据流与目录](../tech/stack.md)
- [docs/framework.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/framework.md)
