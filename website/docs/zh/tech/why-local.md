# 为何本机

**一句话：** 所有数据和特征都在你机器上，换一个对话模型，表还是这张。证据可复查，结果可复现。

## 错 vs 对

| 错 | 对 |
|---|---|
| 「我问网页 AI，它说金叉经典趋势。」 | 「我机器上的特征库和成交印出这张表，金叉近窗年化 −3.1%。」 |

网页 AI 给你的是「它训练数据里的故事」，没法复查。本机给你的是「你下载的成交 + 你算的特征 + 锁死的评测机」印出来的表，可以一行一行复查。

## 本机意味着什么

- 数据：`data/parquet_data/` 里的按月 parquet。
- 特征：`feature_store/` 里的按月 parquet。
- 评测机：`scripts/` 里的锁死脚本。
- 结果：`results/` 里的 KPI 表和日志。

换一个 AI 模型，只要这些文件在，表还是这张。

## 还想看细则

- [数据流与目录](stack.md)
- [docs/philosophy.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/philosophy.md)
