# 数据流与目录

**一句话：** 一条数据流，不是一堆互不相关的脚本。

## 流

```mermaid
flowchart LR
  raw[本机数据 tick/日线/费率] --> fs[特征库]
  fs --> yaml[实验包 YAML]
  yaml --> court[评测机]
  court --> kpi[五项 KPI]
  kpi --> v[人写判决]
```

## 目录对照（人话）

| 路径 | 干什么 |
|---|---|
| `data/` | 成交、费率、A 股日线等原始 / 转换结果 |
| `feature_store/` | 按月特征 parquet |
| `config/strategies/ma_cross/` | 公开练习策略形状 |
| `config/experiments/` | 某一句假设的实验包 |
| `src/features/` · `src/feature_store/` | 计算与读写 |
| `scripts/event_backtest/` | 事件评测机 |
| `src/cli/main.py` | `mlbot` 入口 |

本抽取：**只做验证**，不做自动挖因子，不接下单。执行层是以后的事；验证过的同一句规则可以再接上去。

## 还想看细则

- [架构 · 关键路径](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/ARCHITECTURE.md)
- [使用方法](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/usage.md)
