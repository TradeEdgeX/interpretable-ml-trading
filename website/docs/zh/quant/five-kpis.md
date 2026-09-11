# 五项 KPI

**一句话：** 口头和结案表只用年化、Calmar、胜率、最大回撤、Sharpe。不要用「合计 R」当头条。

## 五个词（人话）

| KPI | 人话 |
|---|---|
| 年化 | 照这句话做，账户一年大概长（或缩）多快 |
| Calmar | 涨得快不快，相对疼得深不深（年化 ÷ 回撤深度） |
| 胜率 | 做对的次数多不多——**赚钱不必胜率高** |
| 最大回撤 | 从高点算，最惨掉了多少 |
| Sharpe | 收益毛不毛（波动大不大） |

「合计 R / Total R」的分母会随权益变，长持肥尾会被摊薄，拿来比策略容易误判。

## 错 vs 对

| 错 | 对 |
|---|---|
| 「Total R = 120，很强。」 | 三段各报年化 / Calmar / 胜率 / MaxDD / Sharpe。 |

## 本仓库怎么用

金叉（BTC · 2h · 本机特征库；来源 [README_CN](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/README_CN.md)）：

| 段 | 年化 | Calmar | 胜率 | 最大回撤 | Sharpe |
|---|---|---|---|---|---|
| 熊市 2022 | +3.7% | 1.51 | 38.3% | −2.4% | 0.15 |
| 牛市 2023–2024 | +3.0% | 0.97 | 33.3% | −3.1% | 0.15 |
| 近窗 | −3.1% | −0.68 | 24.3% | −4.5% | −0.36 |

## 还想看细则

- [教训 · 五项 KPI](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/lessons.md#kpi)
