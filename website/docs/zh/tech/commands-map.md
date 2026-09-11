# 命令地图

**一句话：** 命令是 AI 唯一许用的尺子，不是给人手抄的工具箱。平时说话即可。

## 四组

| 组 | 干什么 | 何时 |
|---|---|---|
| `mlbot research …` | 纸面：validate / index / init / close | 模板与结案；`--trusted` 只找已宣判的 |
| `mlbot data …` | 下载成交 / 费率 / A 股等 | 人说了「测一下」之后；禁止手写 kline |
| `mlbot feature-store …` | 建 / 补特征库 | 缺列才建；同层增量 |
| 评测机 | `research run` / `event_backtest` 等 | 人说了测；分窗、熔断关 |

没有 `mlbot train`。完整开关在仓库文档，站点不复制百科。

## 还想看细则

- [使用方法](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/usage.md)
- [架构 · 命令表](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/ARCHITECTURE.md)
