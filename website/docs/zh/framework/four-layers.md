# 本机四层

**一句话：** 测量锁在本机四层上——换一个 AI 模型也不能改数字。

## 四层

```mermaid
flowchart LR
  data[数据文件] --> fs[特征库]
  fs --> yaml[合同 YAML]
  yaml --> court[评测机]
  court --> kpi[五项 KPI]
  kpi --> human[人宣判]
```

| 层 | 锁什么 | 不锁什么 |
|---|---|---|
| **数据** | 本机可复查的文件；禁止手写 kline | 必须是币安 tick——日线、费率、指数都可以，**和数学对口即可** |
| **特征** | 登记过的闭棒列；缺列先补 | 必须用满某一层全部 100+ 列 |
| **合同** | 进 / 向 / 出写在实验 YAML | 必须是均线家族——公开 `ma_cross` 只是 YAML 形状 |
| **评测机** | 分窗、熔断关、五项 KPI；程序不写判决 | 必须用币圈 `bear_2022`——日历跟市场走 |

## 错 vs 对

| 错 | 对 |
|---|---|
| 用网页文章和聊天记忆当下证据。 | 同一句话、同一把尺子、本机文件可复查。 |

## 还想看细则

- [框架](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/framework.md)
- [架构](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/ARCHITECTURE.md)
