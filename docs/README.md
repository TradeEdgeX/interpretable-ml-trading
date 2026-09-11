# 文档目录

主入口在仓库根目录：[README_CN.md](../README_CN.md) · [README.md](../README.md)。  
完整例子（金叉：网页 AI vs 本机数据 + 特征 + 尺子）写在那里。平时跟 AI 说话就行。这里是给 AI、以及想自己翻的人。

| 文 | 内容 |
|---|---|
| [资金费率极端拥挤就反手](examples/20260910_funding_fade_CN.md) · [EN](examples/20260910_funding_fade.en.md) | 费率 z 分数反手、回到 0 出场、三段数字 |
| [A 股周一跌、后四天涨](examples/20260911_ashare_monday_rebound_CN.md) · [EN](examples/20260911_ashare_monday_rebound.en.md) | 日线下载、A 股三段日历、日历 alpha |
| [BTC 大涨后 AI 山寨跟涨](examples/20260911_btc_lead_ai_alts_CN.md) · [EN](examples/20260911_btc_lead_ai_alts.en.md) | BTC 闭棒领涨 FS 列、beta 声称 |
| [P99 大单 + 布林上轨追涨](examples/20260911_p99_bb_break_chase_CN.md) · [EN](examples/20260911_p99_bb_break_chase.en.md) | tick P99 + bb_position、肥尾右尾 |
| [十倍股 / 小市值长持（cohort）](examples/20260911_tenbagger_smallcap_CN.md) · [EN](examples/20260911_tenbagger_smallcap.en.md) | 入场日小市值长持已量；未来谁十倍不能量 |
| [横截面多因子（cs_panel）](cs_panel_CN.md) · [EN](cs_panel.en.md) | 框架能力：日频打分 / 板块书；不是推荐策略 |
| [A 股动量+成交额横截面](examples/20260911_ashare_cs_mom_amount_CN.md) | 两因子锁死、相对等权失败、已 reject |
| [A 股热板块相对等权](examples/20260911_ashare_cs_sector_cost_CN.md) | 板块排序 + 10bp；相对等权不是 alpha；已 reject |
| [假设模板](hypothesis_template.md) · [EN](hypothesis_template.en.md) | 社会 / 数学 / 统计现象、验证标准、数据范围；AI 验模板再编排 |
| [验证假设](hypothesis.md) · [EN](hypothesis.en.md) | 怎么写成可验证的几条、结论怎么用 |
| [教训](lessons.md) | 闭棒、熔断、五项 KPI |
| [使用方法](usage.md) · [EN](usage.en.md) | 安装、数据、回测命令 |
| [特征计算](features.md) · [EN](features.en.md) | FeatureStore、订单流、数学特征、加速、加一列复用旧层 |
| [哲学](philosophy.md) · [EN](philosophy.en.md) | 为什么这样验 |
| [数学](math.md) · [EN](math.en.md) | 时钟和 KPI |
| [Court](agent/rd_playbook.md) | 谁写数字、谁宣判 |
| [框架：栈 / 流程 / 各层怎么选](framework.md) · [EN](framework.en.md) | 技术架构、使用流程、周期日历品种标准；对照已量例子 |
| [架构](ARCHITECTURE.md) · [EN](ARCHITECTURE.en.md) | 人说话 → 指令 → 命令 → 读报告；YAML 怎么接到回测 |

问答：[agent/rd_qa.yaml](agent/rd_qa.yaml)。结案尺子：[LAYER_PROMOTION_CRITERIA.md](../config/experiments/LAYER_PROMOTION_CRITERIA.md)。
