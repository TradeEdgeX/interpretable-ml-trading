# 文档目录

**人先读教学站** [`website/`](../website/)（GitHub Pages / `mkdocs serve`）。  
这里给 **AI**、以及想自己翻细则的人。

主入口在仓库根目录：[README_CN.md](../README_CN.md) · [README.md](../README.md)。  
完整例子（金叉：网页 AI vs 本机数据 + 特征 + 尺子）写在那里。平时跟 AI 说话就行。

每篇例子都是**同一条故事**换一句台词：网上的 AI 会讲什么、你对本仓库说同一句之后必须走哪几步、程序写出什么报告、人怎么写下成不成立。哲学在 [philosophy.md](philosophy.md)：这个框架帮你验证假设成不成立，优化的不是利润，是「这句话还成不成立」。用法在 [hypothesis.md](hypothesis.md) 和根目录 [README_CN.md](../README_CN.md) 的金叉对照。金叉是第一幕；下面每一篇都按「故事 → 七段表格」写完。

每篇七段能用表格的地方都用表格，不靠一句口号带过：

| 段 | 写什么 |
|---|---|
| 实验怎么设计的 | 人出句 → 模板 → 评测机；五格 + 模板格；信号在收盘 t 可知、仓位从 t+1 起 |
| 数据 | 品种、周期、来源、落盘、分窗起止、缺年就写无样本 |
| 特征 | 哪一列、怎么来、闭棒怎么用；缺列先建 FeatureStore，回测里不 `compute_*` |
| 相关扫描 | 有就算一张表，并写明只当探照灯；没有就写清为什么不该有 |
| 验证 | 事先写好的标准 + 五项数字（或该回测自己的尺子） |
| 结论 | 先说钱从哪来；人自己写下成不成立，不手写 `verdict:` |
| 报告解读 | 怎么读年化、胜率、在市、对照、缺样本；不要另立尺子 |

| 文 | 评测机 | 分类 | 状态 |
|---|---|---|---|
| [资金费率极端拥挤就反手](examples/20260910_funding_fade_CN.md) · [EN](examples/20260910_funding_fade.en.md) | 一根品种进出场 · BTC · 2 小时 | 拥挤回吐 | 最近这段折合成一年 −3.4%；结论留给你写 |
| [A 股周一跌、后四天涨](examples/20260911_ashare_monday_rebound_CN.md) · [EN](examples/20260911_ashare_monday_rebound.en.md) | 一根品种进出场 · 沪深300 · 日线 | 日历规律 | 三段折合成一年都接近零；结论留给你写 |
| [BTC 大涨后 AI 山寨跟涨](examples/20260911_btc_lead_ai_alts_CN.md) · [EN](examples/20260911_btc_lead_ai_alts.en.md) | 一根品种进出场 · 山寨 · 2 小时 | 跟着比特币走 | 熊市那段没有样本；最近这段 −1.52%；结论留给你写 |
| [P99 大单 + 布林上轨追涨](examples/20260911_p99_bb_break_chase_CN.md) · [EN](examples/20260911_p99_bb_break_chase.en.md) | 一根品种进出场 · BTC · 逐笔再合成 2 小时 | 少数很猛的一段 | 牛市那段 −0.25%；结论留给你写 |
| [十倍股 / 小市值长持](examples/20260911_tenbagger_smallcap_CN.md) · [EN](examples/20260911_tenbagger_smallcap.en.md) | 一季一季点名、拿满年数 | 小市值本身的暴露 | 十倍率两边都是 0.11%；看过表的人已判定不成立 |
| [买最热的股票，能跑赢「每人买一点」吗](cs_panel_CN.md) · [EN](cs_panel.en.md) | 每天买最热的，对照每人买一点 | 想法 → 报告 → 读表 | 两句都不成立，不是推荐策略 |
| [A 股动量+成交额](examples/20260911_ashare_cs_mom_amount_CN.md) · [EN](examples/20260911_ashare_cs_mom_amount.en.md) | 每天买最热的一档 | 声称会继续涨，量出来是回头 | 三段都输每人买一点；看过表的人已判定不成立 |
| [A 股热板块相对等权](examples/20260911_ashare_cs_sector_cost_CN.md) · [EN](examples/20260911_ashare_cs_sector_cost.en.md) | 先看热门板块再买票 | 板块轮动 | 牛市、震荡相对每人买一点更差；看过表的人已判定不成立 |
| [买 ETF 会错过美股牛市吗](examples/20260914_eq_us_spy_qqq_beta_CN.md) · [EN](examples/20260914_eq_us_spy_qqq_beta.en.md) | 美股日线、一直拿着对照 | 跟着大盘走 | 等超跌再买，折合成一年低于一直拿着；看过表的人已判定不成立 |
| [AI 算力开支过多时 BTC 是不是熊](examples/20260914_ai_chip_spend_btc_regime_CN.md) · [EN](examples/20260914_ai_chip_spend_btc_regime.en.md) | 只看同一天站在均线哪一侧 | 两件同时发生的市况 | 开支加速的日子，反而更常在均线上；看过表的人已判定不成立 |
| [AI 融资公告后做多 BTC](examples/20260914_ai_financing_btc_CN.md) · [EN](examples/20260914_ai_financing_btc.en.md) | 日线看过了；进出场还没跑 | beta | 日线上没有「该涨」；还不能下结论 |

其余细则：

| 文 | 内容 |
|---|---|
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

人话问答在教学站：[website/docs/zh/qa/index.md](../website/docs/zh/qa/index.md) · [EN](../website/docs/en/qa/index.md)。细则仍是 [agent/rd_qa.yaml](agent/rd_qa.yaml)。结案尺子：[LAYER_PROMOTION_CRITERIA.md](../config/experiments/LAYER_PROMOTION_CRITERIA.md)。
