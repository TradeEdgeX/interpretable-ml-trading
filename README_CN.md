# interpretable-ml-trading

**教学站（GitHub Pages）**：[https://tradeedgex.github.io/interpretable-ml-trading/](https://tradeedgex.github.io/interpretable-ml-trading/)（默认英文）  
中文：[https://tradeedgex.github.io/interpretable-ml-trading/zh/](https://tradeedgex.github.io/interpretable-ml-trading/zh/)

**English README**: [README.md](README.md)

本地也可 `cd website && mkdocs serve`。常识 → 特征 → 框架 → [Q & A](website/docs/zh/qa/index.md) → 金叉对照 → [已量展厅](website/docs/zh/gallery/index.md)。  
人话学习路径在 [`website/`](website/)；细则与 AI 仍翻 [`docs/`](docs/README.md)。

你提出一句交易假设，并写清什么时候做、做错了怎么办。  
打开这个仓库，跟电脑上的 AI 说话。

这个框架是给 AI 助手用的工具库。它帮你验证这句话成不成立，免得你对自己的想法过于乐观。它会写出详细报告，帮你改策略、收策略。排除一百种不成立的假设之后，你也许能找到一句用得上的。你怎么探索的、每次实验得出什么结论，框架也会帮你保管——那才是最值钱的知识。

模板：[docs/hypothesis_template.md](docs/hypothesis_template.md)。

验证过的同一套规则，以后可以接到通用执行层（计划用 Nautilus）。

---

## 完整例子：均线金叉

### 你对网上的 AI 说

> BTC 50 日线上金叉 200 日线做多，跌破 50 就走，能赚钱吗？

它通常会说：这是经典趋势策略，长期看有效，震荡市假信号多，最好再加点过滤。  
没有熊市 / 牛市 / 近三年的数字，也没有「错了算谁输」。换一个模型，答案会变。

### 你对本仓库的 AI 说同一句

> 价格在 50 日线上，金叉 200 日线就做多，跌破 50 就走。  
> 我觉得趋势年能赚钱，震荡年不该亏太多。测试一下这个策略。

你仍然是说话，不用先背命令。差别在 AI **必须走下面这条路**。

### 1. 先写成可验证的几条

| 格 | 这一句怎么写 |
|---|---|
| 机制 | 收盘价在 50 日线上方，且刚上穿 200 日线，才许开多。空头对称。 |
| 预期市况 | 单边年赚钱；震荡年回撤不该更深，折合成一年不该变亏。 |
| 进出场规则 | 跌破 50 日线就停。不加仓、不摊。没有另设止盈。 |
| 怎样算不成立 | 震荡段回撤深于趋势段，或任意一段折合成一年是亏的。 |
| 怎么用 | 机器和人手同一句。 |

| 模板格 | 这一句怎么写 |
|---|---|
| 社会学 | 教科书 50/200 是协调仪式。单边年追线的人付钱；震荡年他们互相付手续费。 |
| 数学 | 收盘才知道的 50 日线、200 日线变号；出场是收盘跌破 50 日线。 |
| 统计学 | 先当 beta / 趋势暴露，不是选点 alpha。三段分列。 |
| 验证标准 | 任一段年化为负，或震荡段回撤更深。 |
| 数据范围 | BTCUSDT · 2 小时 · `bear_2022` / `bull_2023_2024` / `recent_range_to_bear`。 |

网上的 AI 到这里往往停了，开始讲「再配合 RSI」。本仓库还要先写成[假设模板](docs/hypothesis_template.md)（谁付钱、量什么、怎么抽样、哪段数据、哪一项数字坏了就不成立），模板过了才许量。

### 2. 用的不是网页，是你机器上的数据

| 用什么 | 从哪来 | 网上的 AI 没有的 |
|---|---|---|
| K 线 | 你下载的币安成交，转成 parquet | 不是某篇复盘截图，也不是模型记忆里的价格 |
| 特征 | 特征库按月算好的 50 日线、200 日线、金叉 / 死叉事件（仓库里登记了一百多个这类特征） | 不是对话里随手算一根对不齐的均线 |
| 时钟 | 开盘做决定时，只许读 **上一根已经收盘** 的特征 | 网页回测经常把当根最高最低算进去，年化会假好看 |
| 用哪段日子来比 | 熊市 2022-01→2023-11、牛市 2023-06→2025-01、最近这段 2025-01→2026-05。比策略时关掉账户保护。只报折合成一年、Calmar、胜率、最大回撤、Sharpe | 不是「最近半年还行」 |

AI 缺列时要先补特征库，不许在回测里现场编一列。  
练习策略在 `config/strategies/ma_cross/`。金叉用的是特征 `ema_50_200_cross_*`，不是网页上的日线截图。

### 3. 本机跑出来的结论（BTC · 2 小时棒 · EMA50/200）

不是网页上那张日线图。进场是金叉 / 死叉，出场 100% 是收盘跌破 EMA50。数字来自本机成交和特征库，不是模型编的。

| 段 | 年化 | Calmar | 胜率 | 最大回撤 | Sharpe(R) | 笔数 |
|---|---|---|---|---|---|---|
| 熊市 2022 | +3.7% | 1.51 | 38.3% | −2.4% | 0.15 | 47 |
| 牛市 2023–2024 | +3.0% | 0.97 | 33.3% | −3.1% | 0.15 | 33 |
| 最近这段（震荡到熊） | −3.1% | −0.68 | 24.3% | −4.5% | −0.36 | 37 |

最近这段折合成一年是亏的。请你对照事先写好的标准，读完这张表和策略之后，自己写下结论。

网上的 AI 给不了这张表：它没有你的成交，也没有同一套比法。

### 4. 你还可以接着说

> 有没有人验过浅止损 −8%？  
> 先别回测，只建一个实验目录。  
> 这句不成立，记下来。

已经有结论的同一句，先复述，再决定要不要另写一句。  
你可以对本仓库里的 AI 说「测试一下这个策略」，然后才下载数据和跑回测。

### 更多例子

- 资金费率极端拥挤就反手：[docs/examples/20260910_funding_fade_CN.md](docs/examples/20260910_funding_fade_CN.md) · [EN](docs/examples/20260910_funding_fade.en.md)
- A 股周一跌、后四天涨：[docs/examples/20260911_ashare_monday_rebound_CN.md](docs/examples/20260911_ashare_monday_rebound_CN.md) · [EN](docs/examples/20260911_ashare_monday_rebound.en.md)
- BTC 大涨后 AI 山寨跟涨：[docs/examples/20260911_btc_lead_ai_alts_CN.md](docs/examples/20260911_btc_lead_ai_alts_CN.md) · [EN](docs/examples/20260911_btc_lead_ai_alts.en.md)
- P99 大单 + 布林上轨追涨：[docs/examples/20260911_p99_bb_break_chase_CN.md](docs/examples/20260911_p99_bb_break_chase_CN.md) · [EN](docs/examples/20260911_p99_bb_break_chase.en.md)
- 十倍股是不是小市值长持（cohort 已量）：[docs/examples/20260911_tenbagger_smallcap_CN.md](docs/examples/20260911_tenbagger_smallcap_CN.md) · [EN](docs/examples/20260911_tenbagger_smallcap.en.md)
- A 股涨得多又热，相对等权继续涨？（已 reject）：[docs/examples/20260911_ashare_cs_mom_amount_CN.md](docs/examples/20260911_ashare_cs_mom_amount_CN.md) · [EN](docs/examples/20260911_ashare_cs_mom_amount.en.md)
- A 股热板块相对等权是不是选板块 alpha？（已 reject）：[docs/examples/20260911_ashare_cs_sector_cost_CN.md](docs/examples/20260911_ashare_cs_sector_cost_CN.md) · [EN](docs/examples/20260911_ashare_cs_sector_cost.en.md)
- 买 ETF 会错过美股牛市吗（SPY/QQQ beta，已 reject）：[docs/examples/20260914_eq_us_spy_qqq_beta_CN.md](docs/examples/20260914_eq_us_spy_qqq_beta_CN.md) · [EN](docs/examples/20260914_eq_us_spy_qqq_beta.en.md)
- AI 算力开支过多时 BTC 是不是熊（芯片销售环比 vs 收盘才知道的 200 日线，已判定不成立）：[docs/examples/20260914_ai_chip_spend_btc_regime_CN.md](docs/examples/20260914_ai_chip_spend_btc_regime_CN.md) · [EN](docs/examples/20260914_ai_chip_spend_btc_regime.en.md)
- AI 融资公告后做多 BTC：[docs/examples/20260914_ai_financing_btc_CN.md](docs/examples/20260914_ai_financing_btc_CN.md) · [EN](docs/examples/20260914_ai_financing_btc.en.md)
- 买最热的股票，能跑赢「每人买一点」吗：[docs/cs_panel_CN.md](docs/cs_panel_CN.md) · [EN](docs/cs_panel.en.md)

上面金叉是第一幕：网上的 AI 讲故事，本仓库走模板、本机数据和同一套标准。其余例子都是**同一条故事换一句台词**——先讲「这一句在本仓库里怎么测完」，再按 **设计 / 数据 / 特征 / 相关扫描 / 验证 / 结论 / 报告解读** 七段展开。目录：[docs/README.md](docs/README.md) · 展厅：[website/docs/zh/gallery/index.md](website/docs/zh/gallery/index.md)。

---

## 装一次

给 AI 能在你电脑上拉数据、算特征、跑回测。装完继续说话就行。

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
```

拉数据和建特征库的命令在 [docs/usage.md](docs/usage.md)。  
特征怎么算、订单流 / 数学特征、加一列怎么复用旧层：[docs/features.md](docs/features.md)。  
平时仍是跟本仓库里的 AI 说「测试一下这个策略」，不用自己敲命令。

人和 AI、指令、命令怎么接成同一套标准，见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。  
技术栈、使用流程、周期 / 日历 / 品种怎么选（对照金叉和后面几个例子）：[docs/framework.md](docs/framework.md)。

---

想自己翻后文：[docs/README.md](docs/README.md)
