# 三条杠（公开 court）

一条规则要被宣判「还没被证伪」，必须同时满足：

1. **同一把尺子**：`config/market_segment.yaml` 的三段（熊 2022 / 牛 2023–2024 / 近窗震荡到熊）。只看最近半年不算。
2. **熔断关掉**：比 edge 时 `kill_switch: false`。开着熔断会让各段在不同时刻停开，数字不能比。
3. **五项 KPI**：年化、Calmar、胜率、最大回撤、Sharpe。不要把合计 R 当头条。

分段同号、回撤不恶化，才谈得上 Pareto。任一证伪线被打中 → 句子不成立：机器不跑，人也不许手做同一句。

定性先于数字：趋势 / 肥尾收割 / Beta 增强，见 [docs/design/alpha_vs_fattail_vs_beta_CN.md](../../docs/design/alpha_vs_fattail_vs_beta_CN.md)。不要只用去 Top-3 或胜率杀掉趋势句。

`verdict` 只许人用 `mlbot research close <id> --declare` 写。
