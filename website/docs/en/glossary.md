# Glossary

For skimming. Detail pages are in the nav.

| English | 中文 | One-liner |
|---|---|---|
| Closed bar | 闭棒 | At bar open, only read the previous bar’s closed features |
| Three windows | 三段 / 三段日历 | Bear / bull / recent reported separately; recent alone cannot close the case |
| Five KPIs | 五项 KPI | CAGR, Calmar, win rate, max drawdown, Sharpe |
| Falsification | 证伪 | Written in advance: which KPI in which window voids this sentence |
| Kill switch | 熔断 | Off when comparing strategies; a separate on-run only looks at account protection |
| Feature | 特征 | A pre-computed, on-disk measurement column |
| Factor / regime | 因子 / 市况 | Which side of an exposure you stand on (a permission, not an entry) |
| Contract | 合同 | Entry / direction / exit written down hard |
| Feature store | 特征库 | Monthly parquet; at open only read the previous bar |
| Court | 评测机 | A locked backtest that only prints five KPIs by window |
| Gate | 门控 | A hard permission to trade or not |
| Cross-section | 横截面 | Scoring a basket every day, not a single-symbol event axis |
| Hypothesis validator | 假设验证器 | This repo’s role (the name is still interpretable-ml-trading): write ideas as contracts, print tables with the local ruler; ships a base feature library, you write the conclusion after reading the report |
