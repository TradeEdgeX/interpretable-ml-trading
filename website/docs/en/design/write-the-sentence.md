# Fill the template

**One-liner:** Turning a trading idea into a testable contract starts with filling the five boxes of `docs/hypothesis_template.md`. Measured examples are the same story: what a web AI would say → this repo must fill the template, check lineage, and wait for “measure this” before it downloads → the program prints a table → you declare. Then read the seven sections: Design, Data, Features, IC, Validation, Conclusion, How to read the report.

## The five boxes (into `DECISION.md`)

| Box | What to write | Example (MA golden cross) |
|---|---|---|
| Mechanism | Why this sentence makes money, and what must happen before you are allowed to trade | Trend continuation: close above the 50-day and just crossed above the 200-day |
| Expected regime | Which window should earn, which should lose or fail | Earns in trend; chop drawdown must not deepen; CAGR must not flip negative |
| Contract | Entry / side / exit written down hard; adds; kill switch | Exit on a break of the 50-day; no adds; kill switch off; closed-bar |
| Falsification | Which five-KPI in which window voids it | Any window with negative CAGR, or chop MaxDD deeper than trend |
| Landing | Which court; same sentence for the robot and for hands | `event_backtest` + layer `features_ma_cross_120T_<hash>` |

Empty five boxes look like a strategy brochure. You still need sociology / math / stats, plus which tape kills the sentence. Full grid: [hypothesis template](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/hypothesis_template.en.md).

## After it is measured, read the report in seven sections

| Section | You should be able to answer |
|---|---|
| Design | Is the court `event_backtest`, `cs_panel`, `cohort_hold`, or Phase 1? Were the five boxes and the falsifier written first? |
| Data | Symbol, bar, source, window dates. Is a missing year written “no sample”? Is the calendar from the wrong market? |
| Features | Do columns come from FeatureStore? Does the open read only the previous bar? Is there a `compute_*` in the backtest? |
| IC | If there is a table, it is a flashlight. If there is none, is this a single-name 0/1 clock that should not have one? |
| Validation | Is the control cash, equal-weight, or same-window buy-and-hold? Did the relative gap hit the falsifier? |
| Conclusion | Class first (alpha / fat-tail / beta / useless). Human `--declare`. Do not type `verdict:`. |
| How to read | CAGR is not ΣR; win rate is not “it caught the bull”; a shallower hole can be insurance; the recent window cannot promote alone. |

A finished seven-section walkthrough: [Does buying an ETF miss the US bull?](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/examples/20260914_eq_us_spy_qqq_beta.en.md). All measured sentences: [Gallery](../gallery/index.md).

## After filling it

Hand the filled template to the AI; it helps you translate it into YAML, run the backtest, and print the table. You read the table, check against the falsification line, and write your own conclusion. Until you say “measure this,” the AI may only validate the template, check lineage, and create a folder — no download, no backtest.

## Fine print

- [docs/hypothesis_template.en.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/hypothesis_template.en.md)
- [docs/hypothesis.en.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/hypothesis.en.md)
- [docs/examples/](https://github.com/TradeEdgeX/interpretable-ml-trading/tree/main/docs/examples)
