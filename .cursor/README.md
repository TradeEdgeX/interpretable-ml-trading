# Cursor 规则与 skill

正文在 [`docs/`](../docs/README.md)。这里只放**短指针**，方便 Cursor 和其他工具在改文件时被提醒。

不要在 `.mdc` 里再写一套哲学或事故长文。改教训改 [`docs/lessons.md`](../docs/lessons.md)；改循环改 [`docs/hypothesis.md`](../docs/hypothesis.md)。

| 文件 | 指向 |
|---|---|
| `rules/rd-agent-playbook.mdc` | 假设循环、谁写 `verdict` |
| `rules/backtest-no-future-data.mdc` | [lessons.md#closed-bar](../docs/lessons.md#closed-bar) |
| `rules/feature-store-first.mdc` | FeatureStore，禁止回测里 `compute_*` |
| `rules/experiment-kill-switch-isolation.mdc` | [lessons.md#kill-switch](../docs/lessons.md#kill-switch) |
| `rules/report-kpi-cagr-not-sum-r.mdc` | [lessons.md#kpi](../docs/lessons.md#kpi) |
| `rules/strategy-classification-alpha-fattail-beta.mdc` | [lessons.md#classify](../docs/lessons.md#classify) |
| `rules/mlbot-data-download.mdc` | [usage.md](../docs/usage.md) |
| `rules/b-layer-strategy-archetype-paradigm.mdc` | YAML 层 |
| `rules/market-segment-calendars.mdc` | 三段日期 |
| `rules/mean-reversion-event-definition.mdc` | [lessons.md#mean-reversion](../docs/lessons.md#mean-reversion) |
| `rules/python*.mdc` · `pytest.mdc` · `fix-ci-*.mdc` | 代码风格 / 测试 |
| `skills/rd-experiment/SKILL.md` | 用户提出规则时走 court |

主入口：[README_CN.md](../README_CN.md)。文档目录：[docs/README.md](../docs/README.md)。
