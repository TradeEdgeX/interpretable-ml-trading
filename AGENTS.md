# Agent entry (any tool)

This directory is the **open-source extract**, not the private live repo.
Read these before changing strategy or experiment files:

1. [`README_CN.md`](README_CN.md) / [`README.md`](README.md) — 主入口：这个仓库帮你验证策略
2. [`docs/hypothesis.md`](docs/hypothesis.md) — 怎么写成可验证的几条
3. [`docs/lessons.md`](docs/lessons.md) — 闭棒 / 熔断 / KPI
4. [`docs/agent/rd_playbook.md`](docs/agent/rd_playbook.md) — 谁写 `verdict`
5. [`docs/README.md`](docs/README.md) — 其余文档目录

```bash
mlbot research index --trusted --query <slug>
mlbot research harness <family>
mlbot research close --all
mlbot research close <id> --declare reject
```

Do not hand-edit `verdict`. Public dummy: `ma_cross`.
Do not start a backtest unless a human asked to measure that hypothesis.
Never push this tree’s existing git history to `TradeEdgeX/interpretable-ml-trading`.
