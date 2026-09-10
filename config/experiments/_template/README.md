# {{TOPIC}}

## Phase checklist

| Phase | 命令 | 产物 |
|-------|------|------|
| 0 | FeatureStore backfill | layer parquet |
| 1 | `rd_loop` phase1 yaml | `quick_scan/` |
| 2 | 人读 scan → `DECISION.md` | 阈值 |
| 3 | `event_backtest --variant-grid`（kill switch OFF） | 分段 KPI |
| 4 | `mlbot research close` + 人 `--declare` | `verdict` |

## Phase 1

```bash
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/{{TOPIC}}/rd_loop_{{TOPIC}}_phase1.yaml
```

标定 segment: **{{SEGMENT}}**

See [`LAYER_PROMOTION_CRITERIA.md`](../LAYER_PROMOTION_CRITERIA.md).
