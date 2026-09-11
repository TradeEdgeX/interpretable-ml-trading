# {{TOPIC}}

## Checklist

| 步 | 命令 | 产物 |
|---|---|---|
| T | 填 [假设模板](../../../docs/hypothesis_template.md)；`mlbot research validate {{TOPIC}}` | 模板过关（不是判决） |
| 0 | `mlbot feature-store build`（人说了测、且缺列才补） | 特征库 |
| 1 | `mlbot research run {{TOPIC}}` | 三段五项 KPI |
| 2 | `mlbot research close {{TOPIC}}` + 人 `--declare` | `verdict` |

标定 segment: **{{SEGMENT}}**

See [`LAYER_PROMOTION_CRITERIA.md`](../LAYER_PROMOTION_CRITERIA.md).
