# ma_cross archetypes (public B-layer demo)

| File | Layer | This pack |
|---|---|---|
| `regime.yaml` | book open | `atr > 0` |
| `prefilter.yaml` | detector | `|ema_1200_position| >= 0.02` |
| `direction.yaml` | side | `sign(ema_1200_position)` |
| `gate.yaml` | deny | empty |
| `entry_filters.yaml` | timing | empty |
| `execution.yaml` | harvest | ATR trail, no adds |

Defaults are textbook, not locked promote numbers.
