# ai_financing_btc archetypes

| File | Layer | This pack |
|---|---|---|
| `regime.yaml` | book open | `atr > 0` |
| `prefilter.yaml` | detector | `ai_financing_event >= 1` |
| `direction.yaml` | side | always long |
| `gate.yaml` | deny | empty |
| `entry_filters.yaml` | timing | empty |
| `execution.yaml` | harvest | 60 bars (5 UTC days), no adds |
