# `src/config` — strategy path helpers

Strategy **data** lives under `config/strategies/<slug>/`. This package only resolves and validates those trees.

| File | Role |
|------|------|
| `strategy_layout.py` | Path resolution and `extends` merge |
| `regime_layer.py` | Regime YAML helpers |
| `strategy_validation.py` | Required files on a strategy pack |

Public dummy: `config/strategies/ma_cross/`.
