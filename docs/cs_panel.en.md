# Cross-section multi-factor (`cs_panel`)

**中文:** [cs_panel_CN.md](cs_panel_CN.md)

This repo can test **cross-section** claims (score a universe each day, long a quantile), not only single-name event clocks. The court is `cs_panel`, not `event_backtest`. `mlbot research run` does not dispatch it.

Capability is not edge. The measured A-share sentences are `--declare reject`.

- Rank locked columns at the **T close**; first fill is the **next open**; daily book return is next-open → next-next-open.
- Public dummy lock: `mom_20` and `amount_z_20`, equal-weight top 20% vs same-universe equal-weight.
- IC is a flashlight. The court is five KPIs on `bear_2021` / `bull_924` / `chop_recent`.
- Sector books: `scripts/research/cs_sector.py` (`daily` / `ls` / `weekly`). Industry map is a coarse snapshot, not PIT 申万.

```bash
PYTHONPATH=src python -m cli.main research harness ashare_cs_mom_amount
PYTHONPATH=src python scripts/research/cs_panel.py
```

Worked rejects: [hot names vs EW](examples/20260911_ashare_cs_mom_amount_CN.md), [hot sectors vs EW](examples/20260911_ashare_cs_sector_cost_CN.md).
