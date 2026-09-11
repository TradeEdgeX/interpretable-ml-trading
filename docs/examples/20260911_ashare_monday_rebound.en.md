# A-share Monday down → next four sessions up

Experiment: [config/experiments/20260911_ashare_monday_rebound/](../../config/experiments/20260911_ashare_monday_rebound/).

> After CSI 300 Monday close is down, go long and hold four trading days. Measure on the A-share calendar.

---

## Rule

| Piece | Choice |
|---|---|
| Sociology | Weekend bad news priced Monday; panic pays; later sessions may clear. |
| Math | Daily return; `monday_down` known at Monday close; hold 4 bars; **closed-bar**. |
| Stats | Calendar alpha claim. Segments from `market_segment_ashare.yaml` — **not** crypto `bear_2022`. |
| Falsifier | Any window CAGR < 0, or recent MaxDD deeper than both trend windows. |
| Data | `mlbot data download-ashare --symbols 000300.SH` → `data/ashare/daily/` |
| Layer | `features_ashare_monday_1D` (`weekday_monday_f`) |

```bash
PYTHONPATH=src python -m cli.main research validate 20260911_ashare_monday_rebound
PYTHONPATH=src python -m scripts.event_backtest --variant-grid \
  config/experiments/20260911_ashare_monday_rebound/monday_rebound_grid.yaml
```

---

## Local numbers (000300.SH · daily · kill switch off)

| Window | CAGR | Calmar | WR | MaxDD | Sharpe(R) | n |
|---|---:|---:|---:|---:|---:|---:|
| bear_2021 | +0.61% | 1.16 | 52.9% | −0.53% | 0.12 | 34 |
| bull_924 | +0.46% | 1.52 | 50.0% | −0.30% | 0.12 | 16 |
| chop_recent | +0.06% | 0.14 | 58.8% | −0.45% | 0.03 | 17 |

All three CAGRs positive; recent Calmar is much weaker than the trend windows. Declare with `--declare`.

中文：[20260911_ashare_monday_rebound_CN.md](20260911_ashare_monday_rebound_CN.md)
