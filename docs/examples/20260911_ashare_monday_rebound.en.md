# Example: A-share Monday down → next four sessions up

Landing: [README.md](../../README.md).  
Experiment: [config/experiments/20260911_ashare_monday_rebound/](../../config/experiments/20260911_ashare_monday_rebound/).  
Public family stays `ma_cross`. Do **not** edit `config/strategies/ma_cross/`.

---

## What you tell a web AI

> After a down Monday in A-shares, do the next four sessions tend to rebound?

It usually narrates weekend news and Monday panic. It has no A-share segment table and no falsifier.

---

## What you tell this repo

> After CSI 300 Monday close is down, go long and hold four trading days. Treat it as emotion clearing. Measure on the A-share calendar.

The AI must follow the court path below — no web essay as evidence.

---

## Template → validate → data → FeatureStore → three windows

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

All three CAGRs positive; recent Calmar is much weaker than the trend windows. **You declare.** Leave `verdict:` empty.

中文：[20260911_ashare_monday_rebound_CN.md](20260911_ashare_monday_rebound_CN.md)
