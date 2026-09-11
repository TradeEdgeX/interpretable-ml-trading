# Usage

**中文:** [usage.md](usage.md)  
**Landing:** [../README.md](../README.md)

Write the claim first: [hypothesis.en.md](hypothesis.en.md).  
This page is the command path. `mlbot --help` for subcommands.

---

## 1. Install

Python 3.12. Dependencies: `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
make install-hooks   # optional
mlbot --help
```

This extract runs locally. Do not hand-roll kline HTTP into `data/parquet_data`; use `mlbot data`.

---

## 2. Data → FeatureStore → backtest

```bash
mlbot data download --symbols BTCUSDT,ETHUSDT \
  --start-year 2022 --start-month 1 --end-year 2026 --end-month 6
mlbot data convert --symbols BTCUSDT,ETHUSDT

# A-share / index daily (court examples only; no Lab aux panels)
mlbot data download-ashare --symbols 000300.SH --start-date 2019-01-01
mlbot data download-ashare --universe listed,delisted --start-date 2016-01-01 --backend sina
# → data/ashare/daily/ ; calendar: config/market_segment_ashare.yaml

mlbot feature-store build \
  --config config/strategies/ma_cross \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 120T \
  --start-date 2022-01-01 --end-date 2026-06-01
```

If a column is missing, do not `compute_*` inside the backtest. Backfill FeatureStore first.  
How to compute, order-flow / math features, and add a column without rebuilding the layer: [features.en.md](features.en.md).

Universe YAML: `config/download/crypto_4h_token_universe_groups.yaml` (`mlbot data download --universe-config …`).

---

## 3. Research loop

Only after someone says “measure this.” How instructions map onto commands: [ARCHITECTURE.en.md](ARCHITECTURE.en.md).

```bash
mlbot research run <id>

PYTHONPATH=src python -m scripts.event_backtest \
  --strategy ma_cross \
  --symbols BTCUSDT,ETHUSDT \
  --start-date 2022-01-01 --end-date 2026-06-01 \
  --no-kill-switch
```

Read CAGR / MaxDD from `capital_report.json`. Do not conclude from summed R.  
Promotion: [LAYER_PROMOTION_CRITERIA.md](../config/experiments/LAYER_PROMOTION_CRITERIA.md)

---

## 4. Court

```bash
mlbot research harness ma_cross
mlbot research init 20260910_ma_cross_demo --strategy ma_cross
mlbot research close --all
```

Public dummy: `config/strategies/ma_cross/`. Example front-matter: `config/experiments/_examples/`.

Local experiment cards, Q&A, and `results/`:

```bash
mlbot lab
# http://127.0.0.1:8008/rd
```

No A-share / HK / crypto auxiliary panels, and no CMS.

---

## 5. Execution

Public execution will be Nautilus paper later. This tree has no homemade order manager, console, or production keys.

---

## 6. Where config lives

| Need | Path |
|---|---|
| Which packs ship | `config/strategies/STATUS.md` |
| Demo rules | `config/strategies/ma_cross/` |
| Court criteria | `config/experiments/LAYER_PROMOTION_CRITERIA.md` |
| Learning path | [README.md](README.md) |
| Lessons | [lessons.md](lessons.md) |

---

Home: [README.md](../README.md) · prev [Math](math.en.md) · next [Court](agent/rd_playbook.md)
