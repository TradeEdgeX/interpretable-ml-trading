# 使用方法

**English:** [usage.en.md](usage.en.md)  
**入口:** [../README_CN.md](../README_CN.md)

先写假设，再跑命令：[hypothesis.md](hypothesis.md)。  
`mlbot --help` 看子命令。

---

## 1. 安装

Python 3.12。依赖在 `requirements.txt`。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
make install-hooks   # 可选
mlbot --help
```

本抽取副本在本地跑。不要手写 `requests` 去拉 kline 进 `data/parquet_data`，走 `mlbot data`。

---

## 2. 数据 → FeatureStore → 回测

```bash
mlbot data download --symbols BTCUSDT,ETHUSDT \
  --start-year 2022 --start-month 1 --end-year 2026 --end-month 6
mlbot data convert --symbols BTCUSDT,ETHUSDT

mlbot data download-funding-rate --symbols BTCUSDT \
  --start-year 2022 --start-month 1
mlbot data download-open-interest --symbols BTCUSDT \
  --start-year 2022 --start-month 1

mlbot feature-store build \
  --config config/strategies/ma_cross \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 120T \
  --start-date 2022-01-01 --end-date 2026-06-01
```

缺列不要在回测里 `compute_*` 兜底：先 backfill FeatureStore。  
怎么算、订单流 / 数学特征、加一列复用旧层：[features.md](features.md)。规则：`.cursor/rules/feature-store-first.mdc`。

宇宙配置：`config/download/crypto_4h_token_universe_groups.yaml`（`mlbot data download --universe-config …`）。

---

## 3. 研究闭环

人说「测一下」之后，AI 才许跑这些。命令和指令怎么接上，见 [ARCHITECTURE.md](ARCHITECTURE.md)。

```bash
mlbot research run <id>

PYTHONPATH=src python -m scripts.event_backtest --variant-grid \
  config/experiments/<dir>/*_grid.yaml

PYTHONPATH=src python -m scripts.event_backtest \
  --strategy ma_cross \
  --symbols BTCUSDT,ETHUSDT \
  --start-date 2022-01-01 --end-date 2026-06-01 \
  --no-kill-switch
```

看 `capital_report.json` 的年化 / MaxDD，不要把合计 R 当结论。  
Promote 合同：[LAYER_PROMOTION_CRITERIA.md](../config/experiments/LAYER_PROMOTION_CRITERIA.md) · 流程：[agent/rd_playbook.md](agent/rd_playbook.md)

---

## 4. Court

```bash
mlbot research harness ma_cross
mlbot research init 20260910_ma_cross_demo --strategy ma_cross
mlbot research close --all
```

公开 dummy：`config/strategies/ma_cross/`。示例头信息：`config/experiments/_examples/`。

本机看实验卡片、问答、`results/`：

```bash
mlbot lab
# http://127.0.0.1:8008/rd
```

不含 A股 / 港股 / 币圈辅助，也不含 CMS。

---

## 5. 执行

公开执行核日后接 Nautilus paper。本树不包含自制订单管理、控制台或生产密钥。

---

## 6. 配置从哪读

| 目的 | 路径 |
|---|---|
| 本树有哪些策略包 | `config/strategies/STATUS.md` |
| 演示规则 | `config/strategies/ma_cross/` |
| Court 准则 | `config/experiments/LAYER_PROMOTION_CRITERIA.md` |
| 学习路径 | [README.md](README.md) |
| 教训 | [lessons.md](lessons.md) |

---

主入口：[README_CN.md](../README_CN.md) · 上一篇 [数学](math.md) · 下一篇 [Court](agent/rd_playbook.md)
