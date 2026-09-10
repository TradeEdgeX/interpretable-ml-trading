# Public scope（抽取副本白名单）

本目录是开源抽取副本，不是现网。  
私有仓：`/home/yin/trading/ml_trading_bot`（现网 live **不要**在那边删）。

权威计划：[`docs/decisions/2026-09-10_opensource_nautilus_extract_CN.md`](docs/decisions/2026-09-10_opensource_nautilus_extract_CN.md)

---

## 公开（保留）

| 面 | 路径 |
|---|---|
| FeatureStore | `src/feature_store/` |
| 特征计算核 | `src/features/`（通用目录；现网探测器组合已拿掉） |
| Court | `src/research/` · `scripts/research/` |
| 规则引擎 | `src/time_series_model/` 中与 YAML 合同 / 事件回测相关的部分 |
| CLI | `src/cli/main.py`：`features` / `data` / `feature-store` / `research` |
| 事件回测 | `scripts/event_backtest.py` · `scripts/event_backtest/` · `scripts/rd_loop.py` · `scripts/build_feature_store_from_config.py` |
| 公开 dummy | `config/strategies/ma_cross/` |
| 市场阶段 | `config/market_segment.yaml` · `config/market_segment_crypto.yaml` |
| 特征 DAG | `config/feature_dependencies.yaml` |
| Court 示例 | `config/experiments/_examples/` · `_template/` · `LAYER_PROMOTION_CRITERIA.md` · 两个已量的例子 `20260910_ma50_ma200_cross` / `20260910_funding_fade` |
| 研究 constitution | `config/constitution/constitution.yaml`（只有 `ma_cross`，熔断关；不是现网账户文件） |
| 方法论 | 根目录 `README_CN.md` / `README.md`（主入口）· `docs/hypothesis.md` · `docs/lessons.md` · `docs/usage.md` · `docs/examples/` |
| 开源计划 | `docs/decisions/2026-09-10_opensource_nautilus_extract_CN.md` |
| 闭棒教训 | `docs/lessons.md` |
| 合同测试 | `tests/research/` · `tests/features/` · 闭棒 / `event_backtest` 合同测 |
| 纪律 | 见下方 cursor rules 白名单 |

### Cursor rules

短指针，正文在 `docs/`。清单见 [`.cursor/README.md`](.cursor/README.md)。

---

## 永不公开（本轮已删或保持不进树）

| 面 | 路径 / 例子 |
|---|---|
| 自制 OMS / 多账户 | `src/order_management/` · `src/live_data_stream/` |
| CMS | `src/mlbot_console/` · `frontend/` · `website/` |
| A 股 / 港股 / 辅助盘 | `src/ashare*` · `src/hk/` · `src/auxiliary/` · `lab/` |
| 现网 constitution | 私有仓那份真资金文件。本树只留研究用 `constitution.yaml` |
| 实验变体树 | 现网 `20*` 变体。本树只留 `_examples` / `_template` 与上面两个公开例子 |
| 非 dummy 策略包 | 滚仓 / 现货 / 网格 / TPC/BPC 等 |
| 运维 / 密钥面 | `live/` · `deploy/` · `docker/` |
| 个人文档 | 交易人格 / 收入决策 / 持仓笔记 |
| 现网 runbook | SSH、密钥隔离、CMS 写路径等 `.mdc` |

---

## 本轮之后还要做

- 把 `event_backtest` 从已删的 SRB / 现货 / constitution 模块解耦（现在仍有 import）
- 接 Nautilus paper
- 新 `git init` + secret scan 后再推 `TradeEdgeX/interpretable-ml-trading`
