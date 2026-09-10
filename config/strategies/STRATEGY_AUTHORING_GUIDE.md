# 策略编写规范

本抽取副本只公开 `ma_cross` 作为 court 形状 dummy。新规则先写成 YAML 合同，再用 `event_backtest` 证伪，不要先写私有执行引擎。

## 1. 文件结构

```
config/strategies/<name>/
├── meta.yaml
├── features.yaml
├── README.md
└── archetypes/
    ├── prefilter.yaml
    ├── gate.yaml
    ├── direction.yaml
    ├── entry_filters.yaml
    ├── regime.yaml
    └── execution.yaml
```

对照：`config/strategies/ma_cross/`。层职责见 `.cursor/rules/b-layer-strategy-archetype-paradigm.mdc`。

## 2. 特征

引用的列必须能从 FeatureStore 读到，并登记在 `config/feature_dependencies.yaml`。缺列先 backfill，不要在回测路径里现场 `compute_*`。

## 3. 怎么评

```bash
mlbot research init YYYYMMDD_<slug> --strategy ma_cross
python -m scripts.event_backtest --variant-grid config/experiments/<id>/*_grid.yaml
mlbot research close <id>
mlbot research close <id> --declare reject   # 人写 verdict
```

主 KPI：年化 / Calmar / 胜率 / MaxDD / Sharpe。不要把合计 R 当头条。
