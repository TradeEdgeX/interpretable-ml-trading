# 系统架构

> 研究核：FeatureStore → YAML 合同 → 事件回测 / court。不是下单说明书。

---

## 一页速览

- **核心原则**：启发式规则 + 历史证伪。不是 NN / 树模型交易机器人。
- **分层**：Data → FeatureStore（闭棒）→ 规则合同 → 事件回测 / court。
- **规则合同**：Prefilter → Direction → Gate → Entry → Execution → 意图。
- **Court**：人写假设，程序写五项 KPI，人宣判。用错 harness 或开着账户熔断比 edge = 不算结案。
- **执行**：公开仓日后接 Nautilus paper。自制多账户 OMS 不在本树。

---

## 研究管线

```
FeatureStore（开盘索引、收盘可知）
    │
    ├─ Prefilter     这类几何现在在不在
    ├─ Direction     +1 / -1 / 0
    ├─ Gate          hard deny / soft
    ├─ Entry Filter  时机（可空）
    ├─ Execution     止损 / 加仓 / 结构出场
    └─ 意图 → event_backtest（三段、熔断关）→ court
```

FeatureStore 只提供测量。规则写在 `config/strategies/<family>/archetypes/*.yaml`。

---

## 关键路径

| 路径 | 职责 |
|---|---|
| `src/feature_store/` | 月分区 parquet + meta |
| `src/features/` | 特征计算（日后瘦目录） |
| `config/feature_dependencies.yaml` | 特征 DAG |
| `src/research/` | court / gate / harness |
| `scripts/event_backtest/` | B 层事件回测 |
| `config/strategies/ma_cross/` | 公开 dummy |

---

## 研发 → 结案

```
假设 → Phase 1 扫描（可选）
     → event_backtest（三段，熔断关）
     → mlbot research close（程序写 gate_status）
     → 人 --declare
```

反模式：只看最近半年、用开着熔断的数字排名、open-bar 当证据、用事件回测给滚仓排名。

---

## 文档

| 区域 | 路径 |
|---|---|
| 学习路径 | [README.md](README.md) |
| 验证假设 / 教训 | [hypothesis.md](hypothesis.md) · [lessons.md](lessons.md) |
| 哲学 / 数学 / 使用 | [philosophy.md](philosophy.md) · [math.md](math.md) · [usage.md](usage.md) |
| Court | [agent/rd_playbook.md](agent/rd_playbook.md) |
| 定性 | [design/alpha_vs_fattail_vs_beta_CN.md](design/alpha_vs_fattail_vs_beta_CN.md) |

---

主入口：[README_CN.md](../README_CN.md) · 上一篇 [Court](agent/rd_playbook.md)
