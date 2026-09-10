# 开源抽取计划（court + 闭棒特征 + Nautilus）

**DATE:** 2026-09-10  
**STATUS:** 抽取副本已 `git init` 空历史 · 尚未推公开仓  
**取代：** 2026-08-18 自制 testnet `OrderManager` 方案 → 公开执行核改 Nautilus paper  
**清单：** 仓库根 [`PUBLIC_SCOPE.md`](../../PUBLIC_SCOPE.md)

私有仓（不要在本目录操作）：`/home/yin/trading/ml_trading_bot`  
本目录只是抽取副本：`/home/yin/trading/ml_trading_bot_open_source`

拟公开 remote：`git@github.com:TradeEdgeX/interpretable-ml-trading.git`  
**禁止**把本副本或私有仓的现有 git 历史推到该 remote。

---

## 0. 一句话

开源的是**如何雕刻和证伪规则**（闭棒特征 + 注册考试卷 + 人宣判）。  
执行可选，接 **Nautilus paper**。  
不是如何管自己的钱，也不是可解释机器学习交易机器人。

---

## 1. 值不值得开源

**值，但价值不在策略参数或实盘曲线。** 社区里已有很多能下单的 bot。少见的是：

| 真正稀缺 | 本仓已有 |
|---|---|
| 怎么证明一条规则不是数据挖掘 | Phase 1 IC → Phase 3 三段回测 → 三条杠；比 edge 时关账户熔断 |
| 怎么给策略定性 | Alpha / 肥尾收割 / Beta 增强；探测器 ≠ 收割器 |
| 回测怎么不偷未来 | FeatureStore 闭棒、timeline SEVERE、closed-bar 合同 |
| 负结果档案 | 浅止损、行业硬帽、币本位滚仓、open-bar 网格均有结案 |

自制 live / CMS / 多账户 OMS 质量不差，但是**个人运维**，绑死数据库与控制台，公开没有复用价值。

整库原样公开几乎没价值。公开仓必须是**新仓 + 干净历史**。

---

## 2. 两个仓

| 仓 | 职责 | 公开？ |
|---|---|---|
| **私有 prod**（`ml_trading_bot`） | 真密钥、constitution、OMS、CMS、A 股 `trade_log`、deploy | 永远不公开；**现网 live 先不删** |
| **公开核**（本抽取目录 → 新 git） | FeatureStore + court + `ma_cross` + 闭棒测试 + 日后 Nautilus paper | 开源 |

按「可给陌生人」切，不要按「研究 / 执行」横切。

---

## 3. 分层（公开什么）

| 层级 | 东西 | 公开？ |
|---|---|---|
| **S** | Research court（人写假设 / 程序写数字 / 人宣判、harness 合同） | 是 |
| **S** | 闭棒契约 + 未来函数合同测试 | 是 |
| **S** | 三条杠 + Alpha / 肥尾 / Beta + 精选负结果 | 是 |
| **A** | FeatureStore（时钟、DAG、增量 backfill） | 是（目录日后瘦身） |
| **A** | `GenericLiveStrategy` 规则合同 | 是（发出意图，不自制下单） |
| **A** | `event_backtest` + 公开 dummy `ma_cross` | 是 |
| **B** | 语义特征（箱子 / 支撑阻力） | 日后精选，本轮先留计算核 |
| **C** | 整本特征海选、自制 OMS、CMS | 否 |
| **F** | 实验变体树、个人持仓、密钥、constitution 真资金 | 否 |

---

## 4. 实盘：公开仓用 Nautilus，私有仓先不动

```text
FeatureStore / 闭棒 bar
    → GenericLiveStrategy（YAML 合同）
    → TradeIntent
    → Nautilus Strategy / Actor（paper 或 testnet）
```

Nautilus **替代不了**：闭棒时钟、court / harness 合同、账户熔断「挡新开、允许减仓」。  
这些做成适配器与短文，不要把币安多账户补丁堆进公开仓。

**私有现网 OMS 等 Nautilus paper 在 SRB 上跑通意图→成交→对账再迁。**  
本抽取副本删除自制 live / CMS，不等于拆私有仓。

---

## 5. 分阶段

### Phase 0 — 本抽取副本清理（本轮）

- 写下本计划 + `PUBLIC_SCOPE.md`
- 删除私有代码、文档、实验变体树、账号数据、运维面
- 卸掉指向私有 GitHub 的 remote，防止误推
- 私有仓 `ml_trading_bot` 一行不删

### Phase 1 — 最小核能 2 小时跑通

```text
下载 BTC → 建 FeatureStore → 跑 ma_cross 事件回测（三段、熔断关）
→ 五项 KPI → mlbot research close
```

验收：干净机器 `pip install -e .` 后一条命令出主 KPI 表。过不了不推 GitHub。

### Phase 2 — Nautilus paper（Phase 1 绿了再做）

一条 compose：特征发布 + Nautilus paper。不要 systemd / AWS / CMS。

### Phase 3 — 文档按陌生人重写

一页架构、证伪过什么、License + 免责。  
仓库名 `interpretable-ml-trading` 易被读成 SHAP / 树模型；README 第一句必须否定 ML bot。

### Phase 4 — 开源后

Issue 只收：复现失败 / 未来函数 / 文档错。不收「帮我改阈值」。

---

## 6. 本轮明确不做

- 不把本目录现有 git 历史推到 `TradeEdgeX/interpretable-ml-trading`
- 不在私有仓删除 live
- 不在本轮接 Nautilus（先清私有面）
- 不把 270+ 实验变体树当「内容」保留
