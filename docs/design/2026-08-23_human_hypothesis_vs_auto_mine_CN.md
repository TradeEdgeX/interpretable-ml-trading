# 人出假设 · 机器只辅助（对 RD-Agent / FGS）

**DATE:** 2026-08-23  
**STATUS:** 讨论锁定 · 开源时原样输出 · 非策略 promote  
**对象:** 开源说明、Agent 教条、对「为何不做成 fin_factor」的回答

关联：

- Promote：[`../../config/experiments/LAYER_PROMOTION_CRITERIA.md`](../../config/experiments/LAYER_PROMOTION_CRITERIA.md)
- 循环：[`../hypothesis.md`](../hypothesis.md) · [`../ARCHITECTURE.md`](../ARCHITECTURE.md)

---

## 0. 一句话

**发现吞吐不是主循环。人提出可证伪的命题，机器在锁死的评测机上验证，人结案后才执行。**  
`rdagent fin_factor` 和本仓库的 `feature-group-search` 形态像，理念相反。开源时不要写成「我们也有自动挖因子」。

---

## 1. 和 RD-Agent 差在哪

| | RD-Agent `fin_factor` | 本仓库主路径 |
|---|---|---|
| 谁出题 | 模型过夜生成因子 / 改 qlib yaml | **人**（或人用 Cursor 转写） |
| 机器干什么 | 爬山：IC 涨了就当下一轮基线 | Phase 1 扫 IC/plateau；Phase 3 三段验 |
| 何时可信 | 分数停了 | 人看完三段主 KPI，填 `verdict` |
| 执行 | 接进组合回测 | 人审 promote；locked yaml 不自动改 |

`mlbot diagnose feature-group-search` 和 `fin_factor` 同属 **组合空间上的分数爬山**（Pool B + 语义组 → Sharpe/CV）。它是树通道的历史旁路，**不是** B/Rolling/A 股的结案入口。用过的体感——漫长、归因断、不敢执行——是这条路径的性质，不是没搜够。

自动挖漫长且不像实盘，因为：

1. 找的是「哪包 X 更能预测 Y」，不是「哪条开仓语句在换市况后还成立」
2. 多重检验：组合爆炸总会挖到一段好看
3. 失败不可归因：没有一句能证伪的人话
4. 全历史混搜 = 非平稳压成一个全局最优

人出「浅止损能不能削回撤」→ 验证失败，失败的是这句话。机器出「group_17+4，Sharpe+0.08」→ 下次不知道该信哪句。

---

## 2. 锁定分工

```text
人：   提出假设（机制 + 预期市况 + 准备怎么执行）
机器： Phase 1 探照灯（IC / lift / plateau；必要时指定组里 FGS）
人：   读表，写一个 τ / 变体
机器： 对口 harness · canonical 三段 · kill switch OFF
人：   REJECT 或 promote（--yes）
```

机器挖掘只许当 **Phase 1 辅助**。禁止：

- 代替假设
- 代替三段
- 分数涨了就写生产 `features.yaml` / `gate.yaml`
- 把 RD-Agent 的循环接到 locked archetype

若外部用 `fin_factor` 吐出因子：当**候选列表**，由人挑一条机制，再进本仓库法庭。

---

## 3. 开源时怎么写（对外口径）

不要写：第一个 AI 自动挖因子的量化框架。  
要写：

> 给人和 Cursor 用的实验引擎。假设由人定；评测机（harness）锁死；自动搜索只辅助出题。  
> 与 RD-Agent 的差别是判卷，不是吞吐。

产品形态（引擎固定，实验可变）见自然语言研发稿 §0–§4。
