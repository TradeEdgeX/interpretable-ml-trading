# 数学原理

**English:** [math.en.md](math.en.md)  
**入口:** [../README_CN.md](../README_CN.md)

循环在 [hypothesis.md](hypothesis.md)。本文只锁**评测机还认的数学约束**，不抄全部特征公式。实现见 `config/feature_dependencies.yaml` 与 `src/features/`。怎么建库、加列、订单流：[features.md](features.md)。

---

## 1. 闭棒时钟（禁止未来函数）

FeatureStore / feature-bus 的约定：

- 行索引 = bar **开盘**时刻 \(T\)
- 该行的 OHLC 与特征用的是区间 \([T, T+\Delta)\)
- 因此 **\(T\) 行只在墙钟 \(T+\Delta\)（收盘）才可知**

公开 dummy 的 `event_backtest` 先对齐到 bar close 再决策。

**非法：** 在 bar 开盘用同一索引的已完成 FS 行决策。网格曾因此把年化印成 50–70%，数字作废。教训：[lessons.md#closed-bar](lessons.md#closed-bar)。

标签（`forward_rr*`）可以看未来；**禁止**当模型 X / 规则入场特征。

---

## 2. 特征分层：因果 vs 谨慎

| 层 | 可以决定什么 | 不可以 |
|---|---|---|
| 结构 / 量能 / 订单流 | Gate：做 / 不做 | 用慢数学量当硬否决主开关 |
| 数学（Hurst / WPT / 频谱 / Hilbert） | Execution：噪声惩罚、缩小暴露 | 当 Gate 把历史 failure 压到 0（hindsight filter） |
| Regime 慢变量 | 这类行情还做不做 | 当 2h 入场 alpha |

原则一句话：**数学不回答「是不是突破」，只回答「现在该有多小心」。**

再分清三件东西，不要混进同一列特征：

| | 是什么 | 例子 |
|---|---|---|
| **因子 / regime** | 你站在哪一侧暴露上 | `ema_1200_position` 带；日线 200 只作许可，不是进场 |
| **合同几何** | 里 / 外 / 作废；定义 R | 对面 SR、箱沿、跌破 200 |
| **特征** | 对几何或量能的**测量** | `wide_sr_dist_atr`、`srb_sr_success_breakout_score`、CVD |

FeatureStore 不签合同。Archetype YAML 才写进 / 止 / 加 / 出。把测量叠成一体分数，会诱使你把「叠在一起」当成更强 alpha。

旧 path2.5 / Evidence 分档不是现网 SRB 主路径；「数学不作 Gate」这条仍锁。长文（BPC 语境）：[architecture/path2.5_math_features.md](architecture/path2.5_math_features.md)

---

## 3. 评价函数（避免自己骗自己）

**Promote（三条杠）** — [LAYER_PROMOTION_CRITERIA.md](../config/experiments/LAYER_PROMOTION_CRITERIA.md)

1. 在 canonical 三段上期望不全面变差（熊 2022 / 牛 2023–24 / 近窗）
2. MaxDD 不恶化
3. 逻辑可解释、对 regime 有说法

附加合同：

- 比 **edge** 必须 **kill-switch OFF**。熔断会让变体在不同时刻停开仓，制造假赢家。
- 肥尾策略额外做 **去 Top-3**。牛市几乎所有趋势策略去顶都为负——那是形状，不是 bug。
- 对外汇报主 KPI：**年化、Calmar（CAGR/|MaxDD|）、胜率、MaxDD%、Sharpe**。  
  `pnl_r = pnl / 当前 risk_budget`，分母随权益变，长持肥尾会被摊薄，**不要用 Total R 当头条**。

IC / 十分位 / label lift 只生成假设。

---

## 4. 探测器 vs 收割器（不同目标函数）

- 探测器：命中率可以低；问的是「有没有踩上对的路径」。
- 收割器：杠杆、加仓、持有、结构退出；问的是「踩上以后吃多厚、离爆仓多远」。

Rolling 的风险形状必须用**专用 harness**（`trend_rolling_simulate.py`）。用 B 层事件回测代理滚仓 P&L 是合同错误。币本位滚仓在同参下 DD 到爆仓量级 → **U-only**。

---

## 5. 账户与规模（执行数学）

- 启动：REST 权益失败则重试 3 次 / 30s，仍失败 **拒绝启动**（不用离线 10000 当仓）。
- 每次新开：再拉一次权益；快照失败则 **挡开仓**，平仓仍允许。
- 熔断 G0：挡新开 / 加仓 / 现货买；允许减仓 / 平 / 现货卖。不是持仓中途强平。
- 一进程一把密钥。缺密钥 fail-closed。

币本位名义收益好看，经常是 inverse 凸性，不是 alpha。评价对齐「同窗长持该币的美元曲线 + 相近 MaxDD」。

---

## 6. A 股侧（另一套数学）

超跌探测器缩池，组合是 **8 槽等权 + 固定持有日**，不是止损机器。持有期高原在 40–50 个交易日；浅止损 / 保本止损已多次证伪。新杠杆必须过探测器扰动闸门（多组 RSI×amp 同号），单阈值扫出的「最优」作废。

---

## 7. 消融对照物

新几何或新过滤先减两层，再谈 promote：

1. **买入持有**（同窗、同宇宙、相近 MaxDD 的美元曲线）
2. **只看慢 regime**（例如只拿 EMA1200 / 日线 200 一侧）

三层叠完不比第 2 层多，就丢掉箱子和 S/R，合同降级到均线。肉眼只取得「合同能写清、层不打架」的开做资格，不是已经赚到钱。

---

主入口：[README_CN.md](../README_CN.md) · 上一篇 [教训](lessons.md) · 下一篇 [使用方法](usage.md)
