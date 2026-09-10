# Mean-reversion feature events — 口径教训

日期：2026-07-14  
关联：`.cursor/rules/auxiliary-exit-signal-phase2-rejected.mdc`（触发日对齐）、
`config/experiments/20260714_chop_mr_features/`、
`config/experiments/20260714_chop_mr_events/`

## 坑：用「水平态」当「回归事件」

| 错误 | 正确 |
|------|------|
| `rsi >= 80` 的每个 bar 做空 | **上穿 80 → 再跌回 ≤80** 的那一根才算一次 overbought **事件**；从该 bar 起算 forward |
| `box_pos >= 0.8` 持续做空 | 穿出上沿后 **回到带内** 再记 short 事件 |
| 水平 IC(rsi, fwd)>0 就说「动量」 | 水平态本就会混进趋势延伸；要先用 **事件完成** 再谈回归 |

**机制**：刚上穿 80 时，价格往往仍在冲；此时做空测的是「逆动量」，不是「回归完成」。  
回归叙事的自然时点是 **极值过程结束、重新跌破阈值**（或对称的超卖回升）。

与 auxiliary-exit 教训同族：**时间锚必须对齐**——不能拿「仍在极端区的 bar」去代表「回归信号已触发」。

## 代码

- 错误口径扫描：`scripts/scan_chop_mr_features_phase1.py`（水平 / 背离）
- 事件口径扫描：`scripts/scan_chop_mr_events_phase1.py`（cross-and-return）

## 硬规则（以后 MR / 极值 fade Phase-1）

1. 默认用 **cross-into → cross-back** 事件，禁止只用水平阈值 bar。  
2. forward 标签从 **return-cross bar** 起算。  
3. 同时报告水平对照，写明「水平 = 已知坑，仅作对照」。  
4. 事件样本太少（n<50）→ 不判 promote，只记「稀疏」。
