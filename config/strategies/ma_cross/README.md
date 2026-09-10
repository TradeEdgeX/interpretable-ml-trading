# ma_cross — public dummy pack (harness fixture)

Not a live sleeve and **not a job you must run**. It exists so the agent
court has a non-SRB strategy shape. See `docs/agent/rd_playbook.md`.

Textbook B-layer pack for the experiment engine. **Not a live sleeve.**
Do not copy SRB / fade / rolling logic into this tree.

```text
side = sign(ema_1200_position)
enter when |ema_1200_position| >= 0.02
exit = ATR trailing stop
```

| File | Layer | This pack |
|---|---|---|
| `regime.yaml` | when the book is open | `atr > 0` (always-on) |
| `prefilter.yaml` | detector | leave the EMA1200 dead zone |
| `direction.yaml` | side | `sign(ema_1200_position)` |
| `gate.yaml` | hard deny | empty |
| `entry_filters.yaml` | timing enhance | empty |
| `execution.yaml` | harvest | ATR trail, no adds |

策略包已经在这里，**不需要**先开实验目录才能用。

```bash
# 直接回测这套默认规则（死区 0.02 + ATR 跟踪）
python -m scripts.event_backtest --strategy ma_cross --no-kill-switch

# 只有当你有一条可证伪的假设时，才开实验，例如：
# 「把死区从 0.02 改成 0.05 会不会少做震荡」
mlbot research init 20260823_ma_deadzone --strategy ma_cross
```

默认包**不是**快慢均线交叉。交叉是另一条假设，特征已登记为 `ema_50_200_cross_f`，要先 FeatureStore backfill，再用 `direction_ema50_200_cross.yaml`。流程：`docs/agent/rd_playbook.md`。开着的例子：`config/experiments/20260823_ma50_ma200_cross/`。

STATUS: **research / demo**. Not in `constitution.yaml`.
