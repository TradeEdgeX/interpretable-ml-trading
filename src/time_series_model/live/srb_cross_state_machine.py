"""
SRB cross-event 状态机（纯函数 / dataclass 容器）。

**当前消费面**：
- **事件回测**：不再接入 ``srb_staged_entry_2b`` / ``SrbStagedEntry2bRuntime``；SRB 与其它策略
  一样走 PCM → ``open_position``。本状态机仍可被独立脚本或实验代码调用。
- **Live**：未接本状态机；方向与结构由策略内 gate / 特征与 PCM 仲裁完成。

每根 primary bar 调用 ``update_cross_state()``；当价格"穿越关键位"时起一个
候选（``CrossCandidate``），在 ``fake_lookahead`` 根 bar 内：

- 连续 ``confirm_k`` 根 close 站在 level 正确侧 → decision=``confirmed`` →
  顺势入场。
- wick+低量 prior 或连续 ``fail_count >= confirm_k`` → decision=``fake`` →
  反向入场候选（**SRB 不再消费此路径**；保留给策略 X / FBF）。
- 超过 ``fake_lookahead`` 未决 → decision=``expired``。

实现语义复刻 ``docs/archive/rule_based_strategies/sr_breakout_bot.py``
的 ``find_candidate_break`` + ``calc_break`` +
``wick_fake_break_check_for_last_candidate``（L524–735）。

设计要点：
- 无 I/O、无 pandas 依赖；便于单测与 live 集成。
- 候选状态仅跟踪一个 symbol 上"当前"活跃候选；起新候选前要求无持仓
  （legacy：``not has_position``）。
- cooldown：已 confirmed/fake 后 N 根 bar 内不再起新候选（对同一 level）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple


__all__ = [
    "CrossCandidate",
    "CrossConfig",
    "CrossDecision",
    "detect_cross",
    "update_cross_state",
]


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossConfig:
    """从 ``execution.yaml: sr_cross_state_machine`` 读出的参数。"""

    enabled: bool = True
    confirm_k: int = 3
    fake_lookahead: int = 10
    wick_ratio_threshold: float = 2.0
    low_vol_ratio: float = 0.8
    cooldown_bars: int = 10
    max_reverse_per_level: int = 1

    @classmethod
    def from_mapping(cls, raw: Optional[Mapping[str, Any]]) -> "CrossConfig":
        if not isinstance(raw, Mapping):
            return cls()
        return cls(
            enabled=bool(raw.get("enabled", True)),
            confirm_k=max(1, int(raw.get("confirm_k", 3) or 3)),
            fake_lookahead=max(1, int(raw.get("fake_lookahead", 10) or 10)),
            wick_ratio_threshold=float(raw.get("wick_ratio_threshold", 2.0) or 2.0),
            low_vol_ratio=float(raw.get("low_vol_ratio", 0.8) or 0.8),
            cooldown_bars=max(0, int(raw.get("cooldown_bars", 10) or 10)),
            max_reverse_per_level=max(0, int(raw.get("max_reverse_per_level", 1) or 1)),
        )


# ---------------------------------------------------------------------------
# 候选
# ---------------------------------------------------------------------------


@dataclass
class CrossCandidate:
    """单 SR-cross 候选状态。方向命名与 legacy 对齐：``up`` = 向上穿越阻力。"""

    direction: str  # 'up' | 'down'
    level: float
    bar0: int
    confirm_count: int = 0
    fail_count: int = 0
    fake_stage: bool = False
    fake_stage_count: int = 0


@dataclass
class CrossDecision:
    """状态机本根 bar 的决策结果。"""

    status: str  # 'pending' | 'confirmed' | 'fake' | 'expired' | 'idle'
    side: Optional[str] = None  # 'LONG' | 'SHORT'（仅 confirmed/fake 有效）
    level: Optional[float] = None


# ---------------------------------------------------------------------------
# cross 检测
# ---------------------------------------------------------------------------


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def detect_cross(
    close_prev: float,
    close_curr: float,
    support: Optional[float],
    resistance: Optional[float],
) -> Optional[Tuple[str, float]]:
    """检测是否穿越 SR 关键位。

    优先级：resistance(up) → support(down)，与 legacy ``find_candidate_break``
    的 swing 优先级一致（MVP 不使用 zigzag/POC）。

    Returns:
        (direction, level) 或 None。direction ∈ {'up', 'down'}。
    """
    cp = _to_float(close_prev)
    cc = _to_float(close_curr)
    if cp is None or cc is None:
        return None
    r = _to_float(resistance)
    s = _to_float(support)
    if r is not None and cp <= r < cc:
        return ("up", r)
    if s is not None and cc < s <= cp:
        return ("down", s)
    return None


# ---------------------------------------------------------------------------
# 状态机主推进
# ---------------------------------------------------------------------------


def _wick_fake_prior(
    *,
    direction: str,
    open_px: Optional[float],
    high_px: Optional[float],
    low_px: Optional[float],
    close_px: Optional[float],
    volume: Optional[float],
    volume_ma: Optional[float],
    cfg: CrossConfig,
) -> bool:
    """复刻 legacy wick_fake_break_check：大影线 + 量能不足 → 提前进入 fake_stage。

    up-break: 上影长且 volume < low_vol_ratio * vol_ma。
    down-break: 下影长且 volume < low_vol_ratio * vol_ma。
    任一数据缺失或阈值不合法 → 不触发 prior（保守不加速 fake）。
    """
    o = _to_float(open_px)
    h = _to_float(high_px)
    l = _to_float(low_px)
    c = _to_float(close_px)
    v = _to_float(volume)
    vm = _to_float(volume_ma)
    if None in (o, h, l, c, v, vm):
        return False
    if vm <= 0 or cfg.wick_ratio_threshold <= 0:
        return False
    body = abs(c - o)
    if body <= 1e-12:
        body = 1e-12
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    low_vol = v < cfg.low_vol_ratio * vm
    if direction == "up":
        return low_vol and upper_wick >= cfg.wick_ratio_threshold * body
    if direction == "down":
        return low_vol and lower_wick >= cfg.wick_ratio_threshold * body
    return False


def update_cross_state(
    *,
    candidate: Optional[CrossCandidate],
    bar_index: int,
    close_prev: float,
    close_curr: float,
    support: Optional[float],
    resistance: Optional[float],
    has_position: bool,
    cfg: CrossConfig,
    cooldown_until_bar: int = 0,
    open_px: Optional[float] = None,
    high_px: Optional[float] = None,
    low_px: Optional[float] = None,
    volume: Optional[float] = None,
    volume_ma: Optional[float] = None,
) -> Tuple[Optional[CrossCandidate], CrossDecision]:
    """推进 cross 状态机一步。

    调用顺序（每根 primary bar 一次）：
      1. 若 ``candidate is None`` 且 ``not has_position`` 且 ``bar_index >= cooldown_until_bar``
         → 调 ``detect_cross`` 尝试起新候选。
      2. 若仍无候选 → decision=idle。
      3. 有候选：更新 confirm_count / fail_count（规则见 legacy L564–605），
         检查 wick-fake prior（若触发 → 打 fake_stage，legacy L557–562 / L711–734）。
      4. 判定顺序：confirmed > fake_stage(≥K) > fail_count(≥K) > expired。
      5. 任何最终决定（confirmed/fake/expired）会清空候选；cooldown 由调用方基于
         返回值自行维护。

    Returns:
        (new_candidate_state, decision)。new_candidate_state 若 decision 为
        confirmed/fake/expired 则为 None；否则返回被推进后的候选（pending）。
    """
    # 1) 起新候选（无持仓 + 无当前候选 + 不在 cooldown 内）
    if candidate is None:
        if has_position or bar_index < cooldown_until_bar:
            return None, CrossDecision(status="idle")
        cross = detect_cross(close_prev, close_curr, support, resistance)
        if cross is None:
            return None, CrossDecision(status="idle")
        direction, level = cross
        candidate = CrossCandidate(
            direction=direction,
            level=level,
            bar0=bar_index,
            confirm_count=1,  # 起始即已穿越 → 初始计 1
            fail_count=0,
        )
        # 起候选当根就继续评估 wick-prior（与 legacy 一致）
    else:
        # 方向/level 不变，直接推进。
        pass

    cc = _to_float(close_curr)
    lvl = _to_float(candidate.level)
    if cc is None or lvl is None:
        return candidate, CrossDecision(status="pending")

    is_up = candidate.direction == "up"

    # 2) 累积 confirm_count / fail_count —— 依据本根 close 相对 level 的位置
    on_correct_side = (cc > lvl) if is_up else (cc < lvl)
    if candidate.bar0 != bar_index:
        # 非起始根才累积（起始根已在 push 时计 confirm=1）
        if on_correct_side:
            candidate.confirm_count += 1
            candidate.fail_count = 0
            # 若已打了 fake_stage，因 close 重新站稳被撤销
            candidate.fake_stage = False
            candidate.fake_stage_count = 0
        else:
            candidate.fail_count += 1
            if candidate.fake_stage:
                candidate.fake_stage_count += 1

    # 3) wick-prior：大影线 + 低量 → 打 fake_stage（与 legacy 一致）
    if not candidate.fake_stage and _wick_fake_prior(
        direction=candidate.direction,
        open_px=open_px,
        high_px=high_px,
        low_px=low_px,
        close_px=close_curr,
        volume=volume,
        volume_ma=volume_ma,
        cfg=cfg,
    ):
        candidate.fake_stage = True
        candidate.fake_stage_count = 1

    # 4) 判定
    side_long = "LONG"
    side_short = "SHORT"
    if candidate.confirm_count >= cfg.confirm_k:
        decided_side = side_long if is_up else side_short
        return None, CrossDecision(
            status="confirmed", side=decided_side, level=candidate.level
        )

    fake_confirmed = False
    if candidate.fake_stage and candidate.fake_stage_count >= cfg.confirm_k:
        fake_confirmed = True
    if candidate.fail_count >= cfg.confirm_k:
        fake_confirmed = True
    if fake_confirmed:
        decided_side = side_short if is_up else side_long
        return None, CrossDecision(
            status="fake", side=decided_side, level=candidate.level
        )

    if (bar_index - candidate.bar0) > cfg.fake_lookahead:
        return None, CrossDecision(status="expired")

    return candidate, CrossDecision(status="pending")
