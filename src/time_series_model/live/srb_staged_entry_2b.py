"""
SRB 两段式入场（事件回测）：2a = cross 状态机确认；2b = EMA1200 位置 + 斜率同向。

与 ``scripts/experiment_srb_staged_entry_2a2b.py`` 语义对齐；供 ``event_backtest`` 在 PCM 首仓前做 arm 门控：
  - 每根 primary bar 推进状态机；
  - 当 2b 成立时 arm 同向若干根 bar；
  - 仅当 PCM 同向首仓落在 arm 窗口内才允许 ``open_position``。

Live 路径未接（与 sr_cross_state_machine 现状一致）；默认 ``execution.yaml`` enabled: false。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import pandas as pd

from src.time_series_model.live.srb_cross_state_machine import (
    CrossCandidate,
    CrossConfig,
    CrossDecision,
    update_cross_state,
)
from src.time_series_model.live.srb_regime import swing_sr_levels


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def _ema_trend_ok_row(
    df: pd.DataFrame,
    iloc: int,
    side: str,
    slope_bars: int,
    pos_min: float,
    slope_min: float,
) -> bool:
    if iloc < slope_bars or iloc < 0 or iloc >= len(df):
        return False
    pos = _to_float(df["ema_1200_position"].iloc[iloc])
    prev = _to_float(df["ema_1200_position"].iloc[iloc - slope_bars])
    if pos is None or prev is None:
        return False
    d = pos - prev
    su = str(side).upper()
    if su in ("LONG", "BUY"):
        return pos > pos_min and d > slope_min
    if su in ("SHORT", "SELL"):
        return pos < -pos_min and d < -slope_min
    return False


def _safe_atr_df(df: pd.DataFrame, iloc: int) -> float:
    i = max(0, min(len(df) - 1, iloc))
    v = _to_float(df["atr"].iloc[i])
    if v is None or v <= 0 or v > 1e6:
        return float("nan")
    return max(v, 1e-9)


@dataclass
class _SymStaged:
    cand: Optional[CrossCandidate] = None
    cooldown_until: int = 0
    post2a: Optional[Dict[str, Any]] = None
    last_close: Optional[float] = None
    last_sup: Optional[float] = None
    last_res: Optional[float] = None


@dataclass
class SrbStagedEntry2bRuntime:
    """按 symbol 维护 cross + post-2a；PCM 首仓需匹配 arm。"""

    cross_cfg: CrossConfig
    post_2a_max_bars: int
    ema_slope_bars: int
    ema_pos_min: float
    ema_slope_min: float
    arm_pcm_bars: int
    _per_sym: Dict[str, _SymStaged] = field(default_factory=dict)
    _armed: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_execution_block(
        cls, raw: Optional[Mapping[str, Any]]
    ) -> "SrbStagedEntry2bRuntime":
        raw = raw or {}
        cr = raw.get("cross") or {}
        cfg = CrossConfig(
            confirm_k=max(1, int(cr.get("confirm_k", 3) or 3)),
            fake_lookahead=max(1, int(cr.get("fake_lookahead", 10) or 10)),
            cooldown_bars=max(0, int(cr.get("cooldown_bars", 10) or 10)),
        )
        return cls(
            cross_cfg=cfg,
            post_2a_max_bars=max(1, int(raw.get("post_2a_max_bars", 24) or 24)),
            ema_slope_bars=max(1, int(raw.get("ema_slope_bars", 2) or 2)),
            ema_pos_min=float(raw.get("ema_pos_min", 0.0) or 0.0),
            ema_slope_min=float(raw.get("ema_slope_min", 0.0) or 0.0),
            arm_pcm_bars=max(1, int(raw.get("arm_pcm_bars", 8) or 8)),
        )

    def _st(self, symbol: str) -> _SymStaged:
        if symbol not in self._per_sym:
            self._per_sym[symbol] = _SymStaged()
        return self._per_sym[symbol]

    def _clear_stale_arm(self, symbol: str, bar_idx: int) -> None:
        a = self._armed.get(symbol)
        if a is not None and bar_idx > int(a.get("until", 0)):
            self._armed.pop(symbol, None)

    def _arm(self, symbol: str, side: str, bar_idx: int) -> None:
        self._armed[symbol] = {
            "side": str(side).upper(),
            "until": int(bar_idx) + int(self.arm_pcm_bars),
        }

    def match_arm(self, symbol: str, pcm_side: str, bar_idx: int) -> bool:
        """PCM 拟开 SRB 母仓：arm 窗口内且同向（不消费；成功 open 后再 ``consume_arm``）。"""
        self._clear_stale_arm(symbol, bar_idx)
        ps = str(pcm_side).upper()
        if ps in ("BUY",):
            ps = "LONG"
        if ps in ("SELL",):
            ps = "SHORT"
        a = self._armed.get(symbol)
        if not a:
            return False
        if str(a.get("side", "")).upper() != ps:
            return False
        if bar_idx > int(a.get("until", 0)):
            self._armed.pop(symbol, None)
            return False
        return True

    def consume_arm(self, symbol: str) -> None:
        self._armed.pop(symbol, None)

    def advance(
        self,
        *,
        symbol: str,
        df_srb: pd.DataFrame,
        ts: pd.Timestamp,
        bar_idx: int,
        row: Mapping[str, Any],
        has_srb_position: bool,
    ) -> None:
        """推进一根 primary（SRB TF）。"""
        self._clear_stale_arm(symbol, bar_idx)
        st = self._st(symbol)
        if df_srb is None or df_srb.empty or ts not in df_srb.index:
            return

        sub = df_srb.loc[:ts]
        if sub.empty:
            return
        iloc = int(sub.shape[0]) - 1
        cur_close = float(row.get("close", sub["close"].iloc[-1]) or 0.0)
        sup, res = swing_sr_levels(df_srb, ts, 20)

        if st.post2a is not None:
            po = st.post2a
            lvl = float(po["level"])
            side = str(po["side"])
            is_up = side in ("LONG", "BUY")
            on_ok = (cur_close > lvl) if is_up else (cur_close < lvl)
            if not on_ok:
                st.post2a = None
                st.cooldown_until = bar_idx + self.cross_cfg.cooldown_bars
                st.last_close = cur_close
                st.last_sup, st.last_res = sup, res
                return
            if bar_idx > int(po["deadline_bar"]):
                st.post2a = None
                st.cooldown_until = bar_idx + self.cross_cfg.cooldown_bars
                st.last_close = cur_close
                st.last_sup, st.last_res = sup, res
                return
            if _ema_trend_ok_row(
                df_srb,
                iloc,
                side,
                self.ema_slope_bars,
                self.ema_pos_min,
                self.ema_slope_min,
            ):
                self._arm(symbol, side, bar_idx)
                st.post2a = None
                st.cand = None
                st.cooldown_until = bar_idx + self.cross_cfg.cooldown_bars
            st.last_close = cur_close
            st.last_sup, st.last_res = sup, res
            return

        old_cand = st.cand
        new_cand, dec = update_cross_state(
            candidate=st.cand,
            bar_index=bar_idx,
            close_prev=float(st.last_close) if st.last_close is not None else cur_close,
            close_curr=cur_close,
            support=st.last_sup,
            resistance=st.last_res,
            has_position=bool(has_srb_position),
            cfg=self.cross_cfg,
            cooldown_until_bar=st.cooldown_until,
            open_px=_to_float(row.get("open")),
            high_px=_to_float(row.get("high")),
            low_px=_to_float(row.get("low")),
            volume=_to_float(row.get("volume")),
            volume_ma=_to_float(row.get("volume_ma")),
        )
        st.cand = new_cand
        st.last_close = cur_close
        st.last_sup, st.last_res = sup, res

        if dec.status == "confirmed" and old_cand is not None:
            side = str(dec.side)
            b0 = int(old_cand.bar0)
            ix0 = max(0, min(len(df_srb) - 1, b0 - 1))
            c0 = float(df_srb["close"].iloc[ix0])
            ab0 = _safe_atr_df(df_srb, ix0)
            atr_2a = _safe_atr_df(df_srb, iloc)
            if not (atr_2a == atr_2a) or atr_2a > 1e6:
                st.cand = None
                st.cooldown_until = bar_idx + self.cross_cfg.cooldown_bars
                return
            if not (ab0 == ab0):
                ab0 = atr_2a
            if _ema_trend_ok_row(
                df_srb,
                iloc,
                side,
                self.ema_slope_bars,
                self.ema_pos_min,
                self.ema_slope_min,
            ):
                self._arm(symbol, side, bar_idx)
                st.cand = None
                st.cooldown_until = bar_idx + self.cross_cfg.cooldown_bars
            else:
                st.post2a = {
                    "side": side,
                    "level": float(dec.level or 0.0),
                    "bar0_idx": b0,
                    "close_bar0": c0,
                    "atr_bar0": ab0,
                    "bar_2a_idx": bar_idx,
                    "close_2a": cur_close,
                    "iloc_2a": iloc,
                    "atr_2a": atr_2a,
                    "deadline_bar": bar_idx + self.post_2a_max_bars,
                }
                st.cand = None
                st.cooldown_until = 0
        elif dec.status in ("fake", "expired"):
            st.cand = None
            st.cooldown_until = bar_idx + self.cross_cfg.cooldown_bars
