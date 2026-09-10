#!/usr/bin/env python3
"""
Execution Path Score — measures entry quality for immediate profitability.

Core concept: we don't want to "扛单" (endure drawdown). We want the pullback
to be mature, absorption confirmed, and support holding — so the position
is profitable soon after entry.

Three components (each 0-100, combined to 0-100):

1. PULLBACK_MATURITY (35%): How mature is the pullback?
   - Depth: (peak - close) / ATR  → deeper = more mature
   - Duration: bars since peak     → longer = more mature
   - Retracement: % of prior move retraced

2. ABSORPTION (40%): Is price stabilizing / consolidating?
   - Range contraction: today's range / recent avg range → shrinking
   - Volume contraction: today's volume / recent avg volume → drying up
   - Price compression: BB width percentile → squeezing
   - Stabilization: close near middle of recent range

3. SUPPORT_HOLDING (25%): Is support not breaking?
   - Distance from recent low / ATR → staying above support
   - Higher low pattern: last swing low > prior swing low
   - Bounce confirmation: last bar closed above open or mid

Output:
  execution_score (0-100): higher = better entry timing
  execution_flags: list of descriptive labels

Usage:
  from src.execution.execution_path_scorer import ExecutionPathScorer
  scorer = ExecutionPathScorer()
  score = scorer.score(df_ohlcv)  # df with close/high/low/volume
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExecutionScore:
    """Execution path evaluation result."""

    total: float = 50.0  # 0-100
    pullback_maturity: float = 50.0
    absorption: float = 50.0
    support_holding: float = 50.0
    flags: list[str] = field(default_factory=list)

    @property
    def is_good_entry(self) -> bool:
        return self.total >= 60 and self.absorption >= 50

    @property
    def label(self) -> str:
        if self.total >= 75:
            return "🟢 优质入场"
        elif self.total >= 60:
            return "🟡 可入场"
        elif self.total >= 40:
            return "🟠 观望"
        else:
            return "🔴 回避"


class ExecutionPathScorer:
    """Score the quality of the current entry point.

    Parameters
    ----------
    lookback_peak: int
        Bars to look back for local peak (pullback start). Default 60.
    lookback_swing: int
        Bars for swing low detection. Default 20.
    absorption_window: int
        Window for absorption measurement. Default 10.
    vol_window: int
        Volume moving average window. Default 20.
    bb_window: int
        Bollinger band window. Default 20.
    """

    def __init__(
        self,
        lookback_peak: int = 60,
        lookback_swing: int = 20,
        absorption_window: int = 10,
        vol_window: int = 20,
        bb_window: int = 20,
    ):
        self.lookback_peak = lookback_peak
        self.lookback_swing = lookback_swing
        self.absorption_window = absorption_window
        self.vol_window = vol_window
        self.bb_window = bb_window

    # ── Public API ───────────────────────────────────────────────

    def score(self, df: pd.DataFrame) -> ExecutionScore:
        """Compute execution path score from OHLCV DataFrame.

        df must have columns: close, high, low, volume.
        Uses the last row as the current bar.
        """
        if len(df) < max(self.lookback_peak, self.vol_window) + 10:
            return ExecutionScore(total=50.0)

        c = df["close"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        v = df["volume"].astype(float)

        # Compute components
        pb = self._pullback_maturity(c, h, l)
        ab = self._absorption(c, h, l, v)
        sh = self._support_holding(c, h, l)

        # Weighted total
        total = pb * 0.35 + ab * 0.40 + sh * 0.25

        flags = self._build_flags(pb, ab, sh, c, h, l, v)

        return ExecutionScore(
            total=round(total, 1),
            pullback_maturity=round(pb, 1),
            absorption=round(ab, 1),
            support_holding=round(sh, 1),
            flags=flags,
        )

    # ── Component scorers ────────────────────────────────────────

    def _pullback_maturity(self, c: pd.Series, h: pd.Series, l: pd.Series) -> float:
        """Score pullback maturity (0-100). Higher = more mature pullback."""
        n = len(c)
        window = min(self.lookback_peak, n - 1)
        if window < 5:
            return 50.0

        # Find local peak in lookback window
        recent_h = h.iloc[-window:]
        peak_idx = recent_h.idxmax()
        peak_val = recent_h.max()
        current_c = c.iloc[-1]

        # Depth in ATR units
        tr = np.maximum(h - l, np.maximum(abs(h - c.shift()), abs(l - c.shift())))
        atr14 = tr.rolling(14).mean().iloc[-1]
        if atr14 <= 0:
            atr14 = (h.iloc[-1] - l.iloc[-1]) or 0.01

        depth_atr = (peak_val - current_c) / atr14

        # Duration: bars since peak
        peak_pos = (
            c.index.get_loc(peak_idx)
            if hasattr(c.index, "get_loc")
            else list(c.index).index(peak_idx)
        )
        duration_bars = n - 1 - peak_pos
        duration_score = min(duration_bars / 20, 1.0)  # 20 bars = full score

        # Retracement: how much of the prior up-move was retraced?
        start_idx = max(0, peak_pos - window)
        prior_low = l.iloc[start_idx:peak_pos].min()
        prior_range = peak_val - prior_low
        if prior_range > 0:
            retrace_pct = (peak_val - current_c) / prior_range
            retrace_pct = min(retrace_pct, 1.5)  # can exceed 100%
        else:
            retrace_pct = 0.5

        # Score
        score = 50.0
        # Depth: 1-3 ATR is ideal pullback
        if 1.0 <= depth_atr <= 3.0:
            score += 30
        elif 0.5 <= depth_atr < 1.0:
            score += 15
        elif depth_atr > 3.0:
            score -= 10  # too deep = distressed
        elif depth_atr < 0.3:
            score -= 20  # too shallow = no opportunity

        # Duration bonus
        score += duration_score * 15

        # Retracement: 0.3-0.7 retracement is ideal (Fibonacci zone)
        if 0.3 <= retrace_pct <= 0.7:
            score += 15
        elif 0.1 <= retrace_pct < 0.3:
            score += 5
        elif retrace_pct > 0.9:
            score -= 10  # almost full reversal

        return max(0.0, min(100.0, score))

    def _absorption(
        self, c: pd.Series, h: pd.Series, l: pd.Series, v: pd.Series
    ) -> float:
        """Score absorption quality (0-100). Higher = better absorption."""
        n = len(c)
        aw = min(self.absorption_window, n - 1)
        if aw < 3:
            return 50.0

        # Recent bars
        recent_range = h.iloc[-aw:] - l.iloc[-aw:]
        recent_vol = v.iloc[-aw:]

        # Longer context
        ctx_len = min(self.vol_window * 2, n)
        ctx_range = h.iloc[-ctx_len:] - l.iloc[-ctx_len:]
        ctx_vol = v.iloc[-ctx_len:]

        avg_range = ctx_range.mean()
        avg_vol = ctx_vol.mean()

        # Range contraction ratio (lower = more contracted)
        range_ratio = recent_range.mean() / avg_range if avg_range > 0 else 1.0
        range_score = max(0, 1 - range_ratio) * 100

        # Volume contraction ratio
        vol_ratio = recent_vol.mean() / avg_vol if avg_vol > 0 else 1.0
        vol_score = max(0, 1 - vol_ratio) * 100

        # BB width compression
        bb_mid = c.rolling(self.bb_window).mean()
        bb_std = c.rolling(self.bb_window).std()
        bb_width = (2 * bb_std) / bb_mid.replace(0, np.nan)
        bb_width_rank = bb_width.rolling(min(n, 252)).apply(
            lambda x: (x.iloc[-1] <= x).mean() if len(x) > 20 else 0.5
        )
        bb_score = (
            (1 - bb_width_rank.iloc[-1]) * 100
            if not pd.isna(bb_width_rank.iloc[-1])
            else 50
        )

        # Price stabilization: close near mid of recent range
        recent_h_max = h.iloc[-aw:].max()
        recent_l_min = l.iloc[-aw:].min()
        recent_mid = (recent_h_max + recent_l_min) / 2
        if recent_h_max > recent_l_min:
            close_pos = (c.iloc[-1] - recent_l_min) / (recent_h_max - recent_l_min)
            # Prefer close near middle (0.4-0.6) for stabilization
            stab_score = max(0, 1 - abs(close_pos - 0.5) * 2) * 100
        else:
            stab_score = 50

        # Composite absorption
        score = (
            range_score * 0.30 + vol_score * 0.30 + bb_score * 0.25 + stab_score * 0.15
        )

        return max(0.0, min(100.0, score))

    def _support_holding(self, c: pd.Series, h: pd.Series, l: pd.Series) -> float:
        """Score support holding (0-100). Higher = support is strong."""
        n = len(c)
        sw = min(self.lookback_swing, n - 1)
        if sw < 5:
            return 50.0

        # Recent swing lows
        recent_l = l.iloc[-sw:]
        swing_low = recent_l.min()
        current_c = c.iloc[-1]

        # Distance from swing low (in ATR)
        tr = np.maximum(h - l, np.maximum(abs(h - c.shift()), abs(l - c.shift())))
        atr14 = tr.rolling(14).mean().iloc[-1]
        if atr14 <= 0:
            atr14 = 0.01
        dist_atr = (current_c - swing_low) / atr14

        # Higher low detection: does swing low > prior swing low?
        prior_l = l.iloc[-(sw * 2) : -sw].min() if n >= sw * 2 else swing_low
        higher_low = (
            1.0
            if swing_low > prior_l
            else (0.5 if swing_low >= prior_l * 0.98 else 0.0)
        )

        # Bounce: last bar closed in upper half
        last_range = h.iloc[-1] - l.iloc[-1]
        if last_range > 0:
            bounce_quality = (c.iloc[-1] - l.iloc[-1]) / last_range
        else:
            bounce_quality = 0.5

        # Score
        score = 50.0
        # Distance: 0.3-1.5 ATR above support is safe zone
        if 0.5 <= dist_atr <= 1.5:
            score += 25
        elif 0.2 <= dist_atr < 0.5:
            score += 10
        elif dist_atr < 0.1:
            score -= 15  # right on support = risky
        elif dist_atr > 2.0:
            score += 5  # far from support but maybe missed the move

        # Higher low bonus
        score += higher_low * 20

        # Bounce quality
        if bounce_quality > 0.6:
            score += 15
        elif bounce_quality < 0.3:
            score -= 10

        return max(0.0, min(100.0, score))

    def _build_flags(self, pb, ab, sh, c, h, l, v) -> list[str]:
        flags = []
        if pb >= 70:
            flags.append("回调充分")
        elif pb >= 55:
            flags.append("回调中")
        elif pb < 35:
            flags.append("无明显回调")

        if ab >= 70:
            flags.append("吸收良好")
        elif ab >= 55:
            flags.append("弱吸收")
        elif ab < 35:
            flags.append("未吸收")

        if sh >= 70:
            flags.append("支撑坚实")
        elif sh >= 55:
            flags.append("支撑一般")
        elif sh < 35:
            flags.append("支撑脆弱")

        return flags


# ── Convenience: compute OHLCV-based execution score for a symbol ──


def compute_execution_score_from_ohlcv(
    df: pd.DataFrame,
    lookback_bars: int = 200,
) -> ExecutionScore:
    """Compute execution score from a daily OHLCV DataFrame.

    df: DataFrame with columns close/high/low/volume, DatetimeIndex sorted ascending.
    Uses last `lookback_bars` bars for context.
    """
    if len(df) > lookback_bars:
        df = df.tail(lookback_bars)
    scorer = ExecutionPathScorer()
    return scorer.score(df)


if __name__ == "__main__":
    # Quick test
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    fp = "data/ashare/daily/600276.parquet"
    df = pd.read_parquet(fp)
    if "date" in df.columns:
        df = df.rename(columns={"date": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    df = df.sort_index().tail(200)

    scorer = ExecutionPathScorer()
    result = scorer.score(df)
    print(f"Execution Score: {result.total:.1f}/100 — {result.label}")
    print(f"  Pullback Maturity: {result.pullback_maturity:.1f}")
    print(f"  Absorption:        {result.absorption:.1f}")
    print(f"  Support Holding:   {result.support_holding:.1f}")
    print(f"  Flags: {result.flags}")
