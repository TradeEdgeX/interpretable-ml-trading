from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AssetStats:
    mu: float
    sigma: float
    cvar_05: float
    max_dd: float
    stability: float
    fragility: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "mu": float(self.mu),
            "sigma": float(self.sigma),
            "cvar_05": float(self.cvar_05),
            "max_dd": float(self.max_dd),
            "stability": float(self.stability),
            "fragility": float(self.fragility),
        }


def _max_drawdown_from_returns(r: np.ndarray) -> float:
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.maximum(peak, 1e-12)
    return float(np.max(dd)) if dd.size else 0.0


def compute_asset_stats(
    returns: pd.Series | np.ndarray,
    *,
    regime: Optional[pd.Series] = None,
    alpha: float = 0.05,
) -> AssetStats:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return AssetStats(
            mu=0.0, sigma=0.0, cvar_05=0.0, max_dd=0.0, stability=0.0, fragility=0.0
        )

    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
    q = float(np.quantile(r, float(alpha)))
    tail = r[r <= q]
    cvar = float(np.mean(tail)) if tail.size else float(q)
    max_dd = _max_drawdown_from_returns(r)

    # stability: penalize high sigma and high drawdown (bounded 0..1)
    stability = float(1.0 / (1.0 + max(0.0, sigma) + max(0.0, max_dd)))

    # fragility: variance of conditional means across regimes (if provided)
    frag = 0.0
    if regime is not None:
        rg = pd.Series(regime).reset_index(drop=True)
        rr = pd.Series(returns).reset_index(drop=True)
        df = pd.DataFrame({"r": rr, "regime": rg})
        df = df[np.isfinite(df["r"].to_numpy(dtype=float))]
        if len(df) >= 10:
            cond = df.groupby("regime")["r"].mean()
            if len(cond) >= 2:
                frag = float(np.var(cond.to_numpy(dtype=float)))

    return AssetStats(
        mu=mu,
        sigma=sigma,
        cvar_05=cvar,
        max_dd=max_dd,
        stability=stability,
        fragility=float(frag),
    )
