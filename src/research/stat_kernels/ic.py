"""IC / rank correlation kernels with horizon shift support."""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import pandas as pd
from scipy.stats import spearmanr


def resolve_target_col(
    df: pd.DataFrame, target: str, horizon: int
) -> Tuple[Optional[str], bool]:
    """Resolve target column for horizon H.

    Returns (column_name, used_shift). When no ``target_H`` column exists and H>1,
    callers should apply ``shift_target_by_horizon`` on the base target column.
    """
    if horizon <= 1:
        return (target if target in df.columns else None), False
    for cand in (f"{target}_{horizon}", f"{target}{horizon}"):
        if cand in df.columns:
            return cand, False
    if target in df.columns:
        return target, True
    return None, False


def shift_target_by_horizon(
    series: pd.Series,
    horizon: int,
    df: pd.DataFrame,
) -> pd.Series:
    """Shift target forward by ``horizon`` bars (per symbol if grouped column exists)."""
    if horizon <= 1:
        return series
    sym_col = None
    if "symbol" in df.columns:
        sym_col = "symbol"
    elif "_symbol" in df.columns:
        sym_col = "_symbol"
    if sym_col is not None:
        return series.groupby(df[sym_col]).shift(-horizon)
    return series.shift(-horizon)


def rank_ic(x: pd.Series, y: pd.Series, *, min_n: int = 100) -> Tuple[float, float, int]:
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < min_n:
        return float("nan"), float("nan"), n
    rho, p = spearmanr(x[m], y[m])
    return float(rho), float(p), n


def ic_decay_rows(
    df: pd.DataFrame,
    features: Iterable[str],
    horizons: Iterable[int],
    target: str,
    *,
    mask: Optional[pd.Series] = None,
    min_n: int = 100,
) -> List[dict]:
    """Compute IC decay table rows for features × horizons."""
    sub = df.loc[mask] if mask is not None else df
    rows: List[dict] = []
    for feat in features:
        if feat not in sub.columns:
            rows.append(
                {
                    "feature": feat,
                    "horizon": None,
                    "target_col": None,
                    "n": 0,
                    "rank_ic": float("nan"),
                    "p_value": float("nan"),
                    "shifted": False,
                }
            )
            continue
        x = pd.to_numeric(sub[feat], errors="coerce")
        for h in horizons:
            tcol, need_shift = resolve_target_col(sub, target, h)
            if tcol is None:
                rows.append(
                    {
                        "feature": feat,
                        "horizon": h,
                        "target_col": "missing",
                        "n": 0,
                        "rank_ic": float("nan"),
                        "p_value": float("nan"),
                        "shifted": False,
                    }
                )
                continue
            y = pd.to_numeric(sub[tcol], errors="coerce")
            shifted = False
            if need_shift and h > 1:
                y = shift_target_by_horizon(y, h, sub)
                shifted = True
            rho, p, n = rank_ic(x, y, min_n=min_n)
            rows.append(
                {
                    "feature": feat,
                    "horizon": h,
                    "target_col": tcol + (f" (shift -{h})" if shifted else ""),
                    "n": n,
                    "rank_ic": rho,
                    "p_value": p,
                    "shifted": shifted,
                }
            )
    return rows
