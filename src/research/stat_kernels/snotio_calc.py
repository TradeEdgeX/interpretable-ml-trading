"""Snotio (mean R-multiple) KPI and entry-filter plateau detection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.research.expr import OPS


def compute_snotio(r_returns: pd.Series) -> float:
    """snotio = mean(R-multiples) per trade."""
    valid = pd.to_numeric(r_returns, errors="coerce").dropna()
    if valid.empty:
        return 0.0
    return float(valid.mean())


def width_to_confidence(width: float) -> str:
    """Map plateau width to deploy confidence tier."""
    if width >= 0.3:
        return "HIGH"
    if width >= 0.15:
        return "MEDIUM"
    return "LOW"


def scan_snotio_thresholds(
    df: pd.DataFrame,
    feature: str,
    operator: str,
    grid: List[float],
    base_mask: pd.Series,
    *,
    r_col: str = "forward_rr",
    min_trades: int = 20,
) -> List[Dict[str, Any]]:
    """Scan threshold grid; snotio = mean(R) on rows passing feature filter."""
    if feature not in df.columns:
        raise KeyError(f"Feature missing: {feature}")
    if r_col not in df.columns:
        raise KeyError(f"R column missing: {r_col}")
    op_fn = OPS.get(operator)
    if op_fn is None:
        raise ValueError(f"Unsupported operator: {operator!r}")

    s = pd.to_numeric(df[feature], errors="coerce")
    results: List[Dict[str, Any]] = []
    for thr in grid:
        hit = op_fn(s, thr).fillna(False) & base_mask
        n_hit = int(hit.sum())
        if n_hit < min_trades:
            results.append(
                {
                    "threshold": float(thr),
                    "trades": n_hit,
                    "snotio": 0.0,
                    "too_few": True,
                }
            )
            continue
        r = pd.to_numeric(df.loc[hit, r_col], errors="coerce").dropna()
        if len(r) < min_trades:
            results.append(
                {
                    "threshold": float(thr),
                    "trades": len(r),
                    "snotio": 0.0,
                    "too_few": True,
                }
            )
            continue
        results.append(
            {
                "threshold": float(thr),
                "trades": int(len(r)),
                "snotio": compute_snotio(r),
                "too_few": False,
            }
        )
    return results


def snotio_plateau_payload(
    df: pd.DataFrame,
    feature: str,
    operator: str,
    grid: List[float],
    base_mask: pd.Series,
    *,
    r_col: str = "forward_rr",
    min_trades: int = 20,
    window: int = 5,
    snotio_mode: str = "proxy",
    strategy: Optional[str] = None,
    exec_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run snotio grid scan + find_snotio_plateau; JSON-serializable payload."""
    if snotio_mode == "entry_rr":
        if not strategy:
            raise ValueError("entry_rr snotio_mode requires strategy")
        from src.research.execution_kernel.entry_rr_scan import (
            load_strategy_exec_config,
            prepare_entry_rr_frame,
            scan_snotio_entry_rr_thresholds,
        )

        prepared = prepare_entry_rr_frame(df, strategy)
        cfg = exec_config or load_strategy_exec_config(strategy)
        mask = base_mask.reindex(prepared.index).fillna(False)
        rows = scan_snotio_entry_rr_thresholds(
            prepared,
            feature,
            operator,
            grid,
            mask,
            cfg,
            min_trades=min_trades,
        )
        sim = "entry_rr"
    else:
        rows = scan_snotio_thresholds(
            df,
            feature,
            operator,
            grid,
            base_mask,
            r_col=r_col,
            min_trades=min_trades,
        )
        sim = "proxy"

    plateau = find_snotio_plateau(rows, operator=operator, window=window)
    payload: Dict[str, Any] = {
        "kpi": "snotio",
        "snotio_mode": sim,
        "feature": feature,
        "operator": operator,
        "rows": rows,
    }
    if sim == "proxy":
        payload["r_col"] = r_col
    payload.update(plateau)
    if plateau.get("recommended") is not None:
        payload["mid"] = plateau["recommended"]
        payload["recommended_threshold"] = plateau["recommended"]
    return payload


def find_snotio_plateau(
    results: List[Dict[str, Any]],
    *,
    window: int = 5,
    operator: str = ">=",
    snotio_cv_max: float = 0.3,
    trades_cv_max: float = 0.4,
) -> Dict[str, Any]:
    """Find stable snotio plateau from threshold scan results (entry filter KPI)."""
    valid = [r for r in results if not r.get("too_few")]
    if len(valid) < window:
        return {
            "is_plateau": False,
            "reason": f"有效点不足 ({len(valid)} < {window})",
        }

    best_plateau: Optional[Dict[str, Any]] = None
    for i in range(len(valid) - window + 1):
        w = valid[i : i + window]
        snotios = [r["snotio"] for r in w]
        trades_list = [r["trades"] for r in w]
        mean_sn = float(np.mean(snotios))
        std_sn = float(np.std(snotios))
        cv_snotio = std_sn / mean_sn if mean_sn > 1e-8 else 999.0
        mean_tr = float(np.mean(trades_list))
        std_tr = float(np.std(trades_list))
        cv_trades = std_tr / mean_tr if mean_tr > 1e-8 else 999.0

        if cv_snotio < snotio_cv_max and cv_trades < trades_cv_max and mean_sn > 0:
            start_t = w[0]["threshold"]
            end_t = w[-1]["threshold"]
            plateau_width = abs(end_t - start_t)
            robustness = mean_sn * plateau_width
            if best_plateau is None or robustness > best_plateau.get("_robustness", -1.0):
                bias = 0.2 if plateau_width < 0.25 else 0.1
                if operator in (">=", ">"):
                    rec_val = start_t + bias * plateau_width
                else:
                    rec_val = end_t - bias * plateau_width
                rec_idx = int(np.argmin([abs(r["threshold"] - rec_val) for r in w]))
                best_plateau = {
                    "is_plateau": True,
                    "_robustness": float(robustness),
                    "start_threshold": start_t,
                    "end_threshold": end_t,
                    "plateau_width": float(plateau_width),
                    "confidence": width_to_confidence(plateau_width),
                    "mean_snotio": mean_sn,
                    "cv_snotio": float(cv_snotio),
                    "cv_trades": float(cv_trades),
                    "mean_trades": float(mean_tr),
                    "recommended": float(w[rec_idx]["threshold"]),
                }

    if best_plateau is None:
        for i in range(len(valid) - window + 1):
            w = valid[i : i + window]
            snotios = [r["snotio"] for r in w]
            mean_sn = float(np.mean(snotios))
            cv_snotio = float(np.std(snotios) / mean_sn) if mean_sn > 1e-8 else 999.0
            if cv_snotio < snotio_cv_max and mean_sn > 0:
                start_t = w[0]["threshold"]
                end_t = w[-1]["threshold"]
                pw = abs(end_t - start_t)
                robustness = mean_sn * pw
                if best_plateau is None or robustness > best_plateau.get(
                    "_robustness", -1.0
                ):
                    trades_list = [r["trades"] for r in w]
                    cv_trades = (
                        float(np.std(trades_list) / np.mean(trades_list))
                        if np.mean(trades_list) > 0
                        else 999.0
                    )
                    bias = 0.2 if pw < 0.25 else 0.1
                    if operator in (">=", ">"):
                        rec_val = start_t + bias * pw
                    else:
                        rec_val = end_t - bias * pw
                    rec_idx = int(np.argmin([abs(r["threshold"] - rec_val) for r in w]))
                    best_plateau = {
                        "is_plateau": True,
                        "_robustness": float(robustness),
                        "start_threshold": start_t,
                        "end_threshold": end_t,
                        "plateau_width": float(pw),
                        "confidence": width_to_confidence(pw),
                        "mean_snotio": mean_sn,
                        "cv_snotio": cv_snotio,
                        "cv_trades": cv_trades,
                        "cv_trades_warning": True,
                        "mean_trades": float(np.mean(trades_list)),
                        "recommended": float(w[rec_idx]["threshold"]),
                    }

    if best_plateau is None:
        snotios = [r["snotio"] for r in valid]
        best_idx = int(np.argmax(snotios))
        return {
            "is_plateau": False,
            "reason": "无 CV<0.3 的稳定窗口",
            "best_single": {
                "threshold": valid[best_idx]["threshold"],
                "snotio": valid[best_idx]["snotio"],
                "trades": valid[best_idx]["trades"],
            },
        }

    best_plateau.pop("_robustness", None)
    return best_plateau
