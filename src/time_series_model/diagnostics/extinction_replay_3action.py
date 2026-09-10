from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.time_series_model.diagnostics.ood_config import (
    OODConfig,
    compute_size_cap_multiplier,
)
from src.time_series_model.rl.bc_dataset import Router3Action
from src.time_series_model.rl.sim_env_3action import (
    SimEnvConfig,
    simulate_3action_episode,
)


@dataclass(frozen=True)
class ExtinctionReplayConfig:
    """
    Minimal 'extinction replay' for 3-action logs.

    This is intentionally conservative and deterministic:
    - Not a full Nautilus backtest
    - Uses existing counterfactual return columns ret_mean/ret_trend
    - Produces labels and summary stats used by OOD/Survival research reports
    """

    symbol_col: str = "symbol"
    timestamp_col: str = "timestamp"
    mode_col: str = "mode"
    ret_mean_col: str = "ret_mean"
    ret_trend_col: str = "ret_trend"

    # Extinction criteria (path-level)
    equity_floor_frac: float = 0.5  # equity < 50% => extinction
    dd_floor: float = 0.5  # drawdown >= 50% => extinction

    # Survival label horizon (bars)
    survival_horizon_bars: int = 50

    sim_cfg: SimEnvConfig = SimEnvConfig()


def _mode_to_action(m: Any) -> int:
    s = str(m).upper()
    if s == "MEAN":
        return int(Router3Action.MEAN)
    if s == "TREND":
        return int(Router3Action.TREND)
    return int(Router3Action.NO_TRADE)


def _compute_survival_labels_from_equity(
    *,
    equity: np.ndarray,
    horizon: int,
    equity_floor_frac: float,
) -> np.ndarray:
    """
    y_surv[t] = 1 if min(equity[t:t+horizon]) >= equity[t]*equity_floor_frac else 0
    """
    T = len(equity)
    out = np.ones(T, dtype=np.int8)
    h = max(1, int(horizon))
    for t in range(T):
        end = min(T, t + h + 1)
        base = float(equity[t]) if equity[t] != 0 else 1.0
        floor = base * float(equity_floor_frac)
        if float(np.min(equity[t:end])) < float(floor):
            out[t] = 0
    return out


def run_extinction_replay_3action(
    df_logs: pd.DataFrame,
    *,
    cfg: ExtinctionReplayConfig = ExtinctionReplayConfig(),
    ood_cfg: Optional[OODConfig] = None,
    ood_score_col: Optional[str] = None,
    survival_prob_col: Optional[str] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - report: summary dict (per-symbol + overall)
      - sim: per-row simulation output (equity/drawdown/exposure/etc, aligned to df)
      - labels: per-row labels (y_surv, extinct_next_h, etc)

    Policy:
      - actions are derived from df_logs[mode_col]
      - if ood_cfg is provided and both score cols exist, we apply size_cap multipliers
        via mean_multiplier/trend_multiplier arrays (same cap for both modes)
    """
    if df_logs is None or len(df_logs) == 0:
        return {"ok": False, "reason": "empty_logs"}, pd.DataFrame(), pd.DataFrame()
    for c in [
        cfg.symbol_col,
        cfg.timestamp_col,
        cfg.mode_col,
        cfg.ret_mean_col,
        cfg.ret_trend_col,
    ]:
        if c not in df_logs.columns:
            raise ValueError(f"Missing required column: {c}")

    df = df_logs.copy().reset_index(drop=True)
    df[cfg.timestamp_col] = pd.to_datetime(
        df[cfg.timestamp_col], utc=True, errors="coerce"
    )
    df = df.sort_values([cfg.symbol_col, cfg.timestamp_col]).reset_index(drop=True)

    rows_report = []
    all_sim = []
    all_labels = []

    for sym, g in df.groupby(cfg.symbol_col, sort=False):
        g = g.reset_index(drop=True)
        actions = np.asarray(
            [_mode_to_action(x) for x in g[cfg.mode_col].tolist()], dtype=int
        )

        mean_mult = None
        trend_mult = None
        if (
            ood_cfg is not None
            and ood_score_col
            and survival_prob_col
            and ood_score_col in g.columns
            and survival_prob_col in g.columns
        ):
            ood_arr = (
                pd.to_numeric(g[ood_score_col], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            surv_arr = (
                pd.to_numeric(g[survival_prob_col], errors="coerce")
                .fillna(1.0)
                .to_numpy(dtype=float)
            )
            cap = [
                compute_size_cap_multiplier(
                    cfg=ood_cfg, ood_score=float(o), survival_prob=float(s)
                )
                for o, s in zip(ood_arr.tolist(), surv_arr.tolist())
            ]
            mean_mult = cap
            trend_mult = cap

        sim = simulate_3action_episode(
            g,
            actions=actions.tolist(),
            mean_multiplier=mean_mult,
            trend_multiplier=trend_mult,
            cfg=cfg.sim_cfg,
        )
        sim = sim.reset_index(drop=True)
        sim.insert(0, cfg.symbol_col, sym)
        sim.insert(1, cfg.timestamp_col, g[cfg.timestamp_col].values)

        eq = (
            sim["equity"].to_numpy(dtype=float)
            if "equity" in sim.columns
            else np.asarray([], dtype=float)
        )
        dd = (
            sim["drawdown"].to_numpy(dtype=float)
            if "drawdown" in sim.columns
            else np.asarray([], dtype=float)
        )
        equity_floor = float(cfg.sim_cfg.initial_equity) * float(cfg.equity_floor_frac)
        extinct = bool(
            (len(eq) and float(np.min(eq)) < equity_floor)
            or (len(dd) and float(np.max(dd)) >= float(cfg.dd_floor))
        )

        y_surv = _compute_survival_labels_from_equity(
            equity=eq,
            horizon=int(cfg.survival_horizon_bars),
            equity_floor_frac=float(cfg.equity_floor_frac),
        )

        labels = pd.DataFrame(
            {
                cfg.symbol_col: sym,
                cfg.timestamp_col: g[cfg.timestamp_col].values,
                "y_surv": y_surv,
            }
        )

        rows_report.append(
            {
                "symbol": str(sym),
                "n_rows": int(len(g)),
                "final_equity": (
                    float(eq[-1]) if len(eq) else float(cfg.sim_cfg.initial_equity)
                ),
                "max_drawdown": float(np.max(dd)) if len(dd) else 0.0,
                "extinct": bool(extinct),
            }
        )

        all_sim.append(sim)
        all_labels.append(labels)

    per_symbol = pd.DataFrame(rows_report)
    report = {
        "ok": True,
        "n_symbols": int(len(per_symbol)),
        "extinction_rate": (
            float(per_symbol["extinct"].mean()) if len(per_symbol) else 0.0
        ),
        "avg_max_drawdown": (
            float(per_symbol["max_drawdown"].mean()) if len(per_symbol) else 0.0
        ),
        "per_symbol": per_symbol.to_dict(orient="records"),
    }
    return (
        report,
        pd.concat(all_sim, ignore_index=True),
        pd.concat(all_labels, ignore_index=True),
    )
