"""Phase 6 scaffold: holding-path exit label (bad_exit under execution MAE threshold).

Not trained in the execution-aligned label milestone. Documents boundary vs entry
realized-R label in ``execution_realized_r_label.py``.
"""

from __future__ import annotations

import pandas as pd


def compute_execution_holding_exit_label(
    df: pd.DataFrame,
    *,
    mae_r_threshold: float = 1.0,
    atr_col: str = "atr",
) -> pd.Series:
    """Stub — returns NaN until Phase 6 holding-path simulation is implemented."""
    _ = (df, mae_r_threshold, atr_col)
    return pd.Series(float("nan"), index=df.index, name="holding_exit_label")
