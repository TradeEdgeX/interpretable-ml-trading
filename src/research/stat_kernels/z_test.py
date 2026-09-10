"""Two-proportion z-test kernel."""

from __future__ import annotations

import numpy as np


def two_proportion_z(
    p_hit: float, n_hit: int, p_other: float, n_other: int, *, min_n: int = 5
) -> float:
    """Absolute z for two-proportion test; returns 0 when sample too small."""
    if n_hit < min_n or n_other < min_n:
        return 0.0
    pool = (p_hit * n_hit + p_other * n_other) / max(n_hit + n_other, 1)
    var = pool * (1 - pool) * (1 / n_hit + 1 / n_other)
    if var <= 0:
        return 0.0
    return float(abs(p_hit - p_other) / np.sqrt(var))
