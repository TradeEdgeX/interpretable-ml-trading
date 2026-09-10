from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class MeanSystemHealth:
    """
    Gateable mean-system health summary (execution-agnostic).
    """

    mean_only_avg_total_return: float
    mean_only_avg_max_dd: float
    mean_only_sharpe_mean: float
    mean_system_viable: float  # 1.0 / 0.0

    def as_metrics(self) -> Dict[str, float]:
        return {
            "mean_system__mean_only_avg_total_return": float(
                self.mean_only_avg_total_return
            ),
            "mean_system__mean_only_avg_max_dd": float(self.mean_only_avg_max_dd),
            "mean_system__mean_only_sharpe_mean": float(self.mean_only_sharpe_mean),
            "mean_system__viable": float(self.mean_system_viable),
        }


def compute_mean_system_health(
    metrics: Dict[str, Any],
    *,
    max_dd_warn: float = 0.35,
    min_total_return: float = -0.2,
) -> MeanSystemHealth:
    """
    Minimal, deterministic checks for "mean-only can carry the system when TREND=0".

    This is intentionally conservative and stable:
    - We do not hard-fail based on Sharpe (but we record it).
    - We only require that mean-only isn't catastrophically bad.
    """
    tr = float(metrics.get("mean_only_avg_total_return", 0.0))
    dd = float(metrics.get("mean_only_avg_max_dd", 0.0))
    sh = float(metrics.get("mean_only_sharpe_mean", 0.0))
    viable = (
        1.0 if (dd <= float(max_dd_warn) and tr >= float(min_total_return)) else 0.0
    )
    return MeanSystemHealth(
        mean_only_avg_total_return=tr,
        mean_only_avg_max_dd=dd,
        mean_only_sharpe_mean=sh,
        mean_system_viable=viable,
    )
