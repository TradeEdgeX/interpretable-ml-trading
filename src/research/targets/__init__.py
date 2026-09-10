from dataclasses import dataclass


@dataclass(frozen=True)
class LabelTarget:
    col: str = "success_no_rr_extreme"


@dataclass(frozen=True)
class ForwardRRTarget:
    col: str = "forward_rr"
    horizon: int = 1


@dataclass(frozen=True)
class SnotioTarget:
    """Mean R per trade target (entry layer)."""

    exec_config_path: str | None = None


@dataclass(frozen=True)
class RMultipleTarget:
    """Grid / event backtest R-multiple target."""

    grid_config: str | None = None
