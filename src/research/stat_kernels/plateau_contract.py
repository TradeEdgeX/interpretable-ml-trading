"""Re-export plateau stability contract from scripts (single source until full migration)."""

from scripts.plateau_stability import (  # noqa: F401
    PlateauRange,
    decide_plateau_update,
    plateau_range_from_dict,
    plateau_range_to_dict,
    ranges_overlap,
)

__all__ = [
    "PlateauRange",
    "ranges_overlap",
    "decide_plateau_update",
    "plateau_range_from_dict",
    "plateau_range_to_dict",
]
