"""
Backward-compatible wrappers for DL sequence features.

Tests (and some older pipelines) import these entrypoints from:
`src.features.loader.dl_feature_wrappers`.

The canonical implementations live in:
`src.features.time_series.dl_sequence_features`.
"""

from __future__ import annotations

import pandas as pd

from src.features.time_series.dl_sequence_features import (
    compute_dl_sequence_features,
    compute_dl_sequence_features_from_series,
)

__all__ = [
    "compute_dl_sequence_features",
    "compute_dl_sequence_features_from_series",
]


