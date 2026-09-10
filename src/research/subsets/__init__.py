from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import pandas as pd

from src.research.expr import build_calendar_mask, parse_clause


@dataclass(frozen=True)
class BaseSubset:
    def mask(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(True, index=df.index)


@dataclass(frozen=True)
class CalendarWindow:
    start_end: str

    def mask(self, df: pd.DataFrame) -> pd.Series:
        return build_calendar_mask(df, self.start_end)


@dataclass(frozen=True)
class FilterExpr:
    expr: str

    def mask(self, df: pd.DataFrame) -> pd.Series:
        return parse_clause(self.expr)(df)


@dataclass(frozen=True)
class LayerMask:
    """Placeholder: layer-specific mask resolved via layer_registry at CLI boundary."""

    strategy: str
    after_layer: str

    def mask(self, df: pd.DataFrame) -> pd.Series:
        from src.research.layer_registry import build_layer_mask

        return build_layer_mask(df, self.strategy, self.after_layer)


def combine_and(subsets: List[BaseSubset | CalendarWindow | FilterExpr | LayerMask], df: pd.DataFrame) -> pd.Series:
    out = pd.Series(True, index=df.index)
    for s in subsets:
        out = out & s.mask(df)
    return out
