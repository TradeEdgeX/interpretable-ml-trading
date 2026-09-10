"""Tests for research expression DSL."""

import pandas as pd
import pytest

from src.research.expr import build_calendar_mask, parse_clause


def test_parse_clause_and():
    df = pd.DataFrame({"a": [0.1, 0.5, 0.9], "b": [0.2, 0.6, 0.8]})
    mask = parse_clause("a >= 0.4 AND b <= 0.7")(df)
    assert list(mask) == [False, True, False]


def test_build_calendar_mask():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-06-01", "2025-06-01"], utc=True),
            "x": [1, 2],
        }
    )
    m = build_calendar_mask(df, "2024-01-01,2025-01-01")
    assert bool(m.iloc[0]) is True
    assert bool(m.iloc[1]) is False
