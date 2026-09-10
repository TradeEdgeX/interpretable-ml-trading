"""DSL expression parsing for research subset masks."""

from __future__ import annotations

import re
from typing import Callable, Tuple

import pandas as pd

OPS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "le": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "ge": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}

_TOKEN_RE = re.compile(
    r"^\s*(abs\()?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)?\s*(<=|>=|<|>|==|!=)\s*([-+0-9eE\.]+)\s*$"
)


def parse_atom(expr: str) -> Tuple[str, bool, str, float]:
    m = _TOKEN_RE.match(expr)
    if not m:
        raise ValueError(f"Cannot parse atom: {expr!r}")
    abs_, feature, op, value = m.groups()
    return feature, bool(abs_), op, float(value)


def eval_atom(
    df: pd.DataFrame, feature: str, take_abs: bool, op: str, value: float
) -> pd.Series:
    if feature not in df.columns:
        raise KeyError(f"Feature missing: {feature}")
    s = pd.to_numeric(df[feature], errors="coerce")
    if take_abs:
        s = s.abs()
    return OPS[op](s, value).fillna(False)


def parse_clause(expr: str) -> Callable[[pd.DataFrame], pd.Series]:
    parts = [
        p.strip()
        for p in re.split(r"\s+AND\s+", expr, flags=re.IGNORECASE)
        if p.strip()
    ]
    atoms = [parse_atom(p) for p in parts]

    def fn(df: pd.DataFrame) -> pd.Series:
        masks = [eval_atom(df, *a) for a in atoms]
        if not masks:
            return pd.Series(True, index=df.index)
        out = masks[0]
        for m in masks[1:]:
            out = out & m
        return out

    return fn


def build_calendar_mask(df: pd.DataFrame, window: str | None) -> pd.Series:
    if not window:
        return pd.Series(True, index=df.index)
    dt_col = None
    for c in ("datetime", "timestamp", "ts"):
        if c in df.columns:
            dt_col = c
            break
    if dt_col is None:
        raise KeyError("No datetime/timestamp column for calendar window")
    dt = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
    start_s, end_s = [x.strip() for x in window.split(",")]
    start = pd.to_datetime(start_s, utc=True)
    end = pd.to_datetime(end_s, utc=True)
    return ((dt >= start) & (dt < end)).fillna(False)
