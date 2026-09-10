"""Label derivation helpers for research parquet inputs."""

from __future__ import annotations

import pandas as pd

RR_COLUMN_CANDIDATES = (
    "bpc_impulse_return_atr",
    "forward_rr",
    "rr",
    "return_atr",
)


def find_rr_column(df: pd.DataFrame) -> str | None:
    for candidate in RR_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def derive_is_good_from_forward_rr(
    df: pd.DataFrame,
    *,
    threshold: float = -0.8,
    label_col: str = "is_good",
    rr_col: str | None = None,
) -> str:
    """Add ``label_col`` (1=good, 0=bad) from RR column if missing. Returns rr_col used."""
    if label_col in df.columns:
        col = rr_col or find_rr_column(df)
        if col is None:
            raise KeyError(f"Label column '{label_col}' missing and no RR column found")
        return col
    col = rr_col or find_rr_column(df)
    if col is None:
        raise KeyError(
            f"Cannot derive '{label_col}': no RR column among {RR_COLUMN_CANDIDATES}"
        )
    df[label_col] = (pd.to_numeric(df[col], errors="coerce") >= threshold).astype(int)
    return col
