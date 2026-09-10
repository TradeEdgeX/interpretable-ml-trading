"""
Generic narrow selector utilities for composite/selector DAG nodes.

Goal: eliminate pass_full_df:true for pure selector nodes by reassembling a slim output
DataFrame from Series inputs.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from src.features.registry import register_feature


@register_feature("select_columns_from_series", category="selector")
def select_columns_from_series(*, output_columns: Optional[list[str]] = None, **series_kwargs) -> pd.DataFrame:
    """
    Assemble a DataFrame from provided Series inputs.

    This is used for selector/composite nodes where `required_columns == output_columns`.

    Args:
        output_columns: Optional explicit ordering of output columns. If omitted, uses the
            keys in `series_kwargs` in insertion order.
        **series_kwargs: keyword->Series mapping, where each key is an output column name.

    Returns:
        DataFrame with the requested columns and aligned index.
    """
    if not series_kwargs:
        return pd.DataFrame()

    if output_columns is None:
        output_columns = list(series_kwargs.keys())

    # Use the first provided series as the index reference
    first = next(iter(series_kwargs.values()))
    idx = first.index

    out: Dict[str, pd.Series] = {}
    for col in output_columns:
        s = series_kwargs.get(col)
        if s is None:
            # If missing, create empty series aligned to idx (shouldn't happen if required_columns are enforced)
            out[col] = pd.Series(index=idx, dtype=float, name=col)
        else:
            out[col] = s.rename(col).reindex(idx)

    return pd.DataFrame(out, index=idx)


