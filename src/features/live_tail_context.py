"""Thread-local context for live last-row feature computation.

The live path (``IncrementalFeatureComputer``) only consumes the **last row** of
each computed feature snapshot. Expensive per-bar-loop features (notably the SR
strength stack, ``compute_sr_strength_max_from_series``) are strictly causal with
a small internal lookback window, yet today they recompute the *entire* history
(~1800 bars) every 2h bar close — of which only the tail is ever read live.

When a bounded tail is active, such features may compute only the last
``tail`` bars and leave earlier rows at their default. This is bit-identical for
the last output row (and the last ``tail - max_internal_window`` rows, which
covers every downstream consumer's lookback), while cutting the Python-loop cost
by ~1 order of magnitude.

Backtest / FeatureStore builds NEVER enter this context, so their full-history
behavior is unchanged. The context is thread-local and restored on exit, so it
never leaks across compute calls.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator, Optional

_ctx = threading.local()


def get_live_feature_tail() -> Optional[int]:
    """Return the active live tail (number of bars) or ``None`` (full history)."""
    return getattr(_ctx, "tail", None)


def get_live_tick_tail_rows(
    *,
    ofci_window: int = 100,
    percentile_window: int = 540,
    margin: int = 200,
    min_rows: int = 5000,
) -> Optional[int]:
    """When live bar-tail is active, return tick rows to keep for causal tick features.

    OFCI percentile is computed on the tick index with rolling windows; the last
    value only depends on the trailing ``ofci_window + percentile_window`` rows.
    """
    if get_live_feature_tail() is None:
        return None
    return max(ofci_window + percentile_window + margin, min_rows)


def live_tail_slice_len(
    n_rows: int,
    *,
    warmup: int = 0,
) -> Optional[int]:
    """Return slice length for a series of ``n_rows`` under the active live tail.

    Returns ``None`` when the context is inactive or the series is already short
    enough that no slicing is needed.
    """
    tail = get_live_feature_tail()
    if tail is None or n_rows <= tail:
        return None
    return min(n_rows, int(tail) + max(0, int(warmup)))


def restore_live_tail_frame(
    result: Any,
    full_index: Any,
    *,
    fill_value: float = 0.0,
) -> Any:
    """Pad a positionally-sliced live-tail result back onto ``full_index``.

    ``DataFrame.reindex`` rejects duplicate labels. Live 1min OFCI ticks keep
    buy+sell rows that share a minute timestamp, so restore must be positional
    (matching ``iloc[-slice_len:]``), not label-based reindex.
    """
    import numpy as np
    import pandas as pd

    n_full = len(full_index)
    n_tail = len(result)
    if n_tail == n_full:
        out = result.copy()
        out.index = full_index
        return out
    if n_tail > n_full:
        out = result.iloc[-n_full:].copy()
        out.index = full_index
        return out
    pad = n_full - n_tail
    data = {
        c: np.concatenate(
            [
                np.full(pad, fill_value, dtype=float),
                pd.to_numeric(result[c], errors="coerce").to_numpy(dtype=float),
            ]
        )
        for c in result.columns
    }
    return pd.DataFrame(data, index=full_index)


@contextmanager
def live_feature_tail(tail: Optional[int]) -> Iterator[None]:
    """Activate bounded tail compute for tail-aware features within this block.

    Args:
        tail: Number of most-recent bars to compute. ``None`` / ``<= 0`` disables
            (full-history compute, i.e. current behavior).
    """
    prev = getattr(_ctx, "tail", None)
    _ctx.tail = tail if (tail and tail > 0) else None
    try:
        yield
    finally:
        _ctx.tail = prev
