"""Research toolkit: composable stat kernels for R&D (layer-agnostic)."""

from src.research.expr import (
    OPS,
    build_calendar_mask,
    eval_atom,
    parse_atom,
    parse_clause,
)
from src.research.labels import derive_is_good_from_forward_rr

__all__ = [
    "OPS",
    "build_calendar_mask",
    "eval_atom",
    "parse_atom",
    "parse_clause",
    "derive_is_good_from_forward_rr",
]
