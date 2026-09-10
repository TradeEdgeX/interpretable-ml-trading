"""FeatureStore layer column-count health.

A ``tree_full`` layer that was grown with incremental merge can leave some
months as OHLCV stubs (or mid-generation TPC dumps) while later months have
the full registry. Wide IC then silently intersects to ~14 columns.

This module only *reports*. Repair is incremental rebuild of thin months.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pyarrow.parquet as pq


@dataclass(frozen=True)
class MonthCols:
    symbol: str
    month: str
    n_cols: int
    path: str


@dataclass
class LayerColumnSkew:
    layer: str
    n_files: int
    min_cols: int
    max_cols: int
    by_symbol: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)
    thin: List[MonthCols] = field(default_factory=list)
    histogram: Dict[int, int] = field(default_factory=dict)

    @property
    def skewed(self) -> bool:
        if self.n_files == 0 or self.max_cols <= 0:
            return False
        return self.min_cols < max(50, int(0.5 * self.max_cols))


def check_layer_column_health(root: Path | str, layer: str) -> LayerColumnSkew:
    """Scan monthly parquet files and summarize column-count skew."""
    return summarize_column_skew(layer, scan_layer_columns(Path(root), layer))


def scan_layer_columns(root: Path, layer: str) -> List[MonthCols]:
    layer_dir = Path(root) / layer
    if not layer_dir.is_dir():
        return []
    rows: List[MonthCols] = []
    for parquet in sorted(layer_dir.rglob("*.parquet")):
        rel = parquet.relative_to(layer_dir)
        parts = rel.parts
        symbol = parts[0] if parts else "?"
        month = parquet.stem
        n = len(pq.read_schema(parquet).names)
        rows.append(MonthCols(symbol, month, n, str(parquet)))
    return rows


def summarize_column_skew(layer: str, rows: Iterable[MonthCols]) -> LayerColumnSkew:
    rows = list(rows)
    if not rows:
        return LayerColumnSkew(layer=layer, n_files=0, min_cols=0, max_cols=0)
    ns = [r.n_cols for r in rows]
    hist: Dict[int, int] = defaultdict(int)
    per: Dict[str, List[int]] = defaultdict(list)
    for r in rows:
        hist[r.n_cols] += 1
        per[r.symbol].append(r.n_cols)
    max_c = max(ns)
    thin_cut = max(50, int(0.5 * max_c))
    thin = [r for r in rows if r.n_cols < thin_cut]
    by_symbol = {
        sym: (min(vs), max(vs), len(vs)) for sym, vs in sorted(per.items())
    }
    return LayerColumnSkew(
        layer=layer,
        n_files=len(rows),
        min_cols=min(ns),
        max_cols=max_c,
        by_symbol=by_symbol,
        thin=thin,
        histogram=dict(sorted(hist.items())),
    )


def format_skew_report(summary: LayerColumnSkew, *, thin_limit: int = 12) -> str:
    lines = [
        f"layer {summary.layer}: {summary.n_files} files, "
        f"cols {summary.min_cols}–{summary.max_cols}",
    ]
    for sym, (lo, hi, n) in summary.by_symbol.items():
        flag = "  THIN" if lo < max(50, int(0.5 * hi)) else ""
        lines.append(f"  {sym}: {n} months, cols {lo}–{hi}{flag}")
    if summary.skewed:
        lines.append(
            f"SKEW: {len(summary.thin)} month file(s) below "
            f"{max(50, int(0.5 * summary.max_cols))} cols "
            f"(wide IC intersection will collapse)."
        )
        for r in summary.thin[:thin_limit]:
            lines.append(f"  {r.symbol} {r.month}: {r.n_cols} cols")
        extra = len(summary.thin) - thin_limit
        if extra > 0:
            lines.append(f"  ... {extra} more")
        lines.append(
            "Repair: incremental build with features_all.yaml for those "
            "symbol/months (do not treat this layer as a wide pool until then)."
        )
    return "\n".join(lines)
