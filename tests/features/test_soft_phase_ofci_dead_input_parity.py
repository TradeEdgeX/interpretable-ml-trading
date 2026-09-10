"""soft_phase ignores ofci_pct; DAG must not hard-depend on ofci_pct_f.

Workstream B1/B3 of live feature compute speedup (2026-07-14):
- ofci_pct is assigned in soft_phase core then never used in scores
- soft_phase only needs raw ``vpin``, not the vpin_features_f mega-selector
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.bpc_features import (
    compute_bpc_soft_phase_from_series,
    compute_tpc_soft_phase_from_series,
)
from time_series_model.live.live_feature_plan import extract_features_from_archetypes


def _load_fs_slice(
    layer: Path,
    symbol: str = "BTCUSDT",
    months: int = 6,
) -> pd.DataFrame:
    root = layer / symbol / "120T"
    files = sorted(root.glob("*.parquet"))
    if len(files) < 1:
        pytest.skip(f"no FS months under {root}")
    cols = [
        "close",
        "high",
        "low",
        "atr",
        "volume",
        "cvd_change_5",
        "vpin",
        "ofci_pct",
        "bb_width_normalized_pct",
        "ema_1200_position",
    ]
    frames = []
    for f in files[-months:]:
        import pyarrow.parquet as pq

        names = set(pq.read_schema(f).names)
        use = [c for c in cols if c in names]
        frames.append(pd.read_parquet(f, columns=use))
    df = pd.concat(frames).sort_index()
    missing = [
        c for c in ("close", "high", "low", "atr", "volume", "vpin") if c not in df
    ]
    if missing:
        pytest.skip(f"FS slice missing {missing}")
    return df


def _fs_layer() -> Path:
    preferred = Path("feature_store/features_tpc_120T_df79a67ec8")
    if preferred.exists():
        return preferred
    cands = sorted(Path("feature_store").glob("features_tpc_120T_*"))
    if not cands:
        pytest.skip("no TPC FeatureStore layer")
    return cands[0]


def test_tpc_soft_phase_identical_with_or_without_ofci() -> None:
    df = _load_fs_slice(_fs_layer())
    kw = dict(
        close=df["close"],
        high=df["high"],
        low=df["low"],
        atr=df["atr"],
        volume=df["volume"],
        cvd_change_5=df["cvd_change_5"] if "cvd_change_5" in df else None,
        vpin=df["vpin"],
        bb_width_normalized=(
            df["bb_width_normalized_pct"] if "bb_width_normalized_pct" in df else None
        ),
        ema_1200_position=(
            df["ema_1200_position"] if "ema_1200_position" in df else None
        ),
    )
    with_ofci = compute_tpc_soft_phase_from_series(
        **kw, ofci_pct=df["ofci_pct"] if "ofci_pct" in df else None
    )
    without = compute_tpc_soft_phase_from_series(**kw, ofci_pct=None)
    assert list(with_ofci.columns) == list(without.columns)
    for c in with_ofci.columns:
        a = with_ofci[c].to_numpy(dtype=float)
        b = without[c].to_numpy(dtype=float)
        assert np.array_equal(
            a, b, equal_nan=True
        ), f"col {c} changed when ofci dropped"


def test_bpc_soft_phase_identical_with_or_without_ofci() -> None:
    df = _load_fs_slice(_fs_layer())
    kw = dict(
        close=df["close"],
        high=df["high"],
        low=df["low"],
        atr=df["atr"],
        volume=df["volume"],
        cvd_change_5=df["cvd_change_5"] if "cvd_change_5" in df else None,
        vpin=df["vpin"],
        bb_width_normalized=(
            df["bb_width_normalized_pct"] if "bb_width_normalized_pct" in df else None
        ),
        ema_1200_position=(
            df["ema_1200_position"] if "ema_1200_position" in df else None
        ),
    )
    with_ofci = compute_bpc_soft_phase_from_series(
        **kw, ofci_pct=df["ofci_pct"] if "ofci_pct" in df else None
    )
    without = compute_bpc_soft_phase_from_series(**kw, ofci_pct=None)
    for c in with_ofci.columns:
        a = with_ofci[c].to_numpy(dtype=float)
        b = without[c].to_numpy(dtype=float)
        assert np.array_equal(
            a, b, equal_nan=True
        ), f"col {c} changed when ofci dropped"


def test_soft_phase_dag_omits_ofci_and_vpin_features_selector() -> None:
    import yaml

    deps = yaml.safe_load(Path("config/feature_dependencies.yaml").read_text())
    feats = deps["features"]
    for node in ("tpc_soft_phase_f", "bpc_soft_phase_f"):
        dlist = feats[node]["dependencies"]
        assert "ofci_pct_f" not in dlist, node
        assert "vpin_features_f" not in dlist, node
        assert "vpin_base_aligned_features_f" in dlist, node
        maps = feats[node].get("column_mappings") or {}
        assert "ofci_pct" not in maps, node


def test_tpc_live_plan_drops_ofci_and_trade_cluster_bloat() -> None:
    """After soft_phase dep slim: TPC must not pull ofci or trade_cluster family."""
    cols, nodes = extract_features_from_archetypes(
        Path("config/strategies/tpc/archetypes"),
        feature_deps_path=Path("config/feature_dependencies.yaml"),
    )
    assert "ofci_pct_f" not in nodes
    assert "ofci_pct" not in cols
    assert not any("trade_cluster" in n for n in nodes), nodes
    # Still need raw VPIN for soft_phase confirm paths
    assert "vpin_base_aligned_features_f" in nodes
    assert "vpin_features_f" not in nodes
    # Compact plan (was ~42 with mega selector + trade_cluster)
    assert len(nodes) < 25, f"TPC plan still bloated: {len(nodes)} {nodes}"
