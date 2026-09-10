import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.loader.strategy_feature_loader import StrategyFeatureLoader


def _make_price_df(n=200):
    idx = pd.date_range("2024-01-01", periods=n, freq="1H")
    prices = np.linspace(100, 110, n)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices + 0.1,  # slight shift
            "volume": np.linspace(1000, 2000, n),
        },
        index=idx,
    )
    return df


def _run_and_assert_no_inf(requested_features, expected_prefixes=None):
    expected_prefixes = expected_prefixes or requested_features
    df = _make_price_df()
    with tempfile.TemporaryDirectory() as tmp:
        loader = StrategyFeatureLoader(
            feature_deps_path="config/feature_dependencies.yaml",
            cache_dir=Path(tmp),
            use_disk_cache=False,
            use_memory_cache=False,
            max_workers=1,
            parallel_backend="thread",
        )
        out = loader.load_features_from_requested(
            df,
            requested_features=requested_features,
            fit=False,
        )
    # Determine expected output columns from feature deps
    features_cfg = loader.feature_deps.get("features", {})
    expected_cols = []
    for f in requested_features:
        info = features_cfg.get(f, {})
        expected_cols.extend(info.get("output_columns", [f]))
    # Fallback: also match by provided prefixes (handles parent feature aliasing)
    cols = [c for c in expected_cols if c in out.columns]
    if not cols:
        cols = [
            c for c in out.columns if any(c.startswith(p) for p in expected_prefixes)
        ]
    assert cols, f"No output columns produced for {requested_features}"
    values = out[cols].to_numpy()
    assert not np.isinf(values).any(), f"Found inf/-inf in {cols}"


def test_hilbert_cvd_env_no_inf():
    # hilbert_cvd_* are outputs of hilbert_advanced_f
    _run_and_assert_no_inf(
        ["hilbert_advanced_f"],
        expected_prefixes=["hilbert_cvd_env", "hilbert_cvd_price_env_ratio"],
    )


def test_garch_and_extended_vol_no_inf():
    _run_and_assert_no_inf(["garch_features_f", "extended_volatility_features_f"])


def test_evt_features_no_inf():
    _run_and_assert_no_inf(["evt_features_f"])
