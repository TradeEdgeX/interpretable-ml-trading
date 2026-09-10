import numpy as np
import pandas as pd

from src.time_series_model.models.nn.path_primitives_labels import (
    PathPrimitivesLabelConfig,
    compute_path_primitives_labels,
)


def test_compute_path_primitives_labels_shapes_and_masks() -> None:
    n = 200
    rng = np.random.default_rng(0)
    # Simple synthetic price series
    close = 100 + np.cumsum(rng.normal(0, 1, size=n))
    open_ = close + rng.normal(0, 0.1, size=n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0.2, 0.1, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.2, 0.1, size=n))
    atr = np.full(n, 1.5)  # constant ATR

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "atr": atr,
        }
    )

    cfg = PathPrimitivesLabelConfig(horizon_bars=10, entry_offset=1)
    out = compute_path_primitives_labels(df, cfg=cfg)

    for col in ["dir_y", "mfe_atr", "mae_atr", "t_to_mfe", "mfe_valid"]:
        assert col in out.columns
        assert len(out[col]) == n

    # There should be some valid samples (not all NaN)
    assert out["mae_atr"].notna().sum() > 0
    assert out["mfe_valid"].notna().sum() > 0


def test_compute_path_primitives_labels_group_col_no_leakage() -> None:
    # Two symbols back-to-back; labels should be computed within each symbol.
    n = 30
    df_a = pd.DataFrame(
        {
            "symbol": ["A"] * n,
            "open": np.linspace(100, 110, n),
            "high": np.linspace(100, 112, n),
            "low": np.linspace(99, 109, n),
            "close": np.linspace(100, 111, n),
            "atr": np.full(n, 1.0),
        }
    )
    df_b = pd.DataFrame(
        {
            "symbol": ["B"] * n,
            "open": np.linspace(200, 210, n),
            "high": np.linspace(200, 212, n),
            "low": np.linspace(199, 209, n),
            "close": np.linspace(200, 211, n),
            "atr": np.full(n, 1.0),
        }
    )
    df = pd.concat([df_a, df_b], axis=0, ignore_index=True)
    cfg = PathPrimitivesLabelConfig(horizon_bars=5, entry_offset=1)

    out = compute_path_primitives_labels(df, cfg=cfg, group_col="symbol")
    # At the end of each group, last horizons should be NaN.
    assert out.loc[n - 1, "mfe_atr"] != out.loc[n - 1, "mfe_atr"]  # NaN
    assert out.loc[2 * n - 1, "mfe_atr"] != out.loc[2 * n - 1, "mfe_atr"]  # NaN
