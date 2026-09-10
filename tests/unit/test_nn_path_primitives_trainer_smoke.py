import numpy as np
import pandas as pd

from src.time_series_model.models.nn.path_primitives_labels import (
    PathPrimitivesLabelConfig,
)
from src.time_series_model.models.nn.path_primitives_trainer import (
    TrainConfig,
    train_path_primitives_mlp,
)


def test_train_path_primitives_mlp_smoke(tmp_path) -> None:
    n = 400
    rng = np.random.default_rng(123)

    # Synthetic OHLCV-ish
    close = 100 + np.cumsum(rng.normal(0, 0.5, size=n))
    open_ = close + rng.normal(0, 0.05, size=n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0.15, 0.05, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.15, 0.05, size=n))
    atr = np.full(n, 1.0)

    # Simple features
    f1 = rng.normal(0, 1, size=n)
    f2 = rng.normal(0, 1, size=n)
    f3 = (close - pd.Series(close).rolling(5, min_periods=1).mean().to_numpy()) / 1.0

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "atr": atr,
            "f1": f1,
            "f2": f2,
            "f3": f3,
        }
    )

    cfg = TrainConfig(
        label_cfg=PathPrimitivesLabelConfig(
            horizon_bars=10,
            entry_offset=1,
            entry_price_col="open",
            high_col="high",
            low_col="low",
            close_col="close",
            atr_col="atr",
        ),
        epochs=2,
        batch_size=128,
        lr=5e-4,
        hidden=64,
        depth=2,
        dropout=0.0,
        device="cpu",
    )

    save_path = str(tmp_path / "model.pt")
    model, meta = train_path_primitives_mlp(
        df,
        feature_cols=["f1", "f2", "f3"],
        cfg=cfg,
        save_path=save_path,
    )

    assert model is not None
    assert meta["n_samples"] > 50
    assert meta["n_train"] > 0 and meta["n_val"] > 0
