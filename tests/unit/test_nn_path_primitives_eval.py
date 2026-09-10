import numpy as np
import pandas as pd

from src.time_series_model.models.nn.path_primitives_eval import (
    evaluate_path_primitives,
)


def test_evaluate_path_primitives_basic_metrics() -> None:
    n = 200
    rng = np.random.default_rng(42)

    # True labels
    dir_y = rng.integers(0, 2, size=n)
    mfe = rng.normal(1.0, 0.2, size=n).clip(0, None)
    mae = rng.normal(0.8, 0.2, size=n).clip(0, None)
    t = rng.integers(0, 20, size=n).astype(float)
    mask = (mfe > 0.1).astype(float)

    # Predictions correlated with truth
    pred_dir = dir_y * 0.7 + rng.normal(0.0, 0.2, size=n)
    pred_mfe = mfe + rng.normal(0.0, 0.05, size=n)
    pred_mae = mae + rng.normal(0.0, 0.05, size=n)
    pred_t = t + rng.normal(0.0, 0.5, size=n)

    df = pd.DataFrame(
        {
            "dir_y": dir_y,
            "mfe_atr": mfe,
            "mae_atr": mae,
            "t_to_mfe": t,
            "mfe_valid": mask,
            "pred_dir": pred_dir,
            "pred_mfe_atr": pred_mfe,
            "pred_mae_atr": pred_mae,
            "pred_t_to_mfe": pred_t,
        }
    )

    metrics = evaluate_path_primitives(
        df=df,
        pred_cols={
            "dir": "pred_dir",
            "mfe_atr": "pred_mfe_atr",
            "mae_atr": "pred_mae_atr",
            "t_to_mfe": "pred_t_to_mfe",
        },
        true_cols={
            "dir_y": "dir_y",
            "mfe_atr": "mfe_atr",
            "mae_atr": "mae_atr",
            "t_to_mfe": "t_to_mfe",
        },
        mask_col="mfe_valid",
    )

    assert 0.0 <= metrics["dir_auc"] <= 1.0
    assert metrics["dir_auc"] > 0.7  # should be informative
    assert metrics["mfe_atr_spearman"] > 0.5
    assert metrics["mae_atr_spearman"] > 0.5
    assert metrics["mask_rate"] > 0.0
