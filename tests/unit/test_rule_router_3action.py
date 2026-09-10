import numpy as np
import pandas as pd

from src.time_series_model.rule.router_3action import (
    Rule3ActionConfig,
    compute_mode_3action,
)


def test_rule_router_3action_log1p_inverse_and_thresholds() -> None:
    # Construct 3 rows: NO_TRADE (low mfe), MEAN (good eff, quick ttm), TREND (strong dir_conf, high mfe, slow ttm)
    # Use log1p space for mfe/mae/ttm preds.
    df = pd.DataFrame(
        {
            "pred_dir_prob": [0.5, 0.55, 0.9],  # dir_conf: 0, 0.1, 0.8
            "pred_mfe_atr": np.log1p([0.1, 1.2, 1.5]),
            "pred_mae_atr": np.log1p([0.1, 0.8, 0.9]),
            "pred_t_to_mfe": np.log1p([5.0, 6.0, 20.0]),
        }
    )
    cfg = Rule3ActionConfig(
        mfe_min=0.4,
        eff_min=1.05,
        dir_conf_trend_min=0.25,
        mfe_trend_min=0.8,
        ttm_trend_min=8.0,
        eff_mean_min=1.15,
        ttm_mean_max=12.0,
    )
    out = compute_mode_3action(df, cfg=cfg, preds_in_log1p=True)
    assert out["mode"].tolist() == ["NO_TRADE", "MEAN", "TREND"]
    assert out["mode_action"].tolist() == [0, 1, 2]
