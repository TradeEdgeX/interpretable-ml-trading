import numpy as np
import pandas as pd

from src.time_series_model.models.nn.path_primitives_conditions import (
    SRFuseConditionConfig,
    compute_near_sr_mask,
)


def test_compute_near_sr_mask_basic() -> None:
    n = 100
    df = pd.DataFrame(
        {
            "close": np.full(n, 100.0),
            "atr": np.full(n, 2.0),
            # 1% distance => abs_dist = 1.0, norm_atr = 0.5
            "dist_to_nearest_sr": np.full(n, 0.01),
        }
    )
    mask = compute_near_sr_mask(df, cfg=SRFuseConditionConfig(max_dist_atr=1.0))
    assert mask.all()

    mask2 = compute_near_sr_mask(df, cfg=SRFuseConditionConfig(max_dist_atr=0.1))
    assert (~mask2).all()
