import pandas as pd
import numpy as np

from scripts.train_strategy_pipeline import drop_inf_rows


def test_drop_inf_rows_removes_only_inf_rows():
    df = pd.DataFrame(
        {
            "f1": [1.0, np.inf, 3.0, -np.inf, 5.0],
            "f2": [1.0, 2.0, np.inf, 4.0, 5.0],
            "y": [0, 1, 0, 1, 0],
        }
    )

    cleaned = drop_inf_rows(df, ["f1", "f2"])

    # rows with any inf/-inf should be removed (index 1,2,3)
    assert len(cleaned) == 2
    assert cleaned.index.tolist() == [0, 4]
    # values should remain finite
    assert np.isfinite(cleaned[["f1", "f2"]].values).all()
