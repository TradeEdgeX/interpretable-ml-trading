import numpy as np
import pytest

from src.time_series_model.rl.counterfactual_eval_3action import (
    _apply_mean_only_actions,
)


@pytest.mark.unit
def test_apply_mean_only_actions_replaces_trend_with_notrade():
    # Convention from Router3Action enum:
    # 0=NO_TRADE, 1=MEAN, 2=TREND (see _mode_to_action / Router3Action)
    a = np.asarray([0, 1, 2, 2, 1, 0], dtype=np.int64)
    out = _apply_mean_only_actions(a)
    assert out.dtype == np.int64
    assert out.tolist() == [0, 1, 0, 0, 1, 0]
