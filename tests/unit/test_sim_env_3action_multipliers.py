import pandas as pd
import pytest

from src.time_series_model.rl.sim_env_3action import (
    SimEnvConfig,
    simulate_3action_episode,
)
from src.time_series_model.rl.bc_dataset import Router3Action


@pytest.mark.unit
def test_simulate_3action_episode_mean_multiplier_scales_exposure():
    df = pd.DataFrame(
        {
            "ret_mean": [0.10, 0.10, 0.10],
            "ret_trend": [0.10, 0.10, 0.10],
        }
    )
    actions = [int(Router3Action.MEAN)] * len(df)
    cfg = SimEnvConfig(
        entry_delay=0, cost_per_turnover=0.0, slippage_bps=0.0, initial_equity=1.0
    )

    out_full = simulate_3action_episode(df, actions=actions, cfg=cfg)
    out_half = simulate_3action_episode(
        df, actions=actions, cfg=cfg, mean_multiplier=[0.5, 0.5, 0.5]
    )

    # mean target exposure = base_exposure(1.0)*mean_exposure(0.8)=0.8
    # so half multiplier => exposure ~0.4 each step (entry_delay=0).
    assert out_full["exposure"].iloc[0] == pytest.approx(0.8, abs=1e-12)
    assert out_half["exposure"].iloc[0] == pytest.approx(0.4, abs=1e-12)
    # equity should be strictly lower with half exposure for positive returns
    assert float(out_full["equity"].iloc[-1]) > float(out_half["equity"].iloc[-1])
