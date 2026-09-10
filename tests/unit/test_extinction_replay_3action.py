import pandas as pd

from src.time_series_model.diagnostics.extinction_replay_3action import (
    ExtinctionReplayConfig,
    run_extinction_replay_3action,
)


def test_extinction_replay_produces_labels_and_report() -> None:
    # 2 symbols, 6 steps each
    ts = pd.date_range("2025-01-01", periods=6, freq="4H", tz="UTC")
    rows = []
    for sym in ["BTCUSDT", "ETHUSDT"]:
        for i, t in enumerate(ts):
            rows.append(
                {
                    "symbol": sym,
                    "timestamp": t,
                    "mode": "MEAN" if i % 2 == 0 else "NO_TRADE",
                    # Make ETH materially unsafe so y_surv flips to 0 for some t.
                    "ret_mean": -0.3 if (sym == "ETHUSDT" and i >= 2) else 0.01,
                    "ret_trend": 0.0,
                }
            )
    df = pd.DataFrame(rows)

    cfg = ExtinctionReplayConfig(
        survival_horizon_bars=2, equity_floor_frac=0.95, dd_floor=0.8
    )
    report, sim, labels = run_extinction_replay_3action(df, cfg=cfg)

    assert report["ok"] is True
    assert report["n_symbols"] == 2
    assert "extinction_rate" in report
    assert len(sim) == len(df)
    assert len(labels) == len(df)
    assert "y_surv" in labels.columns
    # ETH should become less survivable due to negative returns in later steps.
    eth = labels[labels["symbol"] == "ETHUSDT"]["y_surv"].to_list()
    assert 0 in eth
