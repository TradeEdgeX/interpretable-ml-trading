import pandas as pd

from src.time_series_model.diagnostics.survival_head_mlp import (
    SurvivalHeadTrainConfig,
    train_survival_head,
)


def test_survival_head_train_smoke() -> None:
    ts = pd.date_range("2025-01-01", periods=120, freq="4H", tz="UTC")
    rows = []
    labs = []
    for sym in ["BTCUSDT", "ETHUSDT"]:
        for i, t in enumerate(ts):
            rows.append(
                {
                    "symbol": sym,
                    "timestamp": t,
                    "mode": "MEAN" if i % 3 == 0 else "NO_TRADE",
                    "head_dir_score": 0.1,
                    "head_mfe_atr": 0.8,
                    "head_mae_atr": 0.4,
                    "head_t_to_mfe": 10.0,
                    "drawdown": 0.05,
                }
            )
            # Make later samples less survivable for ETH
            y = 0 if (sym == "ETHUSDT" and i > 80) else 1
            labs.append({"symbol": sym, "timestamp": t, "y_surv": y})
    df_logs = pd.DataFrame(rows)
    df_labels = pd.DataFrame(labs)

    cfg = SurvivalHeadTrainConfig(epochs=2, batch_size=128, train_ratio=0.7)
    metrics, preds_df, curves, roc_png, pr_png, cal_png = train_survival_head(
        df_logs, df_labels, cfg=cfg
    )
    assert metrics["n"] > 0
    assert "auc_test" in metrics
    assert "calibration_bins" in curves
    assert "survival_prob" in preds_df.columns
    assert isinstance(roc_png, (bytes, bytearray))
