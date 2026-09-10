import numpy as np
import pandas as pd

from src.time_series_model.models.nn.path_primitives_labels import (
    PathPrimitivesLabelConfig,
)
from src.time_series_model.models.nn.path_primitives_trainer import (
    TrainConfig,
    train_path_primitives_mlp,
)
from src.time_series_model.models.nn.path_primitives_reporting import (
    evaluate_model_on_df,
    save_train_artifacts,
)


def test_reporting_saves_artifacts(tmp_path) -> None:
    n = 300
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 0.4, size=n))
    open_ = close + rng.normal(0, 0.05, size=n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0.1, 0.05, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.1, 0.05, size=n))
    atr = np.full(n, 1.0)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "atr": atr,
            # make near_sr condition available
            "dist_to_nearest_sr": np.full(n, 0.01),
            "f1": rng.normal(0, 1, size=n),
            "f2": rng.normal(0, 1, size=n),
        }
    )

    label_cfg = PathPrimitivesLabelConfig(horizon_bars=8, entry_offset=1)
    train_cfg = TrainConfig(
        label_cfg=label_cfg,
        epochs=1,
        batch_size=128,
        hidden=32,
        depth=2,
        dropout=0.0,
        device="cpu",
    )

    model, meta = train_path_primitives_mlp(
        df,
        feature_cols=["f1", "f2"],
        cfg=train_cfg,
        save_path=str(tmp_path / "model.pt"),
    )
    metrics, df_eval, extra = evaluate_model_on_df(
        model=model, df_features=df, feature_cols=["f1", "f2"], label_cfg=label_cfg
    )
    # conditional metrics should exist when near_sr mask is non-empty
    assert "near_sr__rate" in metrics
    assert metrics["near_sr__rate"] > 0.0
    # rolling IC artifacts should exist (tail preview)
    assert isinstance(extra, dict)
    assert "rolling_ic" in extra
    assert "preview_by_slice" in (extra["rolling_ic"] or {})
    assert "global" in (extra["rolling_ic"]["preview_by_slice"] or {})

    out_dir = tmp_path / "artifacts"
    if isinstance(extra, dict) and extra.get("rolling_ic") is not None:
        meta["rolling_ic"] = extra.get("rolling_ic")
    save_train_artifacts(
        out_dir=str(out_dir),
        model_path=str(tmp_path / "model.pt"),
        meta=meta,
        metrics=metrics,
        df_pred_sample=df_eval.tail(5),
    )

    assert (out_dir / "meta.json").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "pred_sample.csv").exists()
    assert (out_dir / "model_path.txt").exists()
    assert (out_dir / "report.html").exists()

    html = (out_dir / "report.html").read_text(encoding="utf-8")
    # high-signal sanity checks: title + conditional metric + sample section
    assert "Path Primitives - Report" in html
    assert "near_sr__rate" in html
    assert "Prediction sample" in html
