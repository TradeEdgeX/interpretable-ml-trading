import pandas as pd

from scripts.train_strategy_pipeline import determine_feature_columns
from src.time_series_model.strategy_config.loader import FeaturePipelineConfig


def test_tree_determine_feature_columns_exclude_columns_drops_atr():
    # df keeps atr available (labels/backtest may need it)
    df = pd.DataFrame(
        {
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1.0, 1.0],
            "atr": [0.1, 0.2],
            "rsi": [50.0, 55.0],
            "signal": [0.0, 0.0],
        }
    )

    cfg = FeaturePipelineConfig(
        requested_features=[],
        exclude_columns=["atr"],
    )

    cols = determine_feature_columns(df, cfg)
    assert "atr" not in cols
    assert "rsi" in cols


def test_tree_determine_feature_columns_scoped_to_requested(monkeypatch):
    """Parquet passthrough columns (e.g. cvd_change_5_normalized) must not enter the model."""
    df = pd.DataFrame(
        {
            "box_stability_60": [0.1, 0.2],
            "cvd_change_5_normalized": [0.01, -0.02],
        }
    )
    deps = {
        "box_structure_f": {"output_columns": ["box_stability_60"]},
        "cvd_basic_f": {"output_columns": ["cvd_change_5_normalized"]},
    }
    monkeypatch.setattr(
        "scripts.train_strategy_pipeline._load_valid_output_columns",
        lambda *a, **k: {"box_stability_60", "cvd_change_5_normalized"},
    )
    monkeypatch.setattr(
        "src.research.stat_kernels.ic_prune.load_feature_deps",
        lambda *a, **k: deps,
    )

    cfg = FeaturePipelineConfig(
        requested_features=["atr_f", "box_stability_60"],
        exclude_columns=["atr"],
    )
    cols = determine_feature_columns(df, cfg)
    assert cols == ["box_stability_60"]
