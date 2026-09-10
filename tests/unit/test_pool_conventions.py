from pathlib import Path


def test_factor_ts_eval_default_pool_paths():
    from src.time_series_model.diagnostics import factor_ts_eval as fte

    assert fte._default_pool_b_dir("sr_reversal") == Path(
        "results/pools/sr_reversal/pool_b"
    )
    assert fte._default_pool_b_yaml_path("sr_reversal") == Path(
        "results/pools/sr_reversal/pool_b/features_pool_b.yaml"
    )
