from src.time_series_model.strategies.models.feature_direction import (
    expand_invert_features,
)


def test_expand_invert_features_expands_node_to_output_columns():
    feature_deps = {
        "features": {
            "trend_r2_50_f": {"output_columns": ["trend_r2_50"]},
            "macd_f": {"output_columns": ["macd", "macd_signal", "macd_histogram"]},
        }
    }
    out = expand_invert_features(["trend_r2_50_f", "macd_f"], feature_deps=feature_deps)
    assert out == ["trend_r2_50", "macd", "macd_signal", "macd_histogram"]


def test_expand_invert_features_keeps_column_names_and_dedups_stably():
    feature_deps = {"features": {"foo_f": {"output_columns": ["a", "b"]}}}
    out = expand_invert_features(["a", "foo_f", "a", "b"], feature_deps=feature_deps)
    assert out == ["a", "b"]


def test_expand_invert_features_unknown_entries_pass_through():
    out = expand_invert_features(
        ["nonexistent_col", "nonexistent_node_f"], feature_deps={}
    )
    assert out == ["nonexistent_col", "nonexistent_node_f"]
