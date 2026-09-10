from src.feature_store.layer_health import (
    MonthCols,
    format_skew_report,
    summarize_column_skew,
)


def test_ohlcv_stub_months_flag_skew() -> None:
    rows = [
        MonthCols("BTCUSDT", "2022-01", 29, "btc/2022-01.parquet"),
        MonthCols("BTCUSDT", "2024-01", 949, "btc/2024-01.parquet"),
        MonthCols("ETHUSDT", "2022-01", 949, "eth/2022-01.parquet"),
        MonthCols("ETHUSDT", "2024-01", 957, "eth/2024-01.parquet"),
    ]
    summary = summarize_column_skew("features_tree_full_120T_x", rows)
    assert summary.skewed
    assert summary.min_cols == 29
    assert summary.max_cols == 957
    assert summary.by_symbol["BTCUSDT"][0] == 29
    text = format_skew_report(summary)
    assert "SKEW" in text
    assert "BTCUSDT 2022-01: 29 cols" in text


def test_aligned_layer_not_skewed() -> None:
    rows = [
        MonthCols("ETHUSDT", "2022-01", 949, "a"),
        MonthCols("ETHUSDT", "2023-01", 957, "b"),
        MonthCols("BNBUSDT", "2022-01", 949, "c"),
    ]
    summary = summarize_column_skew("features_tree_full_120T_x", rows)
    assert not summary.skewed
    assert "SKEW" not in format_skew_report(summary)
