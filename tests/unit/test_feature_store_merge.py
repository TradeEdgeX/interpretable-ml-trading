import pandas as pd

from src.feature_store.feature_store import FeatureStore, FeatureStoreSpec


def test_feature_store_write_month_merge_existing_adds_columns(tmp_path) -> None:
    store = FeatureStore(tmp_path)
    spec = FeatureStoreSpec(layer="L", symbol="AAA", timeframe="1D")

    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    df_a = pd.DataFrame({"open": [1, 2, 3], "a": [10.0, 11.0, 12.0]}, index=idx)
    df_b = pd.DataFrame({"open": [1, 2, 3], "b": [20.0, 21.0, 22.0]}, index=idx)

    store.write_month(spec, "2025-01", df_a, overwrite=False, metadata={"v": 1})
    store.write_month(
        spec,
        "2025-01",
        df_b,
        overwrite=False,
        merge_existing=True,
        metadata={"v": 2},
    )

    out = store.read_month(spec, "2025-01")
    assert "a" in out.columns
    assert "b" in out.columns
    assert out.loc[idx[0], "a"] == 10.0
    assert out.loc[idx[0], "b"] == 20.0
