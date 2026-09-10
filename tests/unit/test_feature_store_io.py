import pandas as pd

from src.time_series_model.models.nn.feature_store_io import load_feature_store


def test_load_feature_store_dir_infers_symbol(tmp_path) -> None:
    df_a = pd.DataFrame({"timestamp": [1, 2], "x": [10.0, 11.0]})
    df_b = pd.DataFrame({"timestamp": [1], "x": [20.0]})
    df_a.to_parquet(tmp_path / "features_AAA.parquet", index=False)
    df_b.to_parquet(tmp_path / "features_BBB.parquet", index=False)

    df = load_feature_store(str(tmp_path))
    assert set(df["symbol"].astype(str).unique()) == {"AAA", "BBB"}
    assert "x" in df.columns
