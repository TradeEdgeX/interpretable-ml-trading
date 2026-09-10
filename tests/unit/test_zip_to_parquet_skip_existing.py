import zipfile
from pathlib import Path

import pandas as pd

from src.data_tools.zip_to_parquet import DataConverter


def _write_min_zip(path: Path, *, rows: list[list]):
    # rows are raw CSV rows without header in Binance-like schema:
    # agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker
    content = "\n".join(",".join(map(str, r)) for r in rows) + "\n"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(path.stem + ".csv", content)


def test_convert_skips_when_output_exists(tmp_path):
    in_dir = tmp_path / "agg_data"
    out_dir = tmp_path / "parquet_data"
    in_dir.mkdir()
    out_dir.mkdir()

    zip_path = in_dir / "BTCUSDT-aggTrades-2024-01.zip"
    _write_min_zip(
        zip_path,
        rows=[
            [1, 100.0, 0.5, 1, 1, 1704067200000, True],  # 2024-01-01T00:00:00Z
            [2, 101.0, 0.3, 2, 2, 1704067200500, False],  # same second
        ],
    )

    conv = DataConverter(str(in_dir), str(out_dir), backup_dir=None, force=False)

    r1 = conv.convert_all_files()
    assert r1["total_files"] == 1
    assert len(r1["converted_files"]) == 1
    assert len(r1.get("skipped_files", [])) == 0

    # Second run should skip
    r2 = conv.convert_all_files()
    assert r2["total_files"] == 1
    assert len(r2["converted_files"]) == 0
    assert len(r2.get("skipped_files", [])) == 1

    # Output parquet should exist and be readable
    out_file = Path(r1["converted_files"][0]["output_file"])
    assert out_file.exists()
    df = pd.read_parquet(out_file)
    assert {"timestamp", "price", "volume", "side"}.issubset(df.columns)
