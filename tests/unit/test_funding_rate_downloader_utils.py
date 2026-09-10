from pathlib import Path

import pytest

from src.data_tools.download_funding_rate import (
    _BASE_URL_BY_MARGIN_KIND,
    FundingRateDownloader,
    _month_list,
    _normalize_symbol,
    _should_skip_month,
)


def test_month_list_spans_year_boundary():
    months = _month_list(2023, 11, 2024, 2)
    assert months == [(2023, 11), (2023, 12), (2024, 1), (2024, 2)]


def test_normalize_symbol_adds_usdt():
    assert _normalize_symbol("btc") == "BTCUSDT"
    assert _normalize_symbol("ETHUSDT") == "ETHUSDT"


def test_normalize_symbol_cm_maps_to_dapi_perp():
    assert _normalize_symbol("BTCUSDT", margin_kind="cm") == "BTCUSD_PERP"
    assert _normalize_symbol("btc", margin_kind="cm") == "BTCUSD_PERP"
    # Already-dapi symbols pass through unchanged.
    assert _normalize_symbol("SOLUSD_PERP", margin_kind="cm") == "SOLUSD_PERP"


def test_normalize_symbol_cm_unknown_symbol_raises():
    with pytest.raises(ValueError):
        _normalize_symbol("DOGEUSDT", margin_kind="cm")


def test_downloader_defaults_um_base_url(tmp_path):
    dl = FundingRateDownloader(data_dir=tmp_path / "zip", parquet_dir=tmp_path / "pq")
    assert dl.base_url == _BASE_URL_BY_MARGIN_KIND["um"]


def test_downloader_cm_uses_coin_margin_base_url(tmp_path):
    dl = FundingRateDownloader(
        data_dir=tmp_path / "zip", parquet_dir=tmp_path / "pq", margin_kind="cm"
    )
    assert dl.base_url == _BASE_URL_BY_MARGIN_KIND["cm"]
    assert "futures/cm/monthly/fundingRate" in dl.base_url


def test_should_skip_month_prefers_parquet(tmp_path):
    z = Path(tmp_path) / "x.zip"
    p = Path(tmp_path) / "x.parquet"
    z.write_bytes(b"0" * 2000)
    p.write_bytes(b"1" * 10)
    assert _should_skip_month(zip_path=z, parquet_path=p, force=False) is True


def test_should_skip_month_uses_zip_if_no_parquet(tmp_path):
    z = Path(tmp_path) / "x.zip"
    z.write_bytes(b"0" * 2000)
    assert _should_skip_month(zip_path=z, parquet_path=None, force=False) is True


def test_force_disables_skip(tmp_path):
    z = Path(tmp_path) / "x.zip"
    z.write_bytes(b"0" * 2000)
    assert _should_skip_month(zip_path=z, parquet_path=None, force=True) is False
