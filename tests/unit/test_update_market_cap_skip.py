from pathlib import Path

import pandas as pd

from scripts.update_market_cap import _should_skip_existing


def test_should_skip_static_if_fresh(tmp_path):
    p = Path(tmp_path) / "BTCUSDT.parquet"
    asof = pd.Timestamp.now(tz="UTC").floor("D")
    df = pd.DataFrame(
        {"market_cap_usd": [1.0]}, index=pd.DatetimeIndex([asof], name="date")
    )
    df.to_parquet(p)
    assert (
        _should_skip_existing(out_path=p, mode="static", max_age_days=1, force=False)
        is True
    )


def test_should_not_skip_static_if_too_old(tmp_path):
    p = Path(tmp_path) / "BTCUSDT.parquet"
    asof = pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=10)
    df = pd.DataFrame(
        {"market_cap_usd": [1.0]}, index=pd.DatetimeIndex([asof], name="date")
    )
    df.to_parquet(p)
    assert (
        _should_skip_existing(out_path=p, mode="static", max_age_days=1, force=False)
        is False
    )


def test_force_disables_skip(tmp_path):
    p = Path(tmp_path) / "BTCUSDT.parquet"
    asof = pd.Timestamp.now(tz="UTC").floor("D")
    df = pd.DataFrame(
        {"market_cap_usd": [1.0]}, index=pd.DatetimeIndex([asof], name="date")
    )
    df.to_parquet(p)
    assert (
        _should_skip_existing(out_path=p, mode="static", max_age_days=1, force=True)
        is False
    )
