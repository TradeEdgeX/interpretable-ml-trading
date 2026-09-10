"""Unit tests for OI timestamp alignment in refresh_funding_oi_data."""

from scripts.refresh_funding_oi_data import (
    _OI_RETENTION_DAYS,
    _align_ceil_ms,
    _align_floor_ms,
    _period_ms,
)


def test_period_ms_5m():
    assert _period_ms("5m") == 300_000


def test_align_floor_and_ceil_5m():
    period_ms = _period_ms("5m")
    misaligned = 1776649356869
    assert misaligned % period_ms != 0
    floor = _align_floor_ms(misaligned, period_ms)
    ceil = _align_ceil_ms(misaligned, period_ms)
    assert floor % period_ms == 0
    assert ceil % period_ms == 0
    assert floor < misaligned < ceil


def test_oi_retention_cap_is_below_typical_live_lookback():
    assert _OI_RETENTION_DAYS < 30
