"""Unit tests for mlbot_console ccxt_futures helpers."""

from __future__ import annotations

import pytest

from mlbot_console.services.ccxt_futures import is_plausible_binance_order_id


@pytest.mark.parametrize(
    ("order_id", "expected"),
    [
        ("123456789", True),
        ("1234567890123456789", True),  # 19-digit ETH USDT-M ids seen in prod
        (" 987654321 ", True),
        ("1234567890123456789012", True),  # 22 digits (max)
        ("12345678901234567890123", False),  # 23 digits
        ("", False),
        (None, False),
        ("abc123", False),
        ("12.34", False),
        ("   ", False),
    ],
)
def test_is_plausible_binance_order_id(order_id, expected):
    assert is_plausible_binance_order_id(order_id) is expected
