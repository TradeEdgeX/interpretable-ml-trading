"""Tests for holdability tiers."""

from __future__ import annotations

from ashare.holdability import classify_holdability


def test_hold_ok():
    h = classify_holdability({"pe": 18.0, "roe": 12.0, "revenue_growth": 8.0})
    assert h["hold_tier"] == "hold_ok"


def test_hold_weak_pe():
    h = classify_holdability({"pe": -5.0, "roe": 10.0})
    assert h["hold_tier"] == "hold_weak"


def test_hold_caution_rev():
    h = classify_holdability({"pe": 20.0, "roe": 8.0, "revenue_growth": -10.0})
    assert h["hold_tier"] == "hold_caution"


def test_hold_unknown():
    h = classify_holdability({})
    assert h["hold_tier"] == "hold_unknown"
