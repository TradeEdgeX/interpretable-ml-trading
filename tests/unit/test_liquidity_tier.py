"""Tests for liquidity tier helpers."""

from __future__ import annotations

from ashare.liquidity_tier import (
    classify_liquidity,
    enrich_items_liquidity,
    harvest_pref_for_item,
)


def test_liquidity_ok():
    h = classify_liquidity(6e8, gamma_ok=True)
    assert h["liquidity_tier"] == "liquidity_ok"
    assert h["liquidity_label"] == "充裕"


def test_liquidity_weak():
    h = classify_liquidity(5e7)
    assert h["liquidity_tier"] == "liquidity_weak"


def test_harvest_pref_rare():
    p = harvest_pref_for_item(
        {
            "strong_tier": "rsi20_amp7",
            "hold_tier": "hold_ok",
            "liquidity_tier": "liquidity_ok",
        }
    )
    assert "稀有超跌" in (p or "")


def test_harvest_pref_old_amp5_watch():
    p = harvest_pref_for_item(
        {
            "strong_tier": "rsi20_amp5",
            "hold_tier": "hold_ok",
            "liquidity_tier": "liquidity_ok",
        }
    )
    assert "观察" in (p or "")


def test_enrich_items_liquidity_passthrough():
    out = enrich_items_liquidity(
        [
            {
                "code": "600519",
                "avg_amount_20d": 2e8,
                "gamma_ok": True,
                "strong_tier": "rsi25_amp4",
            }
        ]
    )
    assert out[0]["liquidity_label"] == "够"
    assert out[0]["avg_amount_20d_yi"] == 2.0
