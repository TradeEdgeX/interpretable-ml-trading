from __future__ import annotations

from unittest.mock import MagicMock

from src.time_series_model.live.decision_chain_debug import (
    _BAR_DEDUPE_KEYS,
    chain_debug_enabled,
    infer_block_stage,
    log_spot_no_intent,
    log_trend_no_intent,
    summarize_layers,
    throttle_allows,
)


def test_chain_debug_enabled_scope(monkeypatch):
    monkeypatch.delenv("MLBOT_CHAIN_DEBUG", raising=False)
    monkeypatch.delenv("MLBOT_TREND_CHAIN_DEBUG", raising=False)
    assert not chain_debug_enabled("trend")
    monkeypatch.setenv("MLBOT_TREND_CHAIN_DEBUG", "1")
    assert chain_debug_enabled("trend")


def test_throttle_dedupes_same_bar(monkeypatch):
    monkeypatch.setenv("MLBOT_TREND_CHAIN_DEBUG", "1")
    _BAR_DEDUPE_KEYS.clear()
    ts = "2026-05-19T12:00:00+00:00"
    assert throttle_allows("trend", "BTCUSDT", ts)
    assert not throttle_allows("trend", "BTCUSDT", ts)


def test_infer_block_stage_prefilter():
    funnel = {"prefilter": False, "prefilter_reason": "chop low"}
    assert infer_block_stage(funnel) == "prefilter_deny"
    assert summarize_layers(funnel)["prefilter"] == "fail"


def test_infer_block_stage_simple_deep_bear():
    funnel = {"simple_deep_bear": False, "weekly_ema_200_position": -0.1}
    assert infer_block_stage(funnel) == "simple_not_deep_bear"


def test_log_trend_no_intent_pcm_trace(monkeypatch):
    monkeypatch.setenv("MLBOT_TREND_CHAIN_DEBUG", "1")
    _BAR_DEDUPE_KEYS.clear()
    pcm = MagicMock()
    pcm._last_decide_trace = {"drop_slot": 1, "all_intents": 0}
    strat = MagicMock()
    strat._last_funnel = {
        "prefilter": False,
        "prefilter_reason": "x",
        "direction": False,
    }
    pcm._strategies = {"bpc": strat}
    log_trend_no_intent(
        "BTCUSDT",
        pcm,
        {"timestamp": "2026-05-19T12:00:00+00:00", "close": 100.0},
    )


def test_log_spot_no_intent_throttled(monkeypatch):
    monkeypatch.setenv("MLBOT_SPOT_CHAIN_DEBUG", "1")
    _BAR_DEDUPE_KEYS.clear()
    strat = MagicMock()
    strat._last_funnel = {"simple_deep_bear": False}
    log_spot_no_intent(
        "SOLUSDT",
        strat,
        {
            "timestamp": "2026-05-19T14:00:00+00:00",
            "weekly_ema_200_position": 0.05,
        },
    )
    log_spot_no_intent(
        "SOLUSDT",
        strat,
        {
            "timestamp": "2026-05-19T14:30:00+00:00",
            "weekly_ema_200_position": 0.05,
        },
    )
