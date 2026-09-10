from __future__ import annotations

import pandas as pd

import pytest

from features.btc_roc90 import label_roc90
from mlbot_console.services.regime_hero import (
    _from_daily_frame,
    _weekly_state,
    build_btc_direction_hero,
    build_clocks_cross,
    build_weekly_sleeve_snapshot,
)


def test_label_roc90_thresholds():
    assert label_roc90(0.15) == "bull"
    assert label_roc90(0.149) == "neutral"
    assert label_roc90(-0.15) == "bear"


def test_from_daily_frame_bull():
    idx = pd.date_range("2025-01-01", periods=100, freq="D", tz="UTC")
    close = pd.Series([100.0] * 90 + [120.0] * 10, index=idx)
    daily = pd.DataFrame({"close": close})
    parsed = _from_daily_frame(daily)
    assert parsed is not None
    assert parsed["regime"] == "bull"
    assert parsed["label"] == "牛"
    assert parsed["roc_90"] is not None and parsed["roc_90"] > 0.15


def test_weekly_state_left_deep_bear():
    st = _weekly_state(0.01)
    assert st["id"] == "above"


def test_hero_fallbacks_to_xsection_chop(monkeypatch):
    monkeypatch.setattr(
        "mlbot_console.services.macro_env.ensure_panel_daily",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "mlbot_console.services.macro_env._load_daily_frame",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "mlbot_console.services.macro_env._load_crypto_daily_from_console",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "mlbot_console.services.crypto_xsection_metrics.market_regime",
        lambda: {"regime": "chop", "roc_90": 0.02},
    )
    hero = build_btc_direction_hero(weekly_ema_200_position=-0.08)
    assert hero["weekly"]["id"] == "deep"
    assert hero["available"] is True
    assert hero["regime"] == "neutral"
    assert hero["label"] == "未极端"
    assert "震荡" not in str(hero["label"])
    assert hero["roc_90"] == pytest.approx(0.02)


def test_hero_headline_is_roc90_not_chop(monkeypatch):
    idx = pd.date_range("2025-01-01", periods=100, freq="D", tz="UTC")
    close = pd.Series([100.0] * 100, index=idx)
    monkeypatch.setattr(
        "mlbot_console.services.macro_env.ensure_panel_daily",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "mlbot_console.services.macro_env._load_daily_frame",
        lambda *a, **k: pd.DataFrame({"close": close}),
    )
    hero = build_btc_direction_hero(weekly_ema_200_position=0.126)
    assert hero["label"] == "未极端"
    assert hero["label"] != "震荡"
    assert hero["weekly"]["id"] == "above"
    assert hero["clocks_cross"]["id"] in {
        "slow_bull",
        "quiet_u",
        "quiet_still_dca",
        "unknown",
    }


def test_hero_fills_daily_chips_from_console_when_parquet_missing(monkeypatch):
    """CMS cockpit: no data/macro daily parquet, refresh off → Vision klines."""
    idx = pd.date_range("2025-01-01", periods=120, freq="D", tz="UTC")
    close = pd.Series([100.0] * 90 + [130.0] * 30, index=idx)
    daily = pd.DataFrame({"close": close})
    monkeypatch.setattr(
        "mlbot_console.services.macro_env.ensure_panel_daily",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "mlbot_console.services.macro_env._load_daily_frame",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "mlbot_console.services.macro_env._load_crypto_daily_from_console",
        lambda *a, **k: daily,
    )
    hero = build_btc_direction_hero(
        weekly_ema_200_position=0.138,
        allow_refresh=False,
    )
    assert hero["label"] == "牛"
    assert hero["close"] == 130.0
    assert hero["roc_90"] is not None and hero["roc_90"] > 0.15
    assert hero["vs_sma200"] is not None
    assert hero["source"] == "macro_daily"
    assert hero["sleeve"]["stack"] in {"coin_m_ok", "u_primary", "spot_dca"}


def test_weekly_sleeve_u_primary_is_below_ema50_not_death_cross():
    pullback = build_weekly_sleeve_snapshot(
        weekly_ema_200_position=0.04, weekly_ema_50_position=-0.03
    )
    assert pullback["stack"] == "u_primary"
    assert "EMA50" in pullback["hint"]
    death = build_weekly_sleeve_snapshot(
        weekly_ema_200_position=0.02, weekly_ema_50_position=0.08
    )
    assert death["stack"] == "u_primary"
    assert death["ema50_above_ema200"] is False


def test_weekly_sleeve_three_stacks():
    dca = build_weekly_sleeve_snapshot(
        weekly_ema_200_position=-0.04, weekly_ema_50_position=-0.02
    )
    assert dca["stack"] == "spot_dca"
    coin = build_weekly_sleeve_snapshot(
        weekly_ema_200_position=0.10, weekly_ema_50_position=0.04
    )
    assert coin["stack"] == "coin_m_ok"
    assert coin["ema50_above_ema200"] is True
    u_m = build_weekly_sleeve_snapshot(
        weekly_ema_200_position=0.02, weekly_ema_50_position=0.08
    )
    assert u_m["stack"] == "u_primary"


def test_hero_prefers_vision_seed_for_weekly_200(monkeypatch):
    idx = pd.date_range("2025-01-01", periods=120, freq="D", tz="UTC")
    close = pd.Series([100.0] * 90 + [130.0] * 30, index=idx)
    daily = pd.DataFrame({"close": close})
    monkeypatch.setattr(
        "mlbot_console.services.regime_hero._weekly_ema200_from_seed",
        lambda _d: {
            "id": "above",
            "label": "已离开深熊（周线站上 EMA200）",
            "position": 0.12,
            "source": "vision_seed",
        },
    )
    monkeypatch.setattr(
        "mlbot_console.services.macro_env.ensure_panel_daily",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "mlbot_console.services.macro_env._load_daily_frame",
        lambda *a, **k: daily,
    )
    hero = build_btc_direction_hero(allow_refresh=False)
    assert hero["weekly"]["position"] == pytest.approx(0.12)
    assert hero["weekly"]["source"] == "vision_seed"
    assert hero["sleeve"]["stack"] in {"coin_m_ok", "u_primary"}


def test_clocks_cross_split_cases():
    rebound = build_clocks_cross(
        roc_regime="bull",
        sleeve=build_weekly_sleeve_snapshot(
            weekly_ema_200_position=-0.04, weekly_ema_50_position=-0.02
        ),
    )
    assert rebound["id"] == "rebound_deep_bear"
    assert rebound["split"] is True
    assert "定投" in rebound["action"]

    slow = build_clocks_cross(
        roc_regime="neutral",
        sleeve=build_weekly_sleeve_snapshot(
            weekly_ema_200_position=0.10, weekly_ema_50_position=0.04
        ),
    )
    assert slow["id"] == "slow_bull"
    assert slow["split"] is True

    crash = build_clocks_cross(
        roc_regime="bear",
        sleeve=build_weekly_sleeve_snapshot(
            weekly_ema_200_position=0.02, weekly_ema_50_position=0.08
        ),
    )
    assert crash["id"] == "crash_above_200"
    assert crash["split"] is True
    assert "定投" in crash["action"]
    assert "仍在周线 200 上" in crash["why"]
    assert "跌破周线 50" in crash["why"]

    still = build_clocks_cross(
        roc_regime="bear",
        sleeve=build_weekly_sleeve_snapshot(
            weekly_ema_200_position=0.10, weekly_ema_50_position=0.04
        ),
    )
    assert still["id"] == "crash_still_stacked"
    assert still["split"] is True
    assert "金叉还在" in still["why"]
    assert still["why"] != crash["why"]

    aligned = build_clocks_cross(
        roc_regime="bull",
        sleeve=build_weekly_sleeve_snapshot(
            weekly_ema_200_position=0.10, weekly_ema_50_position=0.04
        ),
    )
    assert aligned["id"] == "aligned_bull"
    assert aligned["split"] is False
