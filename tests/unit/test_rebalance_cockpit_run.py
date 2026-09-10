"""T2d rebalance cockpit scheduled check."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.monitoring.rebalance_cockpit_run import (
    _compact_detail,
    _exit_code,
    _monitor_status,
    format_rebalance_telegram_message,
)


def test_monitor_status_and_exit_code():
    assert _monitor_status("OK") == "OK"
    assert _monitor_status("WATCH") == "ALERT"
    assert _exit_code("OK") == 0
    assert _exit_code("WATCH") == 1
    assert _exit_code("REBALANCE_SUGGEST") == 2


def test_format_rebalance_telegram_message():
    payload = {
        "symbol": "BTCUSDT",
        "composite": {"label_title": "risk-on"},
        "feature_bus": {"stale": False},
        "action_plan": {
            "alert": "WATCH",
            "capital": {"alert": "WATCH", "suggestions": ["考虑增加 beta"]},
            "execution": {"alert": "OK", "suggestions": []},
            "suggestions": ["[Capital] 考虑增加 beta"],
        },
    }
    msg = format_rebalance_telegram_message(
        payload=payload, alert="WATCH", run_ts="20260612_120000"
    )
    assert "WATCH" in msg
    assert "BTCUSDT" in msg
    assert "考虑增加 beta" in msg


def test_notify_ops_pool_regime_stale_if_needed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.monitoring.rebalance_cockpit_run import (
        notify_ops_pool_regime_stale_if_needed,
    )

    sent: list[str] = []
    monkeypatch.setattr(
        "src.monitoring.rebalance_cockpit_run.send_telegram_message",
        lambda msg, **kw: sent.append(msg) or True,
    )
    assert (
        notify_ops_pool_regime_stale_if_needed(
            payload={"ops_pool_regime": {"stale": False}},
        )
        is False
    )
    assert not sent
    assert (
        notify_ops_pool_regime_stale_if_needed(
            payload={
                "ops_pool_regime": {
                    "stale": True,
                    "stale_sleeves": ["ta"],
                    "composite_label": "risk_on",
                    "b_trend_label": "bull",
                    "ta_active_regime": "bear",
                    "suggestion": "flip to bull",
                }
            },
        )
        is True
    )
    assert sent and "Ops pool regime stale" in sent[0]


def test_notify_champion_proximity_if_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.monitoring.rebalance_cockpit_run import notify_champion_proximity_if_needed

    sent: list[str] = []

    monkeypatch.setattr(
        "src.monitoring.telegram.send_telegram_message",
        lambda msg, **kw: sent.append(msg) or True,
    )
    payload = {
        "btc_rolling_signal": {
            "available": True,
            "symbol": "BTCUSDT",
            "bar_ts": "2026-07-06",
            "close": 100000,
            "entry_ready": False,
            "conditions": [
                {"id": "deep_bear", "pass": True, "label": "深熊"},
                {"id": "ema_cross", "pass": True, "label": "EMA金叉"},
                {"id": "momentum", "pass": False, "label": "动量"},
            ],
            "proximity": {
                "level": "imminent",
                "label": "即将触发",
                "notify": True,
                "conditions_met": 2,
                "conditions_total": 3,
                "hint": "差 1 项",
            },
        }
    }
    assert notify_champion_proximity_if_needed(payload=payload) is True
    assert sent


def test_rebalance_check_exits_zero_on_watch(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.monitoring.rebalance_cockpit_run import run_rebalance_cockpit_check

    monkeypatch.setattr(
        "src.monitoring.rebalance_cockpit_run.build_regime_cockpit",
        lambda **kw: {
            "symbol": "BTCUSDT",
            "composite": {"label": "neutral"},
            "feature_bus": {"stale": False},
            "allocation": {"alert": "WATCH", "scopes": [], "suggestions": []},
        },
    )
    summary = run_rebalance_cockpit_check(dry_run=True)
    assert summary["exit_code"] == 1

    import scripts.monitoring.rebalance_cockpit_check as chk

    monkeypatch.setattr(chk, "run_rebalance_cockpit_check", lambda **kw: summary)
    monkeypatch.setattr(chk.sys, "argv", ["rebalance_cockpit_check.py"])
    assert chk.main() == 0


def test_compact_detail():
    payload = {
        "symbol": "BTCUSDT",
        "as_of": "2026-06-12T00:00:00Z",
        "composite": {"label": "neutral"},
        "feature_bus": {"stale": False},
        "allocation": {
            "alert": "WATCH",
            "total_nav_usdt": 1000,
            "scopes": [
                {
                    "scope": "spot",
                    "nav_pct": 0.2,
                    "status": "OK",
                    "band": {"target": 0.25},
                }
            ],
            "suggestions": ["ok"],
        },
    }
    d = _compact_detail(payload)
    assert d["alert"] == "WATCH"
    assert d["scopes"][0]["scope"] == "spot"


def test_composite_rules_cover_risk_on_boundary():
    """Composite follows 大环境 ROC90 only (same badge as the map)."""
    from mlbot_console.services.rebalance_advisor import (
        compute_composite,
        load_rebalance_config,
    )

    config = load_rebalance_config(Path(__file__).resolve().parents[2])
    assert config, "rebalance_targets.yaml not found"

    result = compute_composite({"roc_90": 0.20}, config)
    assert result["label"] == "risk_on", f"total={result['total_score']}"
    assert result["total_score"] == 2

    result_mid = compute_composite({"roc_90": 0.02}, config)
    assert result_mid["label"] == "neutral", f"total={result_mid['total_score']}"

    result_off = compute_composite({"roc_90": -0.20}, config)
    assert result_off["label"] == "risk_off", f"total={result_off['total_score']}"

    missing = compute_composite({}, config)
    assert missing["label"] == "neutral"
    assert missing["total_score"] is None


def test_map_covers_max_possible_total():
    """Last map cap must cover ROC90 max (2)."""
    from mlbot_console.services.rebalance_advisor import load_rebalance_config

    config = load_rebalance_config(Path(__file__).resolve().parents[2])
    map_rules = (config.get("composite") or {}).get("map") or []
    assert map_rules, "map is empty"
    last_cap = int(map_rules[-1].get("max_total") or 0)
    assert last_cap >= 2, f"last map max_total={last_cap} must cover max of 2"
