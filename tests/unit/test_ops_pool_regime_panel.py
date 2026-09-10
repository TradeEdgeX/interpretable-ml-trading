"""Ops pool regime stale panel (bull market vs active_regime=bear)."""

from __future__ import annotations

from pathlib import Path

from mlbot_console.services.rebalance_advisor import build_ops_pool_regime_panel

ROOT = Path(__file__).resolve().parents[2]
LIVE_ROOT = ROOT / "live" / "highcap"

_BEAR_GUARD = """
resource_allocation:
  slot_policy:
    trend_pool_guard:
      active_regime: bear
"""


def _bear_live_root(tmp_path: Path) -> Path:
    cons = tmp_path / "config" / "constitution"
    cons.mkdir(parents=True)
    cons.joinpath("constitution.yaml").write_text(_BEAR_GUARD, encoding="utf-8")
    cons.joinpath("constitution_coin.yaml").write_text(_BEAR_GUARD, encoding="utf-8")
    return tmp_path


def test_ops_pool_regime_stale_when_bull_and_bear_lock(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("MLBOT_TREND_POOL_ACTIVE_REGIME", raising=False)
    panel = build_ops_pool_regime_panel(
        live_root=_bear_live_root(tmp_path),
        composite={"label": "risk_on"},
        layers={"b_trend": {"current_label": "bull"}},
    )
    assert panel["market_bull"] is True
    assert panel["stale"] is True
    assert panel["alert"] == "REBALANCE_SUGGEST"
    assert "ta" in panel["stale_sleeves"]
    assert "coin" in panel["stale_sleeves"]
    assert "active_regime" in panel["suggestion"]


def test_ops_pool_regime_ok_when_live_yaml_bull(monkeypatch) -> None:
    monkeypatch.delenv("MLBOT_TREND_POOL_ACTIVE_REGIME", raising=False)
    panel = build_ops_pool_regime_panel(
        live_root=LIVE_ROOT,
        composite={"label": "risk_on"},
        layers={"b_trend": {"current_label": "bull"}},
    )
    assert panel["stale"] is False
    assert panel["alert"] == "OK"
    assert panel["ta_active_regime"] == "bull"
    assert panel["coin_active_regime"] == "bull"


def test_ops_pool_regime_ok_when_env_bull(monkeypatch) -> None:
    monkeypatch.setenv("MLBOT_TREND_POOL_ACTIVE_REGIME", "bull")
    panel = build_ops_pool_regime_panel(
        live_root=LIVE_ROOT,
        composite={"label": "risk_on"},
        layers={"b_trend": {"current_label": "bull"}},
    )
    assert panel["stale"] is False
    assert panel["alert"] == "OK"
    assert panel["ta_active_regime"] == "bull"


def test_ops_pool_regime_ok_when_not_bull(monkeypatch) -> None:
    monkeypatch.delenv("MLBOT_TREND_POOL_ACTIVE_REGIME", raising=False)
    panel = build_ops_pool_regime_panel(
        live_root=LIVE_ROOT,
        composite={"label": "risk_off"},
        layers={"b_trend": {"current_label": "bear"}},
    )
    assert panel["market_bull"] is False
    assert panel["stale"] is False
