"""Tests for live symbol resolution from universe.yaml."""

from __future__ import annotations

import argparse

import pytest

from src.live_data_stream.universe_symbols import (
    read_universe_bar_only_symbols,
    read_universe_symbols,
    resolve_symbols_csv,
)


def test_read_universe_highcap_includes_hype():
    syms = read_universe_symbols("highcap")
    assert "HYPEUSDT" in syms
    assert "ADAUSDT" not in syms
    # Bus publishes ETH; U-m SRB and coin B both trade it (2026-09-07).
    assert "ETHUSDT" in syms
    assert "PUMPUSDT" in syms
    assert "ASTERUSDT" in syms
    assert set(syms) == {
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "HYPEUSDT",
        "PUMPUSDT",
        "ASTERUSDT",
    }


def test_highcap_bar_only_symbols_are_spot_dca_assets():
    assert read_universe_bar_only_symbols("highcap") == [
        "ASTERUSDT",
        "PUMPUSDT",
    ]


def test_bc_coin_symbols_keep_eth_drop_spot_only_clocks():
    from src.live_data_stream.universe_symbols import resolve_bc_coin_symbols

    syms = resolve_bc_coin_symbols(system="b", universe="highcap")
    assert "ETHUSDT" in syms
    assert "PUMPUSDT" not in syms
    assert "ASTERUSDT" not in syms
    assert "BTCUSDT" in syms
    assert "BNBUSDT" in syms
    assert "SOLUSDT" in syms


def test_resolve_cli_overrides_universe():
    csv = resolve_symbols_csv(cli_symbols="DOGEUSDT", universe="highcap")
    assert csv == "DOGEUSDT"


def test_resolve_universe_when_no_cli():
    expected = ",".join(read_universe_symbols("highcap"))
    csv = resolve_symbols_csv(cli_symbols=None, universe="highcap", env_symbols="")
    assert csv == expected


def test_resolve_env_when_universe_missing(tmp_path, monkeypatch):
    csv = resolve_symbols_csv(
        cli_symbols=None,
        universe="missing",
        env_symbols="LINKUSDT",
        project_root=tmp_path,
    )
    assert csv == "LINKUSDT"


def test_publisher_resolve_symbols(monkeypatch):
    import scripts.run_market_feature_publisher as pub

    ns = argparse.Namespace(symbols=None, universe="highcap")
    syms, full, bar_only = pub._resolve_symbol_profiles(ns)
    assert "HYPEUSDT" in syms
    assert "PUMPUSDT" in syms
    assert bar_only == ["ASTERUSDT", "PUMPUSDT"]
    assert set(full) == set(syms) - set(bar_only)


def test_publisher_cli_symbols_do_not_require_universe_file(tmp_path, monkeypatch):
    import scripts.run_market_feature_publisher as pub

    monkeypatch.setattr(pub, "PROJECT_ROOT", tmp_path)
    ns = argparse.Namespace(symbols="DOGEUSDT", universe="missing")

    assert pub._resolve_symbol_profiles(ns) == (["DOGEUSDT"], ["DOGEUSDT"], [])


def test_feature_bus_manager_must_not_parse_str_none():
    """Regression: args.symbols=None must not become listener key 'NONE'."""
    ns = argparse.Namespace(symbols=None)
    bogus = [s.strip().upper() for s in str(ns.symbols).split(",") if s.strip()]
    assert bogus == ["NONE"]

    import scripts.run_market_feature_publisher as pub

    resolved = pub._resolve_symbols(
        argparse.Namespace(symbols=None, universe="highcap")
    )
    assert "BTCUSDT" in resolved
    assert "NONE" not in resolved


def test_build_feature_bus_manager_receives_only_full_feature_symbols(monkeypatch):
    import scripts.run_market_feature_publisher as pub
    from src.live_data_stream.feature_publisher_stack import build_feature_bus_manager

    captured: dict = {}

    class _FakeManager:
        def __init__(self, symbols, **kwargs):
            captured["symbols"] = list(symbols)
            self.listeners = {s: type("L", (), {})() for s in symbols}

    monkeypatch.setattr(
        "src.live_data_stream.feature_publisher_stack.MultiSymbolManager",
        _FakeManager,
    )
    monkeypatch.setattr(
        "src.live_data_stream.feature_publisher_stack._pick_primary_archetype",
        lambda *a, **k: "tpc",
    )
    monkeypatch.setattr(
        "src.live_data_stream.feature_publisher_stack._disk_package",
        lambda *a, **k: "tpc",
    )
    monkeypatch.setattr(
        "src.live_data_stream.feature_publisher_stack.resolve_strategy_package_under_root",
        lambda *a, **k: pub.PROJECT_ROOT / "config/strategies/tpc",
    )
    monkeypatch.setattr(
        "src.live_data_stream.feature_publisher_stack.resolve_constitution_yaml",
        lambda *a, **k: pub.PROJECT_ROOT / "config/constitution.yaml",
    )
    monkeypatch.setattr(
        "src.live_data_stream.feature_publisher_stack.load_constitution_dict",
        lambda *a, **k: {"resource_allocation": {"enabled_archetypes": ["tpc"]}},
    )
    monkeypatch.setattr(
        "src.live_data_stream.feature_publisher_stack._extract_plan",
        lambda *a, **k: (set(), []),
    )
    monkeypatch.setattr(
        "src.live_data_stream.feature_publisher_stack.IncrementalFeatureComputer",
        lambda *a, **k: type(
            "FC", (), {"live_feature_set": set(), "live_feature_nodes": []}
        )(),
    )

    ns = argparse.Namespace(
        symbols=None,
        universe="highcap",
        strategies_root=str(pub.PROJECT_ROOT / "config/strategies"),
        constitution_yaml=None,
        live_storage_base="data/live",
        memory_window_hours=24,
        feature_compute_interval_minutes=15,
        orderflow_window_minutes=60,
        feature_4h_interval_hours=4,
    )
    resolved, full_symbols, bar_only = pub._resolve_symbol_profiles(ns)
    writer = type("W", (), {"write": lambda *a, **k: None})()
    build_feature_bus_manager(ns, writer, full_symbols)
    assert captured["symbols"] == full_symbols
    assert set(resolved) == set(full_symbols) | set(bar_only)
    assert "BTCUSDT" in captured["symbols"]
    assert "PUMPUSDT" not in captured["symbols"]
    assert "ASTERUSDT" not in captured["symbols"]


def test_resolve_raises_when_all_missing(tmp_path):
    with pytest.raises(ValueError, match="no symbols"):
        resolve_symbols_csv(
            cli_symbols=None,
            universe="missing",
            env_symbols="",
            project_root=tmp_path,
        )
