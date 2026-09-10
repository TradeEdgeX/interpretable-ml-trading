"""CMS settings defaults."""

from __future__ import annotations

from pathlib import Path

from mlbot_console.config import ConsoleSettings


def test_strategies_root_defaults_to_live_highcap(monkeypatch):
    monkeypatch.delenv("MLBOT_CONSOLE_STRATEGIES_ROOT", raising=False)
    monkeypatch.delenv("MLBOT_CONSOLE_LIVE_ROOT", raising=False)
    s = ConsoleSettings.from_env()
    assert s.strategies_root == s.live_root / "config" / "strategies"
    assert "live" in str(s.strategies_root)
    assert (s.repo_root / "config" / "strategies") != s.strategies_root


def test_strategies_root_env_override(monkeypatch, tmp_path: Path):
    research = tmp_path / "config" / "strategies"
    research.mkdir(parents=True)
    monkeypatch.setenv("MLBOT_CONSOLE_STRATEGIES_ROOT", str(research))
    s = ConsoleSettings.from_env()
    assert s.strategies_root.resolve() == research.resolve()
