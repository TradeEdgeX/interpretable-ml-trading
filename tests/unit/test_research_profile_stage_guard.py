"""Guards: research turbo/slow default to rolling_sim; forbid implicit full on CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.rolling_dashboard.pipeline_jobs import validate_payload
from src.config.strategy_layout import is_research_turbo_or_slow_yaml

_ROOT = Path(__file__).resolve().parents[2]


def test_is_research_turbo_or_slow_yaml() -> None:
    assert is_research_turbo_or_slow_yaml(
        _ROOT / "config/strategies/bpc/research/calibrate_roll.default.yaml"
    )
    assert is_research_turbo_or_slow_yaml(
        _ROOT / "config/strategies/bpc/research/research_roll.features_on.yaml"
    )
    assert not is_research_turbo_or_slow_yaml(
        _ROOT / "config/strategies/bpc/research/validate_static.full_study.yaml"
    )


def test_validate_payload_turbo_defaults_to_rolling_sim() -> None:
    norm, err = validate_payload(
        {
            "strategy": "bpc",
            "config_path": "config/strategies/bpc/research/calibrate_roll.default.yaml",
        }
    )
    assert err is None
    assert norm is not None
    assert norm["stage"] == "rolling_sim"


def test_validate_payload_non_rolling_stays_full() -> None:
    norm, err = validate_payload(
        {
            "strategy": "bpc",
            "config_path": "config/strategies/bpc/research/validate_static.full_study.yaml",
        }
    )
    assert err is None
    assert norm is not None
    assert norm["stage"] is None


def test_validate_payload_explicit_stage_preserved() -> None:
    norm, err = validate_payload(
        {
            "strategy": "bpc",
            "config_path": "config/strategies/bpc/research/calibrate_roll.default.yaml",
            "stage": "prefilter",
        }
    )
    assert err is None
    assert norm is not None
    assert norm["stage"] == "prefilter"


def test_cli_rejects_full_on_turbo_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    import scripts.auto_research_pipeline as arp

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "auto_research_pipeline.py",
            "--strategy",
            "bpc",
            "--config",
            str(_ROOT / "config/strategies/bpc/research/calibrate_roll.default.yaml"),
            "--dry-run",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        arp.main()
    assert exc.value.code != 0
