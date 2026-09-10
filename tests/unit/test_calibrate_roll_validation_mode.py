"""calibrate_roll turbo configs are validation-only (M1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.pipeline.config import load_pipeline_config


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "rel",
    [
        "config/strategies/tpc/research/calibrate_roll.default.yaml",
        "config/strategies/bpc/research/calibrate_roll.default.yaml",
        "config/strategies/me/research/calibrate_roll.default.yaml",
    ],
)
def test_calibrate_roll_threshold_calibration_locked(rel: str) -> None:
    cfg = load_pipeline_config(_root() / rel)
    tc = cfg.get("threshold_calibration") or {}
    assert tc.get("enable_model_training") is False
    assert (tc.get("prefilter") or {}).get("optimize") is False
    assert (tc.get("gate") or {}).get("optimize") is False
    assert (tc.get("entry_filter") or {}).get("optimize") is False
    assert (tc.get("execution_opt") or {}).get("enabled") is False
    assert (tc.get("direction_tuning") or {}).get("enabled") is False


@pytest.mark.parametrize(
    "rel",
    [
        "config/strategies/tpc/research/research_roll.features_on.yaml",
        "config/strategies/bpc/research/research_roll.features_on.yaml",
    ],
)
def test_research_roll_shap_audit_only(rel: str) -> None:
    cfg = load_pipeline_config(_root() / rel)
    shap = cfg.get("shap_feature_selection") or {}
    assert shap.get("audit_only") is True
    assert shap.get("promote_to_features_yaml") is False
