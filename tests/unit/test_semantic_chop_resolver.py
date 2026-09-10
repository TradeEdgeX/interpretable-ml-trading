"""Tests for canonical semantic_chop resolution."""

from __future__ import annotations

import math

import pytest

from src.features.semantic_chop import (
    as_finite_float,
    normalize_semantic_chop_aliases,
    resolve_semantic_chop,
    resolve_feature_float,
)


def test_as_finite_float_rejects_nan_and_inf() -> None:
    assert as_finite_float(float("nan")) is None
    assert as_finite_float(float("inf")) is None
    assert as_finite_float(0.42) == pytest.approx(0.42)


def test_resolve_semantic_chop_prefers_canonical() -> None:
    features = {
        "semantic_chop": 0.3,
        "tpc_semantic_chop": 0.9,
        "bpc_semantic_chop": 0.1,
    }
    assert resolve_semantic_chop(features) == pytest.approx(0.3)


def test_resolve_semantic_chop_falls_back_to_tpc() -> None:
    features = {"tpc_semantic_chop": 0.25, "bpc_semantic_chop": float("nan")}
    assert resolve_semantic_chop(features) == pytest.approx(0.25)


def test_resolve_semantic_chop_skips_nan_bpc_for_tpc() -> None:
    features = {"bpc_semantic_chop": float("nan"), "tpc_semantic_chop": 0.18}
    assert resolve_semantic_chop(features) == pytest.approx(0.18)


def test_normalize_semantic_chop_aliases_backfills_aliases() -> None:
    feat = {"tpc_semantic_chop": 0.55, "bpc_semantic_chop": float("nan")}
    normalize_semantic_chop_aliases(feat)
    assert feat["semantic_chop"] == pytest.approx(0.55)
    assert feat["bpc_semantic_chop"] == pytest.approx(0.55)
    assert feat["tpc_semantic_chop"] == pytest.approx(0.55)


def test_resolve_feature_float_ordered_keys() -> None:
    row = {"trend_confidence": float("nan"), "trend_confidence_f": 0.8}
    assert resolve_feature_float(
        row, ["trend_confidence", "trend_confidence_f"]
    ) == pytest.approx(0.8)


def test_normalize_noop_when_all_missing() -> None:
    feat: dict = {"close": 100.0}
    normalize_semantic_chop_aliases(feat)
    assert "semantic_chop" not in feat
    assert not any(math.isnan(v) for v in feat.values() if isinstance(v, float))
