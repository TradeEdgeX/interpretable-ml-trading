"""Stub: multileg runtime features are not used by ma_cross FeatureStore path."""

from __future__ import annotations

from typing import Any, Mapping, Optional


def enrich_multileg_runtime_features(
    features: dict, *, wanted: Optional[Any] = None
) -> dict:
    return features


def live_feature_satisfied(key: str, features: Mapping[str, Any]) -> bool:
    return key in features
