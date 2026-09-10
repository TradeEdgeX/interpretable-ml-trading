"""Unit tests for plateau stability contract."""

from __future__ import annotations

import pytest

from scripts.plateau_stability import (
    PlateauRange,
    decide_plateau_update,
    plateau_range_from_dict,
    plateau_range_to_dict,
    ranges_overlap,
)


def test_plateau_range_invariants():
    PlateauRange(start=0.1, end=0.5, mid=0.3)
    with pytest.raises(ValueError):
        PlateauRange(start=0.5, end=0.1, mid=0.3)
    with pytest.raises(ValueError):
        PlateauRange(start=0.1, end=0.5, mid=0.9)


def test_ranges_overlap_basic():
    a = PlateauRange(0.1, 0.4, 0.25)
    b = PlateauRange(0.3, 0.6, 0.45)
    c = PlateauRange(0.5, 0.8, 0.65)
    assert ranges_overlap(a, b)
    assert ranges_overlap(b, c)
    assert not ranges_overlap(a, c)


def test_plateau_range_from_dict_handles_missing_mid():
    r = plateau_range_from_dict({"start": 0.2, "end": 0.5})
    assert r is not None
    assert r.mid == pytest.approx(0.35)


def test_plateau_range_roundtrip():
    r = PlateauRange(0.2, 0.5, 0.35)
    d = plateau_range_to_dict(r)
    assert plateau_range_from_dict(d) == r


def test_decide_adopt_when_no_prior():
    new = PlateauRange(0.30, 0.45, 0.375)
    out = decide_plateau_update(
        old=None,
        new=new,
        current_value=0.40,
    )
    assert out["action"] == "ADOPT"
    assert out["chosen_value"] == pytest.approx(0.375)
    assert out["overlap"] is True
    assert out["old_range"] is None


def test_decide_adopt_when_plateaus_overlap():
    old = PlateauRange(0.30, 0.45, 0.375)
    new = PlateauRange(0.40, 0.55, 0.475)
    out = decide_plateau_update(
        old=old,
        new=new,
        current_value=0.40,
    )
    assert out["action"] == "ADOPT"
    assert out["chosen_value"] == pytest.approx(0.475)
    assert out["overlap"] is True


def test_decide_alert_and_keep_when_plateaus_disjoint():
    old = PlateauRange(0.20, 0.30, 0.25)
    new = PlateauRange(0.45, 0.60, 0.525)
    out = decide_plateau_update(
        old=old,
        new=new,
        current_value=0.25,
    )
    assert out["action"] == "ALERT"
    assert out["chosen_value"] == pytest.approx(0.25)  # keep current
    assert out["overlap"] is False
    assert "plateau_drift_detected" in out["reason"]


def test_policy_adopt_anyway_bypasses_drift_lock():
    old = PlateauRange(0.20, 0.30, 0.25)
    new = PlateauRange(0.50, 0.60, 0.55)
    out = decide_plateau_update(
        old=old,
        new=new,
        current_value=0.25,
        policy="adopt_anyway",
    )
    assert out["action"] == "ADOPT"
    assert out["chosen_value"] == pytest.approx(0.55)
    assert out["overlap"] is False


def test_policy_unknown_raises():
    new = PlateauRange(0.50, 0.60, 0.55)
    old = PlateauRange(0.20, 0.30, 0.25)
    with pytest.raises(ValueError):
        decide_plateau_update(old=old, new=new, current_value=0.25, policy="anything")


def test_overlap_tolerance():
    a = PlateauRange(0.20, 0.30, 0.25)
    b = PlateauRange(0.305, 0.40, 0.35)
    assert not ranges_overlap(a, b)
    assert ranges_overlap(a, b, tol=0.01)
