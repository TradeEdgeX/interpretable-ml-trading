"""Unit tests for research trail_overlay (default-off)."""

from __future__ import annotations

from src.time_series_model.live.trail_overlay import resolve_trail_overlay


def _pos(overlay: dict) -> dict:
    return {
        "trail_overlay": overlay,
        "_trail_overlay_pause_left": 0,
    }


def test_disabled_is_noop():
    pos = _pos(
        {
            "enabled": False,
            "modes": {
                "widen": {
                    "feature": "me_flow_exhaustion",
                    "min": 0.2,
                    "trail_r_mult": 2.0,
                }
            },
        }
    )
    r, pause, bank, tag = resolve_trail_overlay(
        pos,
        trail_r=6.0,
        is_long=False,
        feats={"me_flow_exhaustion": 0.9},
        trailing_activated=True,
    )
    assert r == 6.0 and not pause and not bank and tag == ""


def test_widen_short_on_exhaustion():
    pos = _pos(
        {
            "enabled": True,
            "modes": {
                "widen": {
                    "enabled": True,
                    "feature": "me_flow_exhaustion",
                    "min": 0.25,
                    "trail_r_mult": 1.75,
                    "side": "short",
                }
            },
        }
    )
    r, pause, bank, tag = resolve_trail_overlay(
        pos,
        trail_r=6.0,
        is_long=False,
        feats={"me_flow_exhaustion": 0.4},
        trailing_activated=True,
    )
    assert abs(r - 10.5) < 1e-9 and not pause and not bank and "widen" in tag


def test_bank_signed_divergence():
    pos = _pos(
        {
            "enabled": True,
            "modes": {
                "bank": {
                    "enabled": True,
                    "feature": "cvd_divergence_score",
                    "compare": "abs_min",
                    "abs_min": 0.25,
                }
            },
        }
    )
    _, _, bank_s, _ = resolve_trail_overlay(
        pos,
        trail_r=6.0,
        is_long=False,
        feats={"cvd_divergence_score": 0.4},
        trailing_activated=True,
    )
    _, _, bank_l, _ = resolve_trail_overlay(
        pos,
        trail_r=6.0,
        is_long=True,
        feats={"cvd_divergence_score": -0.4},
        trailing_activated=True,
    )
    _, _, bank_no, _ = resolve_trail_overlay(
        pos,
        trail_r=6.0,
        is_long=False,
        feats={"cvd_divergence_score": -0.4},
        trailing_activated=True,
    )
    assert bank_s and bank_l and not bank_no


def test_rr_constraints_passes_trail_overlay():
    from src.time_series_model.live.execution_profile_apply import (
        rr_constraints_from_exec_params,
    )

    ov = {"enabled": True, "modes": {"widen": {"feature": "x", "min": 0.1}}}
    rr = rr_constraints_from_exec_params({"trail_overlay": ov, "initial_r": 4.0})
    assert rr["trail_overlay"]["enabled"] is True


def test_pause_counts_down():
    pos = _pos(
        {
            "enabled": True,
            "modes": {
                "pause": {
                    "enabled": True,
                    "feature": "me_flow_exhaustion",
                    "min": 0.3,
                    "bars": 3,
                }
            },
        }
    )
    _, p1, _, _ = resolve_trail_overlay(
        pos,
        trail_r=6.0,
        is_long=False,
        feats={"me_flow_exhaustion": 0.5},
        trailing_activated=True,
    )
    assert p1 and pos["_trail_overlay_pause_left"] == 2
    _, p2, _, _ = resolve_trail_overlay(
        pos,
        trail_r=6.0,
        is_long=False,
        feats={"me_flow_exhaustion": 0.0},
        trailing_activated=True,
    )
    assert p2 and pos["_trail_overlay_pause_left"] == 1
