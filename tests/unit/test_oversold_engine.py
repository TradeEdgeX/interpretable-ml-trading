"""Unit tests for YAML-driven oversold engine."""

from __future__ import annotations

import pytest

from auxiliary.oversold_engine import (
    apply_gates,
    classify_strong_entry,
    classify_tier,
    compute_oversold_score,
    load_oversold_archetype,
    pick_top_by_entry_level,
    pick_top_strong_oversold,
    rank_key,
)


@pytest.fixture(autouse=True)
def _clear_archetype_cache():
    load_oversold_archetype.cache_clear()
    yield
    load_oversold_archetype.cache_clear()


def test_classify_rsi20_amp7_primary_l1():
    out = classify_strong_entry(
        {"rsi": 18.0, "amplitude": 8.0, "dist_ma200_pct": -10.0},
        strategy="ashare_oversold",
    )
    assert out["strong_tier"] == "rsi20_amp7"
    assert out["tier_action"] == "primary"
    assert out["entry_level"] == "L1"
    assert out["display_label"].startswith("L1")


def test_classify_rsi20_amp7_with_deep_ma200_is_l2_size_up():
    out = classify_strong_entry(
        {"rsi": 15.0, "amplitude": 8.0, "dist_ma200_pct": -50.0},
        strategy="ashare_oversold",
    )
    assert out["strong_tier"] == "rsi20_amp7"
    assert out["entry_level"] == "L2"
    assert out["tier_action"] == "size_up"
    assert "L2" in out["display_label"]


def test_classify_rsi20_amp5_demoted_to_l3_watch():
    """5% < amp ≤ 7% no longer L1 after 20260729 retune."""
    out = classify_strong_entry(
        {"rsi": 18.0, "amplitude": 6.0, "dist_ma200_pct": -10.0},
        strategy="ashare_oversold",
    )
    assert out["strong_tier"] == "rsi20_amp5"
    assert out["tier_action"] == "watch"
    assert out["entry_level"] == "L3"


def test_classify_rsi25_amp4_watch():
    out = classify_strong_entry(
        {"rsi": 22.0, "amplitude": 4.5, "dist_ma200_pct": -10.0},
        strategy="ashare_oversold",
    )
    assert out["strong_tier"] == "rsi25_amp4"
    assert out["tier_action"] == "watch"
    assert out["entry_level"] == "L3"


def test_classify_ma200_40_when_rsi_tiers_fail():
    out = classify_strong_entry(
        {"rsi": 22.0, "amplitude": 2.0, "dist_ma200_pct": -45.0},
        strategy="ashare_oversold",
    )
    assert out["strong_tier"] == "ma200_40"
    assert out["entry_level"] == "L3"


def test_rsi20_amp7_beats_ma200_when_both_match():
    tier = classify_tier(
        {"rsi": 15.0, "amplitude": 8.0, "dist_ma200_pct": -50.0},
        strategy="ashare_oversold",
    )
    assert tier == "rsi20_amp7"


def test_rank_key_prefers_stronger_tier():
    strong = {
        "strong_tier": "rsi20_amp7",
        "rsi": 18,
        "amplitude": 8,
        "dist_ma200_pct": -5,
    }
    weak = {"strong_tier": "ma200_40", "rsi": 30, "amplitude": 8, "dist_ma200_pct": -50}
    assert rank_key(strong, strategy="ashare_oversold") < rank_key(
        weak, strategy="ashare_oversold"
    )


def test_pick_top_by_entry_level_caps_each_bucket():
    items = []
    for i in range(5):
        items.append(
            {
                "code": f"l1_{i}",
                "strong_tier": "rsi20_amp7",
                "entry_level": "L1",
                "rsi": 10 + i,
                "amplitude": 9,
                "dist_ma200_pct": -5,
            }
        )
    for i in range(4):
        items.append(
            {
                "code": f"l3_{i}",
                "strong_tier": "rsi20_amp5",
                "entry_level": "L3",
                "rsi": 18,
                "amplitude": 6,
                "dist_ma200_pct": -5,
            }
        )
    top = pick_top_by_entry_level(items, strategy="ashare_oversold", n_per_level=3)
    assert len([x for x in top if x["entry_level"] == "L1"]) == 3
    assert len([x for x in top if x["entry_level"] == "L3"]) == 3
    assert top[0]["entry_level"] == "L1"


def test_pick_top_respects_top_n():
    arch = load_oversold_archetype("ashare_oversold")
    items = [
        {
            "code": "a",
            "strong_tier": "rsi20_amp7",
            "rsi": 10,
            "amplitude": 8,
            "dist_ma200_pct": 0,
        },
        {
            "code": "b",
            "strong_tier": "rsi25_amp4",
            "rsi": 20,
            "amplitude": 5,
            "dist_ma200_pct": 0,
        },
        {
            "code": "c",
            "strong_tier": "ma200_40",
            "rsi": 30,
            "amplitude": 3,
            "dist_ma200_pct": -50,
        },
        {
            "code": "d",
            "strong_tier": "rsi25_amp4",
            "rsi": 24,
            "amplitude": 4.5,
            "dist_ma200_pct": 0,
        },
        {
            "code": "e",
            "strong_tier": "rsi25_amp4",
            "rsi": 23,
            "amplitude": 4.2,
            "dist_ma200_pct": 0,
        },
        {
            "code": "f",
            "strong_tier": "ma200_40",
            "rsi": 35,
            "amplitude": 2,
            "dist_ma200_pct": -45,
        },
        {
            "code": "g",
            "strong_tier": "ma200_40",
            "rsi": 40,
            "amplitude": 2,
            "dist_ma200_pct": -41,
        },
    ]
    top = pick_top_strong_oversold(items, strategy="ashare_oversold")
    assert len(top) == min(len(items), arch.top_n)

    padded = items + [
        {
            "code": f"x{i}",
            "strong_tier": "ma200_40",
            "rsi": 45 + i,
            "amplitude": 2,
            "dist_ma200_pct": -42,
        }
        for i in range(arch.top_n)
    ]
    top_full = pick_top_strong_oversold(padded, strategy="ashare_oversold")
    assert len(top_full) == arch.top_n


def test_core_pool_l1_requires_rsi18_amp8():
    """core554 retuned 20260802 — RSI20+amp7 is only a watch tier there."""
    out = classify_strong_entry(
        {"rsi": 19.0, "amplitude": 7.5, "dist_ma200_pct": -10.0},
        strategy="ashare_oversold",
        pool="core",
    )
    assert out["strong_tier"] == "rsi20_amp7"
    assert out["entry_level"] == "L3"
    assert out["tier_action"] == "watch"

    out = classify_strong_entry(
        {"rsi": 17.0, "amplitude": 8.5, "dist_ma200_pct": -10.0},
        strategy="ashare_oversold",
        pool="core",
    )
    assert out["strong_tier"] == "rsi18_amp8"
    assert out["entry_level"] == "L1"
    assert out["tier_action"] == "primary"


def test_full_pool_keeps_rsi20_amp7_as_l1():
    """Same bar that is only a watch in core554 stays L1 on the 2000 pool."""
    for pool in ("full", ""):
        out = classify_strong_entry(
            {"rsi": 19.0, "amplitude": 7.5, "dist_ma200_pct": -10.0},
            strategy="ashare_oversold",
            pool=pool,
        )
        assert out["strong_tier"] == "rsi20_amp7", pool
        assert out["entry_level"] == "L1", pool
    assert "rsi18_amp8" not in {
        t.id for t in load_oversold_archetype("ashare_oversold", "full").tiers
    }


def test_core_pool_l2_size_up_uses_its_own_primary_tier():
    out = classify_strong_entry(
        {"rsi": 17.0, "amplitude": 8.5, "dist_ma200_pct": -50.0},
        strategy="ashare_oversold",
        pool="core",
    )
    assert out["strong_tier"] == "rsi18_amp8"
    assert out["entry_level"] == "L2"
    assert out["tier_action"] == "size_up"
    assert "RSI<18+amp>8%" in out["display_label"]


def test_core_pool_ranks_its_primary_tier_first():
    core_l1 = {
        "strong_tier": "rsi18_amp8",
        "rsi": 17,
        "amplitude": 8.5,
        "dist_ma200_pct": -5,
    }
    demoted = {
        "strong_tier": "rsi20_amp7",
        "rsi": 12,
        "amplitude": 9.9,
        "dist_ma200_pct": -5,
    }
    assert rank_key(core_l1, strategy="ashare_oversold", pool="core") < rank_key(
        demoted, strategy="ashare_oversold", pool="core"
    )


def test_hk_unaffected_by_ashare_pool_overrides():
    """HK has no pool_tiers block — any pool arg must resolve to its shared stack."""
    base = load_oversold_archetype("hk_oversold")
    for pool in ("core", "full"):
        arch = load_oversold_archetype("hk_oversold", pool)
        assert [t.id for t in arch.tiers] == [t.id for t in base.tiers]
        assert arch.pool == ""


def test_hk_gates_reject_missing_pe():
    ok, notes = apply_gates({"gamma_ok": 1}, strategy="hk_oversold")
    assert not ok
    assert "PE缺失" in notes[0]


def test_hk_gates_reject_negative_pe():
    ok, notes = apply_gates({"pe": -1.0, "gamma_ok": 1}, strategy="hk_oversold")
    assert not ok
    assert "PE亏损" in notes[0]


def test_hk_gates_pass_with_pe_and_gamma():
    ok, notes = apply_gates({"pe": 12.5, "gamma_ok": 1}, strategy="hk_oversold")
    assert ok
    assert any("Gamma达标" in n for n in notes)


def test_compute_oversold_score_higher_for_stronger_tier():
    a = compute_oversold_score(
        {"strong_tier": "rsi20_amp7", "rsi": 15, "amplitude": 8, "dist_ma200_pct": -5},
        strategy="ashare_oversold",
    )
    b = compute_oversold_score(
        {"strong_tier": "ma200_40", "rsi": 30, "amplitude": 3, "dist_ma200_pct": -45},
        strategy="ashare_oversold",
    )
    assert a > b
