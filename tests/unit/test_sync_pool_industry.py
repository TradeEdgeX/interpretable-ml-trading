"""Tests for pool industry sync."""

from __future__ import annotations

from ashare.sync_pool_industry import (
    apply_industry_overrides,
    apply_industry_overrides_to_pool,
    backfill_pool_industry,
)


def test_backfill_preserves_existing_industry():
    pool = [
        {"code": "600276", "name": "恒瑞", "industry": "医药生物"},
        {"code": "688235", "name": "百济", "industry": ""},
    ]
    mp = {"688235": "C27医药制造业", "600276": "化学制药"}
    out, n = backfill_pool_industry(pool, mp, overwrite=False, apply_coarse=True)
    assert out[0]["industry"] == "医药生物"
    assert out[1]["industry"] == "医药生物"
    assert n == 1


def test_industry_overrides_win_over_wrong_auto_bucket():
    mp = {"002008": "汽车", "002747": "汽车", "600104": "汽车"}
    ov = {"002008": "电力设备", "002747": "电力设备"}
    merged, n = apply_industry_overrides(mp, ov)
    assert n == 2
    assert merged["002008"] == "电力设备"
    assert merged["002747"] == "电力设备"
    assert merged["600104"] == "汽车"

    pool = [
        {"code": "002008", "name": "大族激光", "industry": "汽车"},
        {"code": "600104", "name": "上汽集团", "industry": "汽车"},
    ]
    out, n2 = apply_industry_overrides_to_pool(pool, ov)
    assert n2 == 1
    assert out[0]["industry"] == "电力设备"
    assert out[1]["industry"] == "汽车"
