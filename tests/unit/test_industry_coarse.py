"""Tests for industry coarse mapping."""

from __future__ import annotations

from ashare.industry_coarse import is_coarse, to_coarse
from ashare.sync_pool_industry import backfill_pool_industry


def test_to_coarse_exact_and_csrc():
    assert to_coarse("医药生物") == "医药生物"
    assert to_coarse("C27医药制造业") == "医药生物"
    assert to_coarse("C39计算机、通信和其他电子设备制造业") == "科技"
    assert to_coarse("") == "综合"
    assert is_coarse("科技")
    assert not is_coarse("C27医药制造业")


def test_backfill_preserves_coarse_converts_fine():
    pool = [
        {"code": "600276", "name": "恒瑞", "industry": "医药生物"},
        {"code": "688235", "name": "百济", "industry": "C27医药制造业"},
        {"code": "001267", "name": "汇绿", "industry": ""},
    ]
    mp = {"688235": "C27医药制造业"}
    out, n = backfill_pool_industry(pool, mp, overwrite=False, apply_coarse=True)
    assert out[0]["industry"] == "医药生物"
    assert out[1]["industry"] == "医药生物"
    assert out[2]["industry"] == "综合"
    assert n >= 2
