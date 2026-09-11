"""Sector-rank weights and 10bp cost (no network)."""

from __future__ import annotations

import pandas as pd

from src.research.cs_sector import (
    cost_from_turnover,
    load_industry,
    one_way_turnover,
    week_end_dates,
)


def test_one_way_turnover_half_rotate():
    prev = pd.Series({"a": 0.5, "b": 0.5})
    new = pd.Series({"a": 1.0})
    assert abs(one_way_turnover(prev, new) - 0.5) < 1e-12


def test_load_industry_yaml(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("'000001': 金融板块\n600519: 食品饮料\n", encoding="utf-8")
    ser = load_industry(p)
    assert ser["000001"] == "金融板块"
    assert ser["600519"] == "食品饮料"


def test_ten_bp_on_full_deploy():
    assert abs(cost_from_turnover(1.0, bp=10) - 0.001) < 1e-12
    assert abs(cost_from_turnover(0.5, bp=10) - 0.0005) < 1e-12


def test_week_end_is_last_session():
    days = pd.to_datetime(["2024-09-23", "2024-09-24", "2024-09-25", "2024-09-26", "2024-09-27"])
    ends = week_end_dates(days)
    assert pd.Timestamp("2024-09-27") in ends
    assert pd.Timestamp("2024-09-23") not in ends
