"""Closed-bar 2-factor cross-section helpers (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.cs_panel import (
    add_score,
    amount_z_20,
    attach_factors,
    daily_books,
    mom_20,
    rank_ic_day,
    score_weighted_ret,
)


def _tape(n: int = 50, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n)
    close = 10.0 + np.cumsum(rng.normal(0, 0.1, size=n))
    amount = 1e8 + rng.normal(0, 1e6, size=n)
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open": open_, "close": close, "amount": np.abs(amount)}, index=idx)


def test_mom_and_amount_z_need_20_bars():
    close = pd.Series(np.linspace(10, 12, 25))
    mom = mom_20(close)
    assert mom.iloc[:19].isna().all()
    assert np.isfinite(mom.iloc[20])
    amt = amount_z_20(pd.Series(np.linspace(1.0, 2.0, 25)))
    assert amt.iloc[:19].isna().all()


def test_book_ret_skips_signal_bar():
    tab = _tape(40)
    attached = attach_factors(tab)
    # Signal on row i uses next open → open two bars later.
    i = 20
    expect = tab["open"].iloc[i + 2] / tab["open"].iloc[i + 1] - 1.0
    assert abs(attached["book_ret"].iloc[i] - expect) < 1e-12


def test_score_is_locked_average_of_cs_z():
    idx = pd.DatetimeIndex(["2020-01-02"] * 4 + ["2020-01-03"] * 4)
    long = pd.DataFrame(
        {
            "mom_20": [1.0, 0.0, -1.0, 0.5, 1.0, 0.0, -1.0, 0.5],
            "amount_z_20": [1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            "book_ret": 0.01,
            "fwd_20": [0.2, 0.0, -0.2, 0.1, 0.2, 0.0, -0.2, 0.1],
            "symbol": list("abcd") * 2,
        },
        index=idx,
    )
    scored = add_score(long)
    day = scored.loc["2020-01-02"]
    assert abs(day["score"].iloc[0] - 0.5 * day["cs_mom"].iloc[0] - 0.5 * day["cs_amt"].iloc[0]) < 1e-12


def test_rank_ic_positive_when_aligned():
    x = pd.Series([1.0, 2.0, 3.0, 4.0] * 20)
    y = x * 0.5
    assert rank_ic_day(x, y, min_n=50) > 0.9


def test_daily_books_longs_the_top():
    days = pd.bdate_range("2020-01-02", periods=2)
    rows = []
    for day in days:
        for i, sym in enumerate(list("abcdefghij") * 6):  # 60 names
            rows.append(
                {
                    "date": day,
                    "symbol": f"{sym}{i}",
                    "score": float(i),
                    "book_ret": 0.10 if i >= 48 else -0.01,
                    "mom_20": 0.0,
                    "amount_z_20": 0.0,
                    "fwd_20": 0.0,
                }
            )
    long = pd.DataFrame(rows).set_index("date")
    books = daily_books(long, top_q=0.80)
    assert len(books) == 2
    assert books["long_ret"].iloc[0] > books["ew_ret"].iloc[0]
    assert books["low_ret"].iloc[0] < books["ew_ret"].iloc[0]
    assert books["top_sw_ret"].iloc[0] > books["ew_ret"].iloc[0]
    assert books["score_wt_ret"].iloc[0] > books["ew_ret"].iloc[0]


def test_score_weighted_ret_puts_more_on_higher_score():
    rets = pd.Series([0.10, 0.00])
    score = pd.Series([2.0, 1.0])
    got = score_weighted_ret(rets, score)
    assert abs(got - (2 / 3 * 0.10)) < 1e-12
    assert np.isnan(score_weighted_ret(rets, pd.Series([-1.0, -0.5])))
