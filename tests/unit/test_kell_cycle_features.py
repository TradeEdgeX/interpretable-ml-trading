"""Kell stage features: point-in-time, no future rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ashare.kell_cycle_features import kell_money_pattern_frame, kell_stage_frame


def _frame(closes, volumes=None):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    closes = np.asarray(closes, dtype=float)
    df = pd.DataFrame(
        {
            "date": idx,
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": (
                np.asarray(volumes, dtype=float)
                if volumes is not None
                else np.full(len(closes), 100.0)
            ),
        }
    )
    return df


def test_no_future_leak_tail_change_keeps_head():
    closes = list(np.linspace(100, 80, 40)) + list(np.linspace(80, 110, 40))
    vols = [100.0] * 40 + [300.0] * 5 + [100.0] * 35
    full = kell_stage_frame(_frame(closes, vols))
    cut = kell_stage_frame(_frame(closes[:50], vols[:50]))
    cols = ["rev_ext", "wedge_pop", "ema_cross_back", "base_break", "exhaustion"]
    pd.testing.assert_frame_equal(
        full[cols].iloc[:50].reset_index(drop=True),
        cut[cols].reset_index(drop=True),
    )


def test_reversal_extension_requires_deep_discount():
    # steady grind down with a volume spike at the bottom
    closes = list(np.linspace(100, 70, 60))
    vols = [100.0] * 55 + [400.0] * 5
    out = kell_stage_frame(_frame(closes, vols))
    assert out["rev_ext"].iloc[-5:].any()
    # quiet flat tape should not flag extension
    flat = kell_stage_frame(_frame([100.0] * 60))
    assert not flat["rev_ext"].any()


def test_wedge_pop_is_first_cross_after_extension():
    closes = list(np.linspace(100, 70, 50)) + [  # downtrend → extension
        72.0,
        74.0,
        76.0,
        78.0,
        80.0,
        85.0,
        90.0,
    ]  # recover through EMA21
    vols = [100.0] * 45 + [400.0] * 5 + [120.0] * 7
    out = kell_stage_frame(_frame(closes, vols))
    pops = out.index[out["wedge_pop"]]
    assert len(pops) >= 1
    # pop must be a cross-up bar (previous close still below EMA21)
    first = pops[0]
    loc = out.index.get_loc(first)
    assert closes[loc] > out["ema_slow"].iloc[loc]
    # no pop without a prior extension
    calm = kell_stage_frame(_frame(list(np.linspace(50, 90, 60))))
    assert not calm["wedge_pop"].any()


def test_exhaustion_flags_upside_extension():
    closes = list(np.linspace(50, 50, 40)) + list(np.linspace(50, 130, 25))
    vols = [100.0] * 40 + [500.0] * 25
    out = kell_stage_frame(_frame(closes, vols))
    assert out["exhaustion"].iloc[-10:].any()


def test_simulate_flattens_on_kell_exhaustion():
    from ashare.cs_meanrev_signals import wilder_rsi
    from auxiliary.research.ashare_pool_detector_uplift import (
        SlotPolicy,
        SymbolPanel,
        simulate,
    )

    idx = pd.date_range("2024-01-02", periods=80, freq="B")
    close = pd.Series(np.linspace(10.0, 20.0, 80), index=idx)
    rsi = wilder_rsi(close)
    amp = pd.Series(12.0, index=idx)
    dist = pd.Series(-30.0, index=idx)
    panels = {
        "000001": SymbolPanel(close=close, rsi=rsi, amplitude=amp, dist_ma200_pct=dist)
    }
    fire = idx[25]
    events = {fire.normalize(): [{"symbol": "000001", "score": 1.0}]}
    exh = pd.Series(False, index=idx)
    cut = idx[35]
    exh.loc[cut] = True
    res = simulate(
        "core554",
        "exh",
        panels,
        events,
        list(idx),
        slots=1,
        hold_days=40,
        policy=SlotPolicy(kell_exhaustion={"000001": exh}, exit_min_hold_days=0),
    )
    assert res.n_kell_exits == 1
    assert len(res.trades) == 1
    assert res.trades[0].exit_date == cut.normalize()


def _money_bottom_then_pop():
    """Crash → tight chop under the averages → pivot break."""
    down = list(np.linspace(100.0, 68.0, 45))
    chop = [70.0, 71.0, 70.5, 71.5, 70.8, 71.2] * 6  # 36 bars, ~70–71.5
    pop = [76.0, 78.0]
    closes = down + chop + pop
    vols = [100.0] * 40 + [400.0] * 5 + [80.0] * 36 + [250.0, 250.0]
    highs = [c * 1.01 for c in down] + [c + 0.4 for c in chop] + [77.0, 79.0]
    lows = [c * 0.99 for c in down] + [c - 0.4 for c in chop] + [74.0, 75.5]
    idx = pd.date_range("2023-01-02", periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "date": idx,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
        }
    )


def test_money_pattern_no_future_leak():
    df = _money_bottom_then_pop()
    full = kell_money_pattern_frame(df)
    cut_n = 60
    cut = kell_money_pattern_frame(df.iloc[:cut_n].copy())
    cols = ["rev_ext", "contraction", "wedge_pop", "exhaustion"]
    pd.testing.assert_frame_equal(
        full[cols].iloc[:cut_n].reset_index(drop=True),
        cut[cols].reset_index(drop=True),
    )


def test_money_pattern_requires_contraction_then_pivot():
    df = _money_bottom_then_pop()
    out = kell_money_pattern_frame(df)
    assert out["rev_ext"].any()
    assert out["contraction"].any()
    pops = list(out.index[out["wedge_pop"]])
    assert len(pops) >= 1
    first_rev = out.index[out["rev_ext"]][0]
    first_con = out.index[out["contraction"]][0]
    first_pop = pops[0]
    assert first_rev < first_con <= first_pop
    loc = out.index.get_loc(first_pop)
    assert df["close"].iloc[loc] > out["ema_slow"].iloc[loc]
    assert df["close"].iloc[loc] > out["pivot_high"].iloc[loc]


def test_money_pattern_rejects_bare_ma_cross():
    # Recover through the MA with no tight base — diluted pop may fire, strict must not.
    closes = list(np.linspace(100.0, 70.0, 40)) + list(np.linspace(70.0, 95.0, 20))
    vols = [100.0] * 35 + [400.0] * 5 + [120.0] * 20
    raw = _frame(closes, vols)
    diluted = kell_stage_frame(raw)
    strict = kell_money_pattern_frame(raw)
    assert diluted["wedge_pop"].any()
    assert not strict["wedge_pop"].any()


def test_simulate_fail_fast_and_scale_in():
    from ashare.cs_meanrev_signals import wilder_rsi
    from auxiliary.research.ashare_pool_detector_uplift import (
        SlotPolicy,
        SymbolPanel,
        simulate,
    )

    idx = pd.date_range("2024-01-02", periods=40, freq="B")
    close = pd.Series([10.0] * 12 + [8.0] * 28, index=idx)
    rsi = wilder_rsi(close)
    amp = pd.Series(12.0, index=idx)
    dist = pd.Series(-10.0, index=idx)
    panels = {
        "000001": SymbolPanel(close=close, rsi=rsi, amplitude=amp, dist_ma200_pct=dist)
    }
    fire = idx[5]
    add_day = idx[7]
    events = {fire.normalize(): [{"symbol": "000001", "score": 1.0}]}
    fail = pd.Series(9.0, index=idx)
    adds = pd.Series(False, index=idx)
    adds.loc[add_day] = True
    res = simulate(
        "core554",
        "fail_add",
        panels,
        events,
        list(idx),
        slots=2,
        hold_days=40,
        policy=SlotPolicy(
            kell_fail_below={"000001": fail},
            kell_scale_in={"000001": adds},
            kell_scale_in_max=1,
            kell_scale_in_frac=0.5,
            exit_min_hold_days=0,
        ),
    )
    assert res.n_kell_adds == 1
    assert res.n_kell_fail_exits == 1
    assert len(res.trades) == 1
    assert res.trades[0].exit_date == idx[12].normalize()


def test_canslim_fail_closed_on_missing_fields():
    from ashare.kell_canslim import attach_canslim_mask

    ev = pd.DataFrame({"symbol": ["999999"], "date": [pd.Timestamp("2024-06-01")]})
    out = attach_canslim_mask(ev)
    assert bool(out["canslim_ok"].iloc[0]) is False


def test_pit_snapshot_plus90_calendar_lag(tmp_path, monkeypatch):
    from ashare import kell_canslim as kc

    fin = tmp_path / "financial"
    val = tmp_path / "valuation"
    fin.mkdir()
    val.mkdir()
    pd.DataFrame(
        {
            "日期": [pd.Timestamp("2020-12-31")],
            "净利润增长率(%)": [40.0],
            "主营业务收入增长率(%)": [35.0],
            "净资产收益率(%)": [18.0],
            "基本每股收益": [1.2],
        }
    ).to_parquet(fin / "000001.parquet")
    pd.DataFrame(
        {
            "数据日期": [pd.Timestamp("2021-03-01")],
            "流通市值": [80e8],
        }
    ).to_parquet(val / "000001.parquet")
    monkeypatch.setattr(kc, "FIN_DIR", fin)
    monkeypatch.setattr(kc, "VAL_DIR", val)
    dates = pd.DatetimeIndex(["2021-03-30", "2021-03-31", "2021-04-01"])
    snap = kc.pit_snapshot("000001", dates)
    assert pd.isna(snap.loc[pd.Timestamp("2021-03-30"), "roe"])
    assert float(snap.loc[pd.Timestamp("2021-03-31"), "roe"]) == 18.0
    assert float(snap.loc[pd.Timestamp("2021-04-01"), "circ_mcap_yi"]) == 80.0
