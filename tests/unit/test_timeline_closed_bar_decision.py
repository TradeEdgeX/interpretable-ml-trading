"""Timeline must not consume a bar's features at that bar's open (2h_close)."""

from __future__ import annotations

import pandas as pd

from scripts.backtest_multileg_timeline import _lookup, _prev_signal_ts


def test_prev_signal_ts_2h_and_8h() -> None:
    assert _prev_signal_ts(
        pd.Timestamp("2026-07-23 22:00:00", tz="UTC"), "2h"
    ) == pd.Timestamp("2026-07-23 20:00:00", tz="UTC")
    assert _prev_signal_ts(
        pd.Timestamp("2026-07-23 16:00:00", tz="UTC"), "8h"
    ) == pd.Timestamp("2026-07-23 08:00:00", tz="UTC")


def test_lookup_at_fire_open_must_use_previous_closed_row() -> None:
    """At wall 22:00, only feats[20:00] are live-known; feats[22:00] are future."""
    idx = pd.to_datetime(
        [
            "2026-07-23 18:00:00",
            "2026-07-23 20:00:00",
            "2026-07-23 22:00:00",
        ],
        utc=True,
    )
    feats = {
        "BTCUSDT": pd.DataFrame(
            {
                "grid_semantic_chop": [0.0, 0.75, 0.99],
                "box_pos_60": [0.34, 0.44, 0.50],
                "close": [64820.0, 65137.0, 65053.0],
            },
            index=idx,
        )
    }
    fire = pd.Timestamp("2026-07-23 22:00:00", tz="UTC")
    closed = _prev_signal_ts(fire, "2h")
    assert closed == pd.Timestamp("2026-07-23 20:00:00", tz="UTC")

    # Wrong (old) path: lookup at fire open sees 22:00 row → lookahead chop=0.99
    leak = _lookup(feats, "BTCUSDT", fire)
    assert leak["grid_semantic_chop"] == 0.99

    # Correct path: previous closed bar only
    ok = _lookup(feats, "BTCUSDT", closed)
    assert ok["grid_semantic_chop"] == 0.75
    assert ok["box_pos_60"] == 0.44


def test_at_bar_open_previous_row_is_not_the_enter_bar() -> None:
    """Wall 20:00 must still see 18:00 (chop=0), not the eventually-closed 20:00."""
    idx = pd.to_datetime(
        ["2026-07-23 18:00:00", "2026-07-23 20:00:00"],
        utc=True,
    )
    feats = {
        "BTCUSDT": pd.DataFrame(
            {"grid_semantic_chop": [0.0, 0.75], "box_pos_60": [0.34, 0.44]},
            index=idx,
        )
    }
    fire = pd.Timestamp("2026-07-23 20:00:00", tz="UTC")
    closed = _prev_signal_ts(fire, "2h")
    f = _lookup(feats, "BTCUSDT", closed)
    assert f["grid_semantic_chop"] == 0.0
