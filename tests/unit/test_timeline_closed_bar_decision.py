"""Decision clock: wall T may only read the previous closed FeatureStore row."""

from __future__ import annotations

import pandas as pd

from scripts.event_backtest import (
    _align_feature_index_to_bar_close,
    _feature_asof_from_sym_tf_features,
    _timeframe_to_timedelta,
)


def test_prev_closed_ts_2h_and_8h() -> None:
    delta_2h = _timeframe_to_timedelta("2H")
    delta_8h = _timeframe_to_timedelta("8H")
    assert delta_2h == pd.Timedelta(hours=2)
    assert delta_8h == pd.Timedelta(hours=8)
    fire = pd.Timestamp("2026-07-23 22:00:00", tz="UTC")
    assert fire - delta_2h == pd.Timestamp("2026-07-23 20:00:00", tz="UTC")
    assert pd.Timestamp("2026-07-23 16:00:00", tz="UTC") - delta_8h == pd.Timestamp(
        "2026-07-23 08:00:00", tz="UTC"
    )


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
    raw = pd.DataFrame(
        {
            "grid_semantic_chop": [0.0, 0.75, 0.99],
            "box_pos_60": [0.34, 0.44, 0.50],
            "close": [64820.0, 65137.0, 65053.0],
        },
        index=idx,
    )
    aligned = _align_feature_index_to_bar_close(raw, "2H")
    bundle = {"tf_features": {"2H": aligned}}
    fire = pd.Timestamp("2026-07-23 22:00:00", tz="UTC")

    leak = _feature_asof_from_sym_tf_features(
        {"tf_features": {"2H": raw}}, fire, "grid_semantic_chop"
    )
    assert leak == 0.99

    ok = _feature_asof_from_sym_tf_features(bundle, fire, "grid_semantic_chop")
    assert ok == 0.75
    assert _feature_asof_from_sym_tf_features(bundle, fire, "box_pos_60") == 0.44


def test_at_bar_open_previous_row_is_not_the_enter_bar() -> None:
    """Wall 20:00 must still see 18:00 (chop=0), not the eventually-closed 20:00."""
    idx = pd.to_datetime(
        ["2026-07-23 18:00:00", "2026-07-23 20:00:00"],
        utc=True,
    )
    raw = pd.DataFrame(
        {"grid_semantic_chop": [0.0, 0.75], "box_pos_60": [0.34, 0.44]},
        index=idx,
    )
    aligned = _align_feature_index_to_bar_close(raw, "2H")
    bundle = {"tf_features": {"2H": aligned}}
    fire = pd.Timestamp("2026-07-23 20:00:00", tz="UTC")
    assert (
        _feature_asof_from_sym_tf_features(bundle, fire, "grid_semantic_chop") == 0.0
    )
