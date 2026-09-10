"""Tests for BTC (anchor) macro_tp_vwap_1200_position overlay."""

from __future__ import annotations

import pandas as pd

from src.features.cross_symbol.macro_tp_vwap_anchor import (
    ANCHOR_COLUMN,
    apply_macro_tp_vwap_anchor,
    apply_macro_tp_vwap_from_anchor_frame,
    parse_macro_tp_vwap_anchor_config,
)


def test_parse_defaults_enabled_btc():
    en, sym = parse_macro_tp_vwap_anchor_config()
    assert en is True
    assert sym == "BTCUSDT"


def test_parse_disabled_false_shorthand():
    en, sym = parse_macro_tp_vwap_anchor_config(
        meta_yaml_full={"macro_tp_vwap_anchor": False}
    )
    assert en is False


def test_apply_overlay_multi_symbol():
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"], utc=True
            ),
            "symbol": ["BTCUSDT", "XRPUSDT"],
            ANCHOR_COLUMN: [0.5, 0.1],
        }
    )
    out = apply_macro_tp_vwap_anchor(
        df,
        anchor_symbol="BTCUSDT",
        enabled=True,
        symbol_col="symbol",
        time_col="datetime",
    )
    assert float(out.loc[out["symbol"] == "BTCUSDT", ANCHOR_COLUMN].iloc[0]) == 0.5
    assert float(out.loc[out["symbol"] == "XRPUSDT", ANCHOR_COLUMN].iloc[0]) == 0.5


def test_apply_overlay_multi_timestamp_proves_alt_tracks_btc_not_local():
    """ALT 在多个时间戳上应与 BTC 列一致，而非保留本地原值。"""
    t = pd.to_datetime(
        [
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z",
            "2024-01-02T12:00:00Z",
            "2024-01-02T12:00:00Z",
            "2024-01-03T08:00:00Z",
            "2024-01-03T08:00:00Z",
        ],
        utc=True,
    )
    df = pd.DataFrame(
        {
            "datetime": t,
            "symbol": ["BTCUSDT", "XRPUSDT"] * 3,
            ANCHOR_COLUMN: [0.5, 0.09, 0.72, 0.11, 0.33, 0.07],
        }
    )
    out = apply_macro_tp_vwap_anchor(
        df,
        anchor_symbol="BTCUSDT",
        enabled=True,
        symbol_col="symbol",
        time_col="datetime",
    )
    for ts in t.unique():
        b_local = float(
            df.loc[
                (df["datetime"] == ts) & (df["symbol"] == "BTCUSDT"), ANCHOR_COLUMN
            ].iloc[0]
        )
        a_local = float(
            df.loc[
                (df["datetime"] == ts) & (df["symbol"] == "XRPUSDT"), ANCHOR_COLUMN
            ].iloc[0]
        )
        assert a_local != b_local
        b_out = float(
            out.loc[
                (out["datetime"] == ts) & (out["symbol"] == "BTCUSDT"), ANCHOR_COLUMN
            ].iloc[0]
        )
        a_out = float(
            out.loc[
                (out["datetime"] == ts) & (out["symbol"] == "XRPUSDT"), ANCHOR_COLUMN
            ].iloc[0]
        )
        assert a_out == b_out == b_local


def test_apply_from_anchor_frame():
    main = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01T00:00:00Z"], utc=True),
            ANCHOR_COLUMN: [0.99],
        }
    )
    anchor = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01T00:00:00Z"], utc=True),
            ANCHOR_COLUMN: [0.42],
        }
    )
    out = apply_macro_tp_vwap_from_anchor_frame(main, anchor, time_col="datetime")
    assert float(out[ANCHOR_COLUMN].iloc[0]) == 0.42
