"""BTC daily ROC90 (±15%) as a research-time feature / gate.

CMS badge / xsection gate semantics (see 20260718_btc_regime_label_compare):
  bull:    roc_90 >= +0.15
  bear:    roc_90 <= -0.15
  neutral: otherwise

Not a FeatureStore column — injected into event_backtest frames or built
inside the Rolling harness. Live must not silently fall back to this path.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_PARQUET_ROOT = _REPO / "data" / "parquet_data"
DEFAULT_CACHE_120T = _REPO / "cache" / "timeframes" / "BTCUSDT_120T.parquet"

ROC90_BULL = 0.15
ROC90_BEAR = -0.15
FEATURE_NAME = "btc_roc_90"


def label_roc90(
    value: float,
    *,
    bull: float = ROC90_BULL,
    bear: float = ROC90_BEAR,
) -> str:
    if value != value:  # NaN
        return "neutral"
    if value >= bull:
        return "bull"
    if value <= bear:
        return "bear"
    return "neutral"


def daily_roc90_from_close(close: pd.Series, *, lookback: int = 90) -> pd.Series:
    """UTC daily close → ROC(lookback). Index = daily timestamps."""
    c = pd.to_numeric(close, errors="coerce").astype(float)
    if not isinstance(c.index, pd.DatetimeIndex):
        raise ValueError("close index must be DatetimeIndex")
    if c.index.tz is None:
        c.index = c.index.tz_localize("UTC")
    else:
        c.index = c.index.tz_convert("UTC")
    daily = c.resample("1D").last().dropna()
    roc = daily / daily.shift(int(lookback)) - 1.0
    roc.name = FEATURE_NAME
    return roc


def align_roc90_to_index(roc_daily: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill daily ROC onto an arbitrary bar index (e.g. 120T)."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    s = roc_daily.copy()
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    else:
        s.index = s.index.tz_convert("UTC")
    s = s.sort_index()
    out = s.reindex(idx, method="ffill")
    out.name = FEATURE_NAME
    return out


def _load_btc_close_from_parquet(root: Path) -> pd.Series:
    files = sorted(root.glob("BTCUSDT_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no BTCUSDT_*.parquet under {root}")
    frames = []
    for p in files:
        df = pd.read_parquet(p, columns=["timestamp", "close"])
        ts = pd.to_datetime(df["timestamp"], utc=True)
        frames.append(pd.Series(df["close"].to_numpy(), index=ts, name="close"))
    close = pd.concat(frames).sort_index()
    close = close[~close.index.duplicated(keep="last")]
    return close


def _load_btc_close_from_cache(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("timestamp")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"bad index in {path}")
    idx = pd.to_datetime(df.index, utc=True)
    return pd.Series(
        pd.to_numeric(df["close"], errors="coerce").to_numpy(),
        index=idx,
        name="close",
    )


@lru_cache(maxsize=4)
def load_btc_daily_roc90(
    source: str = "auto",
    *,
    lookback: int = 90,
) -> pd.Series:
    """Load BTC daily ROC90. source: auto|parquet|cache."""
    if source == "auto":
        if DEFAULT_CACHE_120T.is_file():
            close = _load_btc_close_from_cache(DEFAULT_CACHE_120T)
        elif DEFAULT_PARQUET_ROOT.is_dir():
            close = _load_btc_close_from_parquet(DEFAULT_PARQUET_ROOT)
        else:
            raise FileNotFoundError(
                "need cache/timeframes/BTCUSDT_120T.parquet or data/parquet_data"
            )
    elif source == "cache":
        close = _load_btc_close_from_cache(DEFAULT_CACHE_120T)
    elif source == "parquet":
        close = _load_btc_close_from_parquet(DEFAULT_PARQUET_ROOT)
    else:
        raise ValueError(f"unknown source={source!r}")
    return daily_roc90_from_close(close, lookback=lookback)


def inject_btc_roc90_into_frames(
    sym_data: dict,
    *,
    roc_daily: Optional[pd.Series] = None,
) -> int:
    """Attach FEATURE_NAME onto every tf_features frame in event_backtest sym_data.

    Returns number of frames mutated.
    """
    roc = roc_daily if roc_daily is not None else load_btc_daily_roc90()
    n = 0
    for _sym, data in sym_data.items():
        tf_features = data.get("tf_features") or {}
        for tf, df in list(tf_features.items()):
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            aligned = align_roc90_to_index(roc, df.index)
            out = df.copy()
            out[FEATURE_NAME] = aligned.to_numpy()
            tf_features[tf] = out
            n += 1
    return n


def entry_allow_roc90(
    i: int,
    _close: float,
    ohlc: pd.DataFrame,
    *,
    mode: str = "bull",
    col: str = FEATURE_NAME,
    bull: float = ROC90_BULL,
    bear: float = ROC90_BEAR,
) -> bool:
    """Rolling entry_allow_fn helper. mode: bull|not_bear|off."""
    if mode in ("off", "", "none"):
        return True
    if col not in ohlc.columns:
        return False
    v = ohlc[col].iloc[i]
    try:
        x = float(v)
    except (TypeError, ValueError):
        return False
    if x != x:
        return False
    if mode == "bull":
        return x >= bull
    if mode == "not_bear":
        return x > bear
    raise ValueError(f"unknown roc90 entry mode={mode!r}")
