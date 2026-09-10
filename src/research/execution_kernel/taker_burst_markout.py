"""Trade-clock taker-burst markout (absorption vs displacement).

Calendar bars / FeatureStore IC are the wrong clock. Events are 1-second
aggregates of Binance aggTrades. ``y`` is signed markout in basis points
after the burst, not next-bar close-to-close RR.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

SEC1_COLUMNS = (
    "timestamp",
    "last",
    "high",
    "low",
    "vwap",
    "buy_qty",
    "sell_qty",
    "n_trades",
)

DEFAULT_HORIZONS_S: tuple[int, ...] = (5, 30, 60, 120, 300, 900)
DEFAULT_CHUNKSIZE = 1_000_000


@dataclass(frozen=True)
class BurstSpec:
    window_s: int = 10
    lookback_s: int = 3600
    vol_quantile: float = 0.99
    min_imbalance: float = 0.6
    refractory_s: int = 30
    pre_range_s: int = 60
    touch_bp: float = 5.0
    fee_rt_bp: float = 10.0
    horizons_s: tuple[int, ...] = DEFAULT_HORIZONS_S


def _aggtrade_read_params(first_line: str) -> dict:
    params: dict = {"low_memory": False}
    head = first_line.strip().split(",")[0].replace(".", "")
    if head.isdigit():
        params.update(
            {
                "header": None,
                "names": [
                    "agg_trade_id",
                    "price",
                    "quantity",
                    "first_trade_id",
                    "last_trade_id",
                    "transact_time",
                    "is_buyer_maker",
                ],
            }
        )
    return params


def _chunk_to_1s(chunk: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(chunk["transact_time"], unit="ms", utc=True)
    price = pd.to_numeric(chunk["price"], errors="coerce")
    qty = pd.to_numeric(chunk["quantity"], errors="coerce")
    side = np.where(chunk["is_buyer_maker"].astype(bool), -1, 1)
    ok = price.notna() & qty.notna() & (qty > 0)
    work = pd.DataFrame(
        {
            "timestamp": ts[ok].dt.floor("s"),
            "price": price[ok].to_numpy(),
            "qty": qty[ok].to_numpy(),
            "side": side[ok],
        }
    )
    if work.empty:
        return pd.DataFrame(columns=list(SEC1_COLUMNS))
    buy = np.where(work["side"] == 1, work["qty"], 0.0)
    sell = np.where(work["side"] == -1, work["qty"], 0.0)
    work = work.assign(buy_qty=buy, sell_qty=sell, px_qty=work["price"] * work["qty"])
    grouped = work.groupby("timestamp", sort=True)
    out = grouped.agg(
        last=("price", "last"),
        high=("price", "max"),
        low=("price", "min"),
        buy_qty=("buy_qty", "sum"),
        sell_qty=("sell_qty", "sum"),
        n_trades=("qty", "size"),
        px_qty=("px_qty", "sum"),
    )
    tot = out["buy_qty"] + out["sell_qty"]
    out["vwap"] = out["px_qty"] / tot.replace(0.0, np.nan)
    out = out.drop(columns=["px_qty"]).reset_index()
    return out.loc[:, list(SEC1_COLUMNS)]


def _merge_1s_parts(parts: Sequence[pd.DataFrame]) -> pd.DataFrame:
    frames = [p for p in parts if p is not None and not p.empty]
    if not frames:
        return pd.DataFrame(columns=list(SEC1_COLUMNS))
    raw = pd.concat(frames, ignore_index=True)
    grouped = raw.groupby("timestamp", sort=True)
    tot_buy = grouped["buy_qty"].sum()
    tot_sell = grouped["sell_qty"].sum()
    tot = tot_buy + tot_sell
    vwap_num = (raw["vwap"] * (raw["buy_qty"] + raw["sell_qty"])).groupby(
        raw["timestamp"]
    ).sum()
    out = pd.DataFrame(
        {
            "timestamp": tot_buy.index,
            "last": grouped["last"].last().to_numpy(),
            "high": grouped["high"].max().to_numpy(),
            "low": grouped["low"].min().to_numpy(),
            "vwap": (vwap_num / tot.replace(0.0, np.nan)).to_numpy(),
            "buy_qty": tot_buy.to_numpy(),
            "sell_qty": tot_sell.to_numpy(),
            "n_trades": grouped["n_trades"].sum().to_numpy(),
        }
    )
    return out


def aggregate_aggtrades_zip_to_1s(
    zip_path: str | Path,
    *,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> pd.DataFrame:
    """Stream a monthly Vision aggTrades zip into 1-second bars (one row / second)."""
    zip_path = Path(zip_path)
    parts: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(csv_name) as handle:
            first_line = handle.readline().decode("utf-8", errors="ignore")
        params = _aggtrade_read_params(first_line)
        with zf.open(csv_name) as handle:
            reader = pd.read_csv(handle, chunksize=chunksize, **params)
            for chunk in reader:
                parts.append(_chunk_to_1s(chunk))
    sec = _merge_1s_parts(parts)
    if sec.empty:
        return sec
    sec["timestamp"] = pd.to_datetime(sec["timestamp"], utc=True)
    return sec.sort_values("timestamp").reset_index(drop=True)


def load_or_build_sec1(
    zip_path: str | Path,
    cache_path: str | Path,
    *,
    force: bool = False,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> pd.DataFrame:
    cache_path = Path(cache_path)
    if cache_path.exists() and cache_path.stat().st_size > 0 and not force:
        sec = pd.read_parquet(cache_path)
        sec["timestamp"] = pd.to_datetime(sec["timestamp"], utc=True)
        return sec.sort_values("timestamp").reset_index(drop=True)
    sec = aggregate_aggtrades_zip_to_1s(zip_path, chunksize=chunksize)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    sec.to_parquet(cache_path, index=False)
    return sec


def _reindex_seconds(sec: pd.DataFrame) -> pd.DataFrame:
    work = sec.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    work = work.sort_values("timestamp").drop_duplicates("timestamp")
    full_idx = pd.date_range(
        work["timestamp"].iloc[0],
        work["timestamp"].iloc[-1],
        freq="s",
        tz="UTC",
    )
    work = work.set_index("timestamp").reindex(full_idx)
    work["last"] = work["last"].ffill()
    work["high"] = work["high"].fillna(work["last"])
    work["low"] = work["low"].fillna(work["last"])
    work["vwap"] = work["vwap"].fillna(work["last"])
    for col in ("buy_qty", "sell_qty", "n_trades"):
        work[col] = work[col].fillna(0.0)
    work.index.name = "timestamp"
    return work.reset_index()


def detect_bursts(sec: pd.DataFrame, spec: BurstSpec | None = None) -> pd.DataFrame:
    """P99 volume windows with directional imbalance; one peak per refractory."""
    spec = spec or BurstSpec()
    work = _reindex_seconds(sec)
    tot = work["buy_qty"] + work["sell_qty"]
    signed = work["buy_qty"] - work["sell_qty"]
    vol_w = tot.rolling(spec.window_s, min_periods=spec.window_s).sum()
    signed_w = signed.rolling(spec.window_s, min_periods=spec.window_s).sum()
    imb = signed_w / vol_w.replace(0.0, np.nan)
    thresh = vol_w.shift(1).rolling(
        spec.lookback_s, min_periods=max(60, spec.lookback_s // 4)
    ).quantile(spec.vol_quantile)
    raw = (vol_w >= thresh) & (imb.abs() >= spec.min_imbalance) & thresh.notna()
    events = _peak_with_refractory(
        work["timestamp"].to_numpy(),
        vol_w.to_numpy(),
        raw.to_numpy(dtype=bool),
        refractory_s=spec.refractory_s,
    )
    if not events:
        return pd.DataFrame()
    idx = np.asarray(events, dtype=int)
    pre_high = (
        work["high"]
        .shift(spec.window_s)
        .rolling(spec.pre_range_s, min_periods=max(5, spec.pre_range_s // 4))
        .max()
    )
    pre_low = (
        work["low"]
        .shift(spec.window_s)
        .rolling(spec.pre_range_s, min_periods=max(5, spec.pre_range_s // 4))
        .min()
    )
    sign = np.sign(signed_w.to_numpy()[idx])
    last = work["last"].to_numpy()[idx]
    displaced = ((sign > 0) & (last > pre_high.to_numpy()[idx])) | (
        (sign < 0) & (last < pre_low.to_numpy()[idx])
    )
    out = pd.DataFrame(
        {
            "timestamp": work["timestamp"].to_numpy()[idx],
            "bar_i": idx,
            "sign": sign.astype(int),
            "vol_w": vol_w.to_numpy()[idx],
            "signed_w": signed_w.to_numpy()[idx],
            "imbalance": imb.to_numpy()[idx],
            "px": last,
            "pre_high": pre_high.to_numpy()[idx],
            "pre_low": pre_low.to_numpy()[idx],
            "displaced": displaced.astype(bool),
        }
    )
    out["sleeve"] = np.where(out["displaced"], "displaced", "absorbed")
    return out


def _peak_with_refractory(
    timestamps: np.ndarray,
    vol_w: np.ndarray,
    mask: np.ndarray,
    *,
    refractory_s: int,
) -> list[int]:
    hits = np.flatnonzero(mask & np.isfinite(vol_w))
    if hits.size == 0:
        return []
    kept: list[int] = []
    run_peak = int(hits[0])
    last_hit = int(hits[0])
    ts_ns = pd.DatetimeIndex(timestamps).asi8
    for i in hits[1:]:
        i = int(i)
        if i == last_hit + 1:
            if vol_w[i] > vol_w[run_peak]:
                run_peak = i
            last_hit = i
            continue
        kept.append(run_peak)
        run_peak = i
        last_hit = i
    kept.append(run_peak)
    filtered: list[int] = []
    last_ts = None
    for i in kept:
        t = int(ts_ns[i])
        if last_ts is not None and (t - last_ts) < refractory_s * 1_000_000_000:
            continue
        filtered.append(i)
        last_ts = t
    return filtered


def attach_markout(
    sec: pd.DataFrame,
    events: pd.DataFrame,
    spec: BurstSpec | None = None,
) -> pd.DataFrame:
    spec = spec or BurstSpec()
    if events.empty:
        return events.copy()
    work = _reindex_seconds(sec)
    last = work["last"].to_numpy()
    n = len(work)
    out = events.copy()
    p0 = out["px"].to_numpy()
    sign = out["sign"].to_numpy()
    i0 = out["bar_i"].to_numpy()
    for h in spec.horizons_s:
        j = np.minimum(i0 + int(h), n - 1)
        valid = i0 + int(h) < n
        y = np.where(valid, 1e4 * sign * (last[j] - p0) / p0, np.nan)
        out[f"y_{h}s_bp"] = y
    touch_same, touch_opp, touch_none = _first_touch(
        last, i0, p0, sign, touch_bp=spec.touch_bp, max_s=max(spec.horizons_s)
    )
    out["first_touch"] = np.where(
        touch_same, "same", np.where(touch_opp, "opp", "none")
    )
    out["touch_same"] = touch_same
    out["touch_opp"] = touch_opp
    out["touch_none"] = touch_none
    return out


def _first_touch(
    last: np.ndarray,
    i0: np.ndarray,
    p0: np.ndarray,
    sign: np.ndarray,
    *,
    touch_bp: float,
    max_s: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(last)
    same = np.zeros(len(i0), dtype=bool)
    opp = np.zeros(len(i0), dtype=bool)
    none = np.zeros(len(i0), dtype=bool)
    thresh = touch_bp / 1e4
    for k, i in enumerate(i0):
        i = int(i)
        end = min(i + int(max_s), n - 1)
        if end <= i:
            none[k] = True
            continue
        path = sign[k] * (last[i + 1 : end + 1] - p0[k]) / p0[k]
        hit_same = int(np.argmax(path >= thresh)) if np.any(path >= thresh) else -1
        hit_opp = int(np.argmax(path <= -thresh)) if np.any(path <= -thresh) else -1
        if hit_same < 0 and hit_opp < 0:
            none[k] = True
        elif hit_opp < 0 or (hit_same >= 0 and hit_same <= hit_opp):
            same[k] = True
        else:
            opp[k] = True
    return same, opp, none


def summarize_markout(
    events: pd.DataFrame,
    spec: BurstSpec | None = None,
    *,
    month: str | None = None,
    symbol: str | None = None,
) -> pd.DataFrame:
    spec = spec or BurstSpec()
    if events.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    groups: Iterable[tuple[str, pd.DataFrame]] = [("all", events)]
    if "sleeve" in events.columns:
        groups = [
            ("all", events),
            ("absorbed", events.loc[events["sleeve"] == "absorbed"]),
            ("displaced", events.loc[events["sleeve"] == "displaced"]),
        ]
    for name, part in groups:
        if part.empty:
            continue
        row: dict = {
            "month": month,
            "symbol": symbol,
            "sleeve": name,
            "n": int(len(part)),
            "n_same_first": int(part["touch_same"].sum()) if "touch_same" in part else 0,
            "n_opp_first": int(part["touch_opp"].sum()) if "touch_opp" in part else 0,
            "n_no_touch": int(part["touch_none"].sum()) if "touch_none" in part else 0,
            "p_same_first": float(part["touch_same"].mean()) if "touch_same" in part else np.nan,
            "p_opp_first": float(part["touch_opp"].mean()) if "touch_opp" in part else np.nan,
        }
        for h in spec.horizons_s:
            col = f"y_{h}s_bp"
            if col not in part:
                continue
            y = part[col].dropna()
            mean = float(y.mean()) if len(y) else np.nan
            row[f"mean_cont_{h}s_bp"] = mean
            row[f"p_cont_{h}s"] = float((y > 0).mean()) if len(y) else np.nan
            row[f"mom_net_{h}s_bp"] = mean - spec.fee_rt_bp
            row[f"fade_net_{h}s_bp"] = (-mean) - spec.fee_rt_bp
        rows.append(row)
    return pd.DataFrame(rows)


def run_month(
    sec: pd.DataFrame,
    spec: BurstSpec | None = None,
    *,
    month: str | None = None,
    symbol: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = spec or BurstSpec()
    events = detect_bursts(sec, spec)
    events = attach_markout(sec, events, spec)
    summary = summarize_markout(events, spec, month=month, symbol=symbol)
    return events, summary
