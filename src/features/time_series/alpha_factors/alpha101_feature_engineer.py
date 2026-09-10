"""Alpha101 feature wrapper.

Vendored base formulas from https://raw.githubusercontent.com/lansetaowa/alpha101-crypto/main/alpha_functions.py
and adapted for single-asset usage.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from . import alpha101_raw
from .alpha_utils import (
    rank,
    scale,
    ts_mean,
    ts_weighted_mean,
)

SeriesLike = pd.Series | pd.DataFrame


@dataclass
class Alpha101FeatureEngineer:
    """Compute Alpha101 factors for a single-asset OHLCV dataframe."""

    included_alphas: Optional[Iterable[str]] = None
    exclude_neutralized: bool = True
    _alpha_funcs: Dict[str, Callable] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        func_dict: Dict[str, Callable] = {
            name: getattr(alpha101_raw, name)
            for name in dir(alpha101_raw)
            if name.startswith("alpha") and callable(getattr(alpha101_raw, name))
        }
        if self.exclude_neutralized:
            func_dict = {
                name: func
                for name, func in func_dict.items()
                if not self._requires_neutralization(func)
            }
        if self.included_alphas is not None:
            selection = set(self.included_alphas)
            func_dict = {
                name: func for name, func in func_dict.items() if name in selection
            }
        self._alpha_funcs = dict(sorted(func_dict.items()))

    @staticmethod
    def _requires_neutralization(func: Callable) -> bool:
        params = inspect.signature(func).parameters
        neutral_args = {"industry", "sector", "subindustry", "IndClass", "cap"}
        return any(name in neutral_args for name in params)

    def compute(
        self,
        df: pd.DataFrame,
        symbol: str = "asset",
        required_features: Optional[set] = None,
    ) -> pd.DataFrame:
        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing columns for Alpha101 computation: {sorted(missing)}"
            )

        base = df[sorted(required_cols)].astype(float).copy()

        # Check and handle duplicate index entries
        if base.index.duplicated().any():
            # Remove duplicates by keeping first occurrence
            base = base[~base.index.duplicated(keep="first")]

        base.columns = pd.Index(base.columns, name="field")

        data_frames = {
            "o": self._to_panel(base["open"], symbol),
            "h": self._to_panel(base["high"], symbol),
            "l": self._to_panel(base["low"], symbol),
            "c": self._to_panel(base["close"], symbol),
            "v": self._to_panel(base["volume"], symbol),
        }

        # Derived inputs
        # CRITICAL: pct_change() uses current and previous data, which is correct for predicting future_return[i]
        # returns[i] = (close[i] - close[i-1]) / close[i-1] is valid for predicting future_return[i]
        data_frames["r"] = (
            data_frames["c"].pct_change().replace([-np.inf, np.inf], np.nan).fillna(0.0)
        )
        data_frames["vwap"] = self._compute_vwap(base, symbol)
        data_frames["adv20"] = ts_mean(data_frames["v"], 20)
        data_frames["adv40"] = ts_mean(data_frames["v"], 40)
        data_frames["adv60"] = ts_mean(data_frames["v"], 60)
        data_frames["adv81"] = ts_mean(data_frames["v"], 81)
        data_frames["adv120"] = ts_mean(data_frames["v"], 120)
        data_frames["adv150"] = ts_mean(data_frames["v"], 150)
        data_frames["adv180"] = ts_mean(data_frames["v"], 180)

        feature_df = pd.DataFrame(index=df.index)

        # 如果指定了required_features，只计算需要的alpha因子
        alpha_funcs_to_compute = self._alpha_funcs
        if required_features:
            # 找出所有需要的alpha101_*特征（格式：alpha101_001, alpha101_002等）
            needed_alphas = []
            for name in self._alpha_funcs.keys():
                # alpha函数名格式：alpha001, alpha002等
                alpha_num = (
                    name[5:] if name.startswith("alpha") else name
                )  # 提取数字部分
                alpha_col_name = (
                    f"alpha101_{alpha_num.zfill(3)}"  # 格式化为 alpha101_001
                )
                # 检查是否有匹配的特征
                if any(
                    f == alpha_col_name
                    or f.startswith(f"alpha101_{alpha_num}")
                    or f.startswith(f"alpha_{alpha_num}")
                    or f == name
                    for f in required_features
                ):
                    needed_alphas.append(name)
            if needed_alphas:
                alpha_funcs_to_compute = {
                    name: self._alpha_funcs[name] for name in needed_alphas
                }
            else:
                # 如果没有匹配的alpha特征，返回空DataFrame
                return feature_df

        for name, func in alpha_funcs_to_compute.items():
            try:
                args = [
                    self._resolve_argument(param, data_frames)
                    for param in inspect.signature(func).parameters
                ]
            except KeyError:
                # Skip functions requiring unavailable inputs (e.g. industry neutralized variants)
                continue

            try:
                raw = func(*args)
            except Exception:
                continue

            formatted = self._format_output(raw, symbol)
            if formatted is None:
                continue

            col_name = f"alpha101_{name[5:].zfill(3)}"
            feature_df[col_name] = formatted

        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

        # Fill NaN values: forward fill first, then backward fill for initial NaNs
        # This handles NaN values caused by rolling windows, shifts, and other time-series operations
        # Note: NaN values at the beginning are normal for time-series features that require historical windows
        # We use forward fill to propagate the first valid value backward, then fill remaining NaNs with 0
        # CRITICAL: Only forward fill (don't backward fill) to avoid using future data
        # Backward fill would use future values to fill past NaNs, causing data leakage
        feature_df = feature_df.ffill().fillna(0.0)

        return feature_df

    def _resolve_argument(
        self, name: str, cache: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        if name in cache:
            return cache[name]
        if name.startswith("adv"):
            window = int(float(name[3:]))
            cache[name] = ts_mean(cache["v"], window)
            return cache[name]
        if name == "vwap":
            return cache["vwap"]
        if name == "cap":
            cache["cap"] = cache["c"] * cache["v"]
            return cache["cap"]
        raise KeyError(name)

    @staticmethod
    def _format_output(result: SeriesLike, symbol: str) -> Optional[pd.Series]:
        if result is None:
            return None
        if isinstance(result, pd.DataFrame):
            data = result.copy()
        else:
            series = result
            if isinstance(series.index, pd.MultiIndex):
                # Check for duplicate index entries before unstack
                if series.index.duplicated().any():
                    # Remove duplicates by keeping first occurrence
                    series = series[~series.index.duplicated(keep="first")]
                if "ticker" in series.index.names:
                    try:
                        data = series.unstack("ticker")
                    except ValueError as e:
                        if "duplicate entries" in str(e).lower():
                            # If still fails, try resetting index and handling differently
                            series_reset = series.reset_index()
                            if "ticker" in series_reset.columns:
                                data = series_reset.set_index(
                                    [
                                        col
                                        for col in series_reset.columns
                                        if col != "ticker"
                                        and col != series_reset.columns[-1]
                                    ]
                                )[series_reset.columns[-1]].to_frame(symbol)
                            else:
                                return None
                        else:
                            raise
                else:
                    try:
                        data = series.unstack(level=-1)
                    except ValueError as e:
                        if "duplicate entries" in str(e).lower():
                            # If still fails, try resetting index and handling differently
                            series_reset = series.reset_index()
                            if len(series_reset.columns) > 1:
                                data = series_reset.set_index(series_reset.columns[0])[
                                    series_reset.columns[-1]
                                ].to_frame(symbol)
                            else:
                                return None
                        else:
                            raise
            else:
                data = series.to_frame(symbol)
        # Skip empty frames to avoid shape mismatch errors
        if data is None or data.shape[1] == 0:
            return None
        if symbol not in data.columns:
            if isinstance(data.columns, pd.MultiIndex):
                level = data.columns.get_level_values(-1)
                if symbol in level:
                    extracted = data.xs(symbol, axis=1, level=-1)
                    if isinstance(extracted, pd.DataFrame):
                        if extracted.shape[1] == 0:
                            return None
                        column = extracted.iloc[:, 0]
                    else:
                        column = extracted
                    column.name = symbol
                    return column
            if data.shape[1] == 1:
                column = data.iloc[:, 0]
                column.name = symbol
                return column
            return None
        column = data[symbol]
        column.name = symbol
        return column

    @staticmethod
    def _to_panel(series: pd.Series, symbol: str) -> pd.DataFrame:
        df = series.to_frame(symbol)
        df.columns.name = "ticker"
        return df

    def _compute_vwap(self, base: pd.DataFrame, symbol: str) -> pd.DataFrame:
        # CRITICAL: VWAP calculation should only use historical data
        # Using cumsum() is correct as it only uses data up to current time
        price = (base["high"] + base["low"] + base["close"]) / 3.0
        volume = base["volume"].replace(0, np.nan)
        cumulative = (price * volume).cumsum()
        # Only use forward fill (not backward fill) to avoid future data leakage
        vwap_series = cumulative.div(volume.cumsum()).ffill().fillna(price)
        df = vwap_series.to_frame(symbol)
        df.columns.name = "ticker"
        return df
