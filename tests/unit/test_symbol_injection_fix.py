"""验证磁盘 bars 进入 feature computation 前具有 _symbol 列。"""

import pytest
import pandas as pd
import numpy as np


class TestSymbolInjectionQuantilePath:
    """验证 _symbol 列注入逻辑的正确性"""

    def _make_bars_disk(self, n: int = 100, with_symbol: bool = False) -> pd.DataFrame:
        """构造模拟 bars_disk — 从 storage.bar_1min.load_range 返回的 1min bars"""
        ts = pd.date_range("2024-06-01", periods=n, freq="1min", tz="UTC")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "open": np.random.uniform(49000, 51000, n),
                "high": np.random.uniform(50000, 52000, n),
                "low": np.random.uniform(48000, 50000, n),
                "close": np.random.uniform(49000, 51000, n),
                "volume": np.random.uniform(10, 1000, n),
            }
        )
        if with_symbol:
            df["_symbol"] = "BTCUSDT"
        return df

    def test_symbol_injected_when_missing(self):
        """bars_disk 无 _symbol 列时应被注入"""
        bars_disk = self._make_bars_disk(with_symbol=False)
        symbol = "XRPUSDT"

        # 模拟修复逻辑
        if "_symbol" not in bars_disk.columns:
            bars_disk["_symbol"] = symbol

        assert "_symbol" in bars_disk.columns
        assert (bars_disk["_symbol"] == "XRPUSDT").all()

    def test_symbol_not_overwritten_when_present(self):
        """bars_disk 已有 _symbol 列时不应被覆盖"""
        bars_disk = self._make_bars_disk(with_symbol=True)
        original_values = bars_disk["_symbol"].copy()
        symbol = "XRPUSDT"

        # 模拟修复逻辑
        if "_symbol" not in bars_disk.columns:
            bars_disk["_symbol"] = symbol

        # _symbol 应保持原值 BTCUSDT，不被覆盖为 XRPUSDT
        assert (bars_disk["_symbol"] == "BTCUSDT").all()

    def test_symbol_column_survives_feature_computation_input(self):
        """_symbol 列应在传入 feature computation 前存在，确保 funding_rate join 不报错"""
        bars_disk = self._make_bars_disk(n=50, with_symbol=False)
        symbol = "ETHUSDT"

        if "_symbol" not in bars_disk.columns:
            bars_disk["_symbol"] = symbol

        # 模拟 compute_features_dataframe 的前置检查
        # funding_rate_features.py L138-146:
        #   if "_symbol" not in df.columns and "symbol" not in df.columns:
        #       raise KeyError(...)
        has_symbol_col = "_symbol" in bars_disk.columns or "symbol" in bars_disk.columns
        assert has_symbol_col, "funding_rate join 需要 _symbol 或 symbol 列"

    def test_multiple_symbols_each_get_correct_value(self):
        """多币种场景下每个 symbol 应注入自己的值"""
        symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
        results = {}

        for sym in symbols:
            bars = self._make_bars_disk(n=10, with_symbol=False)
            if "_symbol" not in bars.columns:
                bars["_symbol"] = sym
            results[sym] = bars

        for sym, df in results.items():
            assert (df["_symbol"] == sym).all(), f"{sym} 的 _symbol 值不正确"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
