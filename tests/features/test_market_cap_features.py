"""
Market Cap 特征测试

测试内容：
1. 无未来函数测试（修改未来数据不影响历史特征值）⭐⭐⭐⭐⭐
2. 多资产归一化测试（特征分布对齐）⭐⭐⭐⭐
3. 流式vs批量一致性测试 ⭐⭐⭐⭐
4. 特征数学正确性验证
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import tempfile
import shutil

from src.features.time_series.market_cap_features import (
    compute_market_cap_normalized_orderflow_from_df,
)
from src.features.normalization.feature_contract import (
    collect_feature_normalization_meta,
)


def create_mock_market_cap_data(symbol: str, market_cap_dir: Path, n_days: int = 100):
    """创建模拟的 market cap 数据文件"""
    market_cap_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    # 模拟市值数据（随时间增长）
    market_caps = 1e9 + np.cumsum(np.random.randn(n_days) * 1e7)  # 从10亿开始

    df = pd.DataFrame(
        {"market_cap_usd": market_caps},
        index=dates,
    )

    output_path = market_cap_dir / f"{symbol}.parquet"
    df.to_parquet(output_path)
    return output_path


def create_mock_data(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """创建模拟数据用于测试"""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="4H")

    # 生成价格数据
    returns = np.random.randn(n_samples) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_samples) * 0.001),
            "high": prices * (1 + np.abs(np.random.randn(n_samples) * 0.002)),
            "low": prices * (1 - np.abs(np.random.randn(n_samples) * 0.002)),
            "close": prices,
            "volume": np.random.uniform(1000, 10000, n_samples),
        },
        index=dates,
    )

    # 添加 symbol 列（用于 market cap join）
    df["_symbol"] = "BTCUSDT"

    # 添加 buy_qty 和 sell_qty（用于 net_buy_qty 计算）
    df["buy_qty"] = np.random.uniform(100, 1000, n_samples)
    df["sell_qty"] = np.random.uniform(100, 1000, n_samples)

    return df


class TestMarketCapFeatures:
    """Market Cap 特征测试"""

    @pytest.fixture
    def temp_market_cap_dir(self):
        """创建临时 market cap 目录"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_market_cap_basic(self, temp_market_cap_dir):
        """基础功能测试"""
        # 创建模拟 market cap 数据
        create_mock_market_cap_data("BTCUSDT", temp_market_cap_dir)

        df = create_mock_data(200)

        result = compute_market_cap_normalized_orderflow_from_df(
            df,
            market_cap_dir=str(temp_market_cap_dir),
            on_missing_market_cap="nan",
        )

        # 检查输出列
        expected_cols = [
            "market_cap_usd",
            "dollar_volume_over_mcap",
            "turnover_over_mcap",
            "net_buy_usd_over_mcap",
            "abs_net_buy_usd_over_mcap",
        ]
        assert all(col in result.columns for col in expected_cols)
        assert len(result) == len(df)

        # 检查数值合理性
        valid_mcap = result["market_cap_usd"].dropna()
        if len(valid_mcap) > 0:
            assert (valid_mcap > 0).all(), "Market cap 应该为正数"

        valid_ratio = result["dollar_volume_over_mcap"].dropna()
        if len(valid_ratio) > 0:
            assert (valid_ratio >= 0).all(), "dollar_volume_over_mcap 应该非负"

    def test_no_future_leak(self, temp_market_cap_dir):
        """
        测试1：无未来函数（修改未来数据不影响历史特征值）⭐⭐⭐⭐⭐
        """
        # 创建模拟 market cap 数据
        create_mock_market_cap_data("BTCUSDT", temp_market_cap_dir)

        df = create_mock_data(300)

        # 计算第一次特征
        result1 = compute_market_cap_normalized_orderflow_from_df(
            df,
            market_cap_dir=str(temp_market_cap_dir),
            on_missing_market_cap="nan",
        )
        mcap_1 = result1["market_cap_usd"].copy()

        # 修改未来数据
        df_future_modified = df.copy()
        if len(df) > 100:
            df_future_modified.loc[df_future_modified.index[100] :, "close"] *= 2.0
            df_future_modified.loc[df_future_modified.index[100] :, "volume"] *= 2.0

            # 重新计算特征
            result2 = compute_market_cap_normalized_orderflow_from_df(
                df_future_modified,
                market_cap_dir=str(temp_market_cap_dir),
                on_missing_market_cap="nan",
            )
            mcap_2 = result2["market_cap_usd"].copy()

            # 检查前50个时间点的特征值（应该不受未来数据影响）
            # 注意：market_cap 是基于日期 join 的，所以只要日期相同，market_cap 值应该相同
            check_idx = df.index[:50]
            mcap_1_check = mcap_1.loc[check_idx].dropna()
            mcap_2_check = mcap_2.loc[check_idx].dropna()

            if len(mcap_1_check) > 0 and len(mcap_2_check) > 0:
                # 找到共同索引
                common_idx = mcap_1_check.index.intersection(mcap_2_check.index)
                if len(common_idx) > 0:
                    diff = (
                        mcap_1_check.loc[common_idx] - mcap_2_check.loc[common_idx]
                    ).abs()
                    max_diff = diff.max()

                    # Market cap 是基于日期 join 的，所以应该完全相同
                    assert (
                        max_diff < 1e-6
                    ), f"未来数据变化不应影响历史 Market Cap 特征值，最大差异: {max_diff}"

    def test_normalization_multi_asset(self, temp_market_cap_dir):
        """
        测试2：多资产归一化（特征分布对齐）⭐⭐⭐⭐

        验证：
        - 不同价格水平和市值的资产，归一化后的特征应该在相似范围内
        - dollar_volume_over_mcap 应该对不同资产的价格水平不敏感
        """
        np.random.seed(42)
        n = 200

        # 创建不同资产的 market cap 数据
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        for sym in symbols:
            create_mock_market_cap_data(sym, temp_market_cap_dir, n_days=100)

        # 不同价格水平的资产
        assets = {
            "BTCUSDT": 50000 + np.cumsum(np.random.randn(n) * 100),
            "ETHUSDT": 3000 + np.cumsum(np.random.randn(n) * 10),
            "SOLUSDT": 100 + np.cumsum(np.random.randn(n) * 0.5),
        }

        results = []
        for symbol, prices in assets.items():
            dates = pd.date_range("2024-01-01", periods=n, freq="4H")
            df = pd.DataFrame(
                {
                    "close": prices,
                    "volume": np.random.uniform(1000, 10000, n),
                    "buy_qty": np.random.uniform(100, 1000, n),
                    "sell_qty": np.random.uniform(100, 1000, n),
                },
                index=dates,
            )
            df["_symbol"] = symbol

            result = compute_market_cap_normalized_orderflow_from_df(
                df,
                market_cap_dir=str(temp_market_cap_dir),
                on_missing_market_cap="nan",
            )
            result["_symbol"] = symbol
            results.append(result)

        combined = pd.concat(results, ignore_index=False)

        # 检查：不同资产的归一化特征应该在相似范围内
        for col in ["dollar_volume_over_mcap", "turnover_over_mcap"]:
            valid_data = combined[col].dropna()
            if len(valid_data) > 0:
                # 检查不同资产的特征分布是否相似
                by_symbol = combined.groupby("_symbol")[col].agg(["mean", "std"])
                # 归一化后的特征，不同资产的均值应该在相似范围内
                mean_range = by_symbol["mean"].max() - by_symbol["mean"].min()
                # 允许一定的差异，因为不同资产的基础特征可能不同
                # 但归一化后应该更接近
                assert mean_range < 1.0, (
                    f"{col} 在不同资产间的均值差异过大: {mean_range:.4f}，"
                    f"可能归一化不正确。各资产均值: {by_symbol['mean'].to_dict()}"
                )

    def test_streaming_vs_batch_consistency(self, temp_market_cap_dir):
        """
        测试3：流式 vs 批量一致性 ⭐⭐⭐⭐
        对生产部署至关重要：生产环境往往是流式推理，而训练是批量计算
        """
        # 创建模拟 market cap 数据
        create_mock_market_cap_data("BTCUSDT", temp_market_cap_dir)

        df = create_mock_data(300)

        # 批量计算（一次性计算所有数据）
        batch_result = compute_market_cap_normalized_orderflow_from_df(
            df,
            market_cap_dir=str(temp_market_cap_dir),
            on_missing_market_cap="nan",
        )

        # 流式计算（分块处理，模拟在线推理）
        chunk_size = 100
        streaming_results = []

        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i : i + chunk_size].copy()
            chunk_result = compute_market_cap_normalized_orderflow_from_df(
                chunk_df,
                market_cap_dir=str(temp_market_cap_dir),
                on_missing_market_cap="nan",
            )
            streaming_results.append(chunk_result)

        if len(streaming_results) > 0:
            streaming_result = pd.concat(streaming_results, axis=0)

            # 比较关键特征
            key_col = "dollar_volume_over_mcap"
            if key_col in batch_result.columns and key_col in streaming_result.columns:
                batch_vals = batch_result[key_col].dropna()
                stream_vals = streaming_result[key_col].dropna()

                # 找到共同索引
                common_idx = batch_vals.index.intersection(stream_vals.index)
                if len(common_idx) > 10:  # 至少需要10个数据点
                    diff = (
                        batch_vals.loc[common_idx] - stream_vals.loc[common_idx]
                    ).abs()
                    max_diff = diff.max()
                    mean_diff = diff.mean()

                    # 允许一定的数值误差（由于分块计算可能导致边界处理略有不同）
                    assert max_diff < 1e-5, (
                        f"流式与批量计算不一致，最大差异: {max_diff:.8f}, "
                        f"平均差异: {mean_diff:.8f}"
                    )

    def test_market_cap_math_correctness(self, temp_market_cap_dir):
        """测试：Market Cap 特征数学正确性"""
        # 创建模拟 market cap 数据
        create_mock_market_cap_data("BTCUSDT", temp_market_cap_dir)

        df = create_mock_data(100)

        result = compute_market_cap_normalized_orderflow_from_df(
            df,
            market_cap_dir=str(temp_market_cap_dir),
            on_missing_market_cap="nan",
        )

        # 手动计算验证
        dollar_volume_manual = df["close"] * df["volume"]
        net_buy_qty_manual = df["buy_qty"] - df["sell_qty"]
        net_buy_usd_manual = net_buy_qty_manual * df["close"]

        # 与特征值比较（允许微小误差）
        mcap = result["market_cap_usd"]
        valid_idx = mcap.dropna().index

        if len(valid_idx) > 0:
            # 验证 dollar_volume_over_mcap
            expected_ratio = (dollar_volume_manual / mcap).loc[valid_idx]
            actual_ratio = result["dollar_volume_over_mcap"].loc[valid_idx]

            diff = (expected_ratio - actual_ratio).abs()
            max_diff = diff.max()

            assert (
                max_diff < 1e-10
            ), f"Market Cap 归一化计算不正确: 最大差异={max_diff:.10f}"

    def test_market_cap_normalization_contract(self):
        """
        Ensure normalization contract captures mixed outputs:
        - market_cap_usd is raw USD scale (NOT cross-asset comparable by itself)
        - *_over_mcap are unitless ratios (cross-asset comparable)
        """
        import yaml

        fd = yaml.safe_load(
            Path(
                "/workspaces/ml_trading_bot/config/feature_dependencies.yaml"
            ).read_text()
        )
        rows = collect_feature_normalization_meta(
            fd, only_features=["market_cap_normalized_orderflow_f"]
        )
        by_col = {r["column"]: r for r in rows}

        assert by_col["market_cap_usd"]["method"] in {"usd", "price_unit", "raw"}
        assert by_col["market_cap_usd"]["cross_asset_comparable"] is False

        for c in [
            "dollar_volume_over_mcap",
            "turnover_over_mcap",
            "net_buy_usd_over_mcap",
            "abs_net_buy_usd_over_mcap",
        ]:
            assert by_col[c]["method"] == "unitless"
            assert by_col[c]["cross_asset_comparable"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
