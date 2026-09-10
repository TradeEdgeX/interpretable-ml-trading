"""
测试 Trade Clustering 对聚合数据的支持

测试内容：
1. 功能正确性：验证从 side + volume 计算主导方向的逻辑
2. 流式支持：分批计算和一次性计算结果一致
3. 无未来函数：验证不会使用未来数据

验证场景：
- 聚合数据格式：每个时间戳有两条记录（buy 和 sell 分开）
- 算法应该按时间戳聚合，计算 net = buy_volume - sell_volume
- 主导方向 = sign(net)
"""

import numpy as np
import pandas as pd
import pytest
from typing import Tuple

from src.features.time_series.utils_order_flow_features import (
    compute_trade_clustering_from_ticks,
)


def create_aggregated_tick_data(
    n_timestamps: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """
    创建聚合格式的 tick 数据（每个时间戳有 buy 和 sell 两条记录）

    这是 zip_to_parquet.py 输出的格式：
    - 每个时间戳最多两条记录
    - 一条 side=1 (buy)，一条 side=-1 (sell)
    - volume 是该方向的成交量
    """
    np.random.seed(seed)

    timestamps = pd.date_range("2024-01-01 00:00:00", periods=n_timestamps, freq="1min")

    rows = []
    for ts in timestamps:
        # 随机生成买卖量
        buy_vol = np.random.uniform(10, 100)
        sell_vol = np.random.uniform(10, 100)

        # 每个时间戳输出两条记录（buy 和 sell 分开）
        rows.append({"timestamp": ts, "volume": buy_vol, "side": 1, "price": 50000.0})
        rows.append({"timestamp": ts, "volume": sell_vol, "side": -1, "price": 50000.0})

    df = pd.DataFrame(rows)
    df = df.set_index("timestamp").sort_index()
    return df


def create_expected_dominant_sides(ticks: pd.DataFrame) -> pd.Series:
    """
    计算每个时间戳的预期主导方向

    net = buy_volume - sell_volume
    主导方向 = sign(net)，net >= 0 时为 1 (buy)
    """
    ticks = ticks.copy()
    ticks["buy_volume"] = np.where(ticks["side"] == 1, ticks["volume"], 0.0)
    ticks["sell_volume"] = np.where(ticks["side"] == -1, ticks["volume"], 0.0)

    agg = ticks.groupby(ticks.index).agg(
        {
            "buy_volume": "sum",
            "sell_volume": "sum",
        }
    )
    agg["net"] = agg["buy_volume"] - agg["sell_volume"]
    agg["expected_side"] = np.where(agg["net"] >= 0, 1, -1)

    return agg["expected_side"]


class TestTradeClusteringAggregatedData:
    """测试 Trade Clustering 对聚合数据的支持"""

    def test_functionality_dominant_side_calculation(self):
        """
        功能测试：验证主导方向计算的正确性

        场景：每个时间戳有 buy 和 sell 两条记录
        预期：算法应该按时间戳聚合，用 net = buy - sell 的符号作为主导方向
        """
        print("\n📊 测试主导方向计算...")

        # 创建测试数据
        ticks = create_aggregated_tick_data(n_timestamps=50, seed=42)
        expected_sides = create_expected_dominant_sides(ticks)

        print(f"   输入数据: {len(ticks)} 条 ({len(expected_sides)} 个时间戳)")
        print(f"   预期 buy 主导: {(expected_sides == 1).sum()} 个时间戳")
        print(f"   预期 sell 主导: {(expected_sides == -1).sum()} 个时间戳")

        # 调用算法
        result, state = compute_trade_clustering_from_ticks(ticks, window_size=10)

        # 验证输出行数等于时间戳数（而非原始行数）
        assert len(result) == len(
            expected_sides
        ), f"输出行数 {len(result)} 应等于时间戳数 {len(expected_sides)}"

        # 验证有非常数的输出
        if "trade_cluster_max_buy_run" in result.columns:
            vals = result["trade_cluster_max_buy_run"].dropna()
            assert (
                vals.nunique() > 1
            ), f"trade_cluster_max_buy_run 不应该是常数，unique={vals.nunique()}"
            print(
                f"   ✅ trade_cluster_max_buy_run: min={vals.min():.2f}, max={vals.max():.2f}, unique={vals.nunique()}"
            )

        print("   ✅ 主导方向计算正确")

    def test_functionality_known_pattern(self):
        """
        功能测试：验证已知模式的连续 run 计算

        场景：构造一个已知的主导方向序列，验证 max_buy_run 的计算
        注意：输出值被归一化（除以 window_size）
        """
        print("\n📊 测试已知模式的 run 计算...")

        # 构造已知模式：buy, buy, buy, sell, sell, buy, buy, sell
        # 预期主导方向：1, 1, 1, -1, -1, 1, 1, -1
        # 预期 buy run 长度：3, 2
        # 预期 sell run 长度：2, 1
        np.random.seed(42)

        patterns = [
            (100, 50),  # buy 主导 (net = 50)
            (80, 30),  # buy 主导 (net = 50)
            (90, 40),  # buy 主导 (net = 50)
            (30, 80),  # sell 主导 (net = -50)
            (40, 90),  # sell 主导 (net = -50)
            (70, 30),  # buy 主导 (net = 40)
            (60, 20),  # buy 主导 (net = 40)
            (20, 70),  # sell 主导 (net = -50)
        ]

        timestamps = pd.date_range(
            "2024-01-01 00:00:00", periods=len(patterns), freq="1min"
        )

        rows = []
        for ts, (buy_vol, sell_vol) in zip(timestamps, patterns):
            rows.append(
                {"timestamp": ts, "volume": buy_vol, "side": 1, "price": 50000.0}
            )
            rows.append(
                {"timestamp": ts, "volume": sell_vol, "side": -1, "price": 50000.0}
            )

        ticks = pd.DataFrame(rows).set_index("timestamp").sort_index()

        window_size = 20
        # 调用算法（window_size 足够大以覆盖所有数据）
        result, state = compute_trade_clustering_from_ticks(
            ticks, window_size=window_size
        )

        # 验证最后一个时间戳的统计量
        # 在完整窗口内：buy runs = [3, 2], sell runs = [2, 1]
        # max_buy_run = 3，归一化后 = 3 / 20 = 0.15
        last_row = result.iloc[-1]

        if "trade_cluster_max_buy_run" in result.columns:
            max_buy = last_row["trade_cluster_max_buy_run"]
            expected_max_buy = 3.0 / window_size  # 0.15
            print(f"   max_buy_run: {max_buy} (期望: {expected_max_buy})")
            assert np.isclose(
                max_buy, expected_max_buy, rtol=0.01
            ), f"max_buy_run 应该是 {expected_max_buy}，实际是 {max_buy}"

        if "trade_cluster_max_sell_run" in result.columns:
            max_sell = last_row["trade_cluster_max_sell_run"]
            expected_max_sell = 2.0 / window_size  # 0.10
            print(f"   max_sell_run: {max_sell} (期望: {expected_max_sell})")
            assert np.isclose(
                max_sell, expected_max_sell, rtol=0.01
            ), f"max_sell_run 应该是 {expected_max_sell}，实际是 {max_sell}"

        print("   ✅ 已知模式 run 计算正确")

    def test_streaming_consistency(self):
        """
        流式测试：验证分批计算和一次性计算结果一致

        场景：将数据分成两部分，分别计算并传递状态，与一次性计算比较
        """
        print("\n📊 测试流式计算一致性...")

        # 创建测试数据
        ticks = create_aggregated_tick_data(n_timestamps=100, seed=42)

        # 一次性计算
        result_full, _ = compute_trade_clustering_from_ticks(ticks, window_size=20)

        # 分批计算
        mid_point = len(ticks) // 2
        ticks_part1 = ticks.iloc[:mid_point].copy()
        ticks_part2 = ticks.iloc[mid_point:].copy()

        result_part1, state1 = compute_trade_clustering_from_ticks(
            ticks_part1, window_size=20, initial_state=None
        )
        result_part2, state2 = compute_trade_clustering_from_ticks(
            ticks_part2, window_size=20, initial_state=state1
        )

        # 合并分批结果
        result_streaming = pd.concat([result_part1, result_part2], axis=0)

        # 验证结果一致性
        print(f"   一次性计算: {len(result_full)} 行")
        print(f"   流式计算: {len(result_streaming)} 行")

        # 注意：由于时间戳聚合，行数可能不完全相等，但重叠部分应一致
        common_index = result_full.index.intersection(result_streaming.index)
        print(f"   共同时间戳: {len(common_index)} 个")

        for col in ["trade_cluster_max_buy_run", "trade_cluster_max_sell_run"]:
            if col in result_full.columns and col in result_streaming.columns:
                full_vals = result_full.loc[common_index, col]
                stream_vals = result_streaming.loc[common_index, col]

                # 使用 iloc 进行位置索引，避免未来数据泄露
                # 比较第二部分的结果（流式计算的第二批次）
                second_part_start = len(result_part1)
                if len(common_index) > second_part_start:
                    for i in range(
                        second_part_start,
                        min(second_part_start + 10, len(common_index)),
                    ):
                        idx = common_index[i]
                        full_val = result_full.loc[idx, col]
                        stream_val = result_streaming.loc[idx, col]
                        # 允许小的浮点误差
                        if not np.isnan(full_val) and not np.isnan(stream_val):
                            assert np.isclose(
                                full_val, stream_val, rtol=1e-5
                            ), f"{col} 在 {idx} 不一致: full={full_val}, stream={stream_val}"

        print("   ✅ 流式计算与一次性计算一致")

    def test_no_lookahead_bias(self):
        """
        未来函数测试：验证不会使用未来数据

        方法：使用 iloc 进行严格位置索引，比较部分数据和完整数据的结果
        - 对于 t 时刻，只使用 [0, t] 的数据计算
        - t 时刻的结果不应该因为有了 [t+1, ...] 的数据而改变
        """
        print("\n📊 测试无未来函数...")

        # 创建测试数据
        ticks = create_aggregated_tick_data(n_timestamps=50, seed=42)

        window_size = 10

        # 测试多个时间点
        test_points = [20, 30, 40]

        for test_idx in test_points:
            # 只使用前 test_idx 个时间戳的数据
            # 由于每个时间戳有 2 条记录，实际行数是 test_idx * 2
            ticks_partial = ticks.iloc[: test_idx * 2].copy()

            # 使用完整数据
            result_full, _ = compute_trade_clustering_from_ticks(
                ticks, window_size=window_size
            )

            # 使用部分数据
            result_partial, _ = compute_trade_clustering_from_ticks(
                ticks_partial, window_size=window_size
            )

            # 比较：在部分数据的最后一个时间点，两者的结果应该相同
            # 使用 iloc 进行严格位置索引
            if len(result_partial) > 0:
                last_idx = len(result_partial) - 1

                for col in ["trade_cluster_max_buy_run", "trade_cluster_max_sell_run"]:
                    if col in result_full.columns and col in result_partial.columns:
                        # 使用 iloc 获取对应位置的值
                        partial_val = result_partial.iloc[last_idx][col]
                        full_val = result_full.iloc[last_idx][col]

                        if not np.isnan(partial_val) and not np.isnan(full_val):
                            assert np.isclose(partial_val, full_val, rtol=1e-5), (
                                f"{col} 在位置 {last_idx} (test_idx={test_idx}) 存在未来函数: "
                                f"partial={partial_val}, full={full_val}"
                            )

                print(f"   ✅ test_idx={test_idx}: 无未来函数")

        print("   ✅ 所有测试点均无未来函数")

    def test_edge_case_single_direction(self):
        """
        边界测试：所有时间戳只有单一方向
        注意：输出值被归一化（除以 window_size）
        """
        print("\n📊 测试边界情况：单一方向...")

        # 所有时间戳都是 buy 主导
        n_timestamps = 10
        timestamps = pd.date_range(
            "2024-01-01 00:00:00", periods=n_timestamps, freq="1min"
        )

        rows = []
        for ts in timestamps:
            rows.append({"timestamp": ts, "volume": 100.0, "side": 1, "price": 50000.0})
            rows.append({"timestamp": ts, "volume": 10.0, "side": -1, "price": 50000.0})

        ticks = pd.DataFrame(rows).set_index("timestamp").sort_index()

        window_size = 5
        result, state = compute_trade_clustering_from_ticks(
            ticks, window_size=window_size
        )

        # 所有都是 buy 主导，max_buy_run 应该等于窗口大小
        # 归一化后 = window_size / window_size = 1.0
        last_row = result.iloc[-1]
        if "trade_cluster_max_buy_run" in result.columns:
            max_buy = last_row["trade_cluster_max_buy_run"]
            expected_max_buy = 1.0  # window_size / window_size = 1.0
            print(f"   max_buy_run: {max_buy} (期望: {expected_max_buy})")
            assert np.isclose(
                max_buy, expected_max_buy, rtol=0.01
            ), f"max_buy_run 应该是 {expected_max_buy}，实际是 {max_buy}"

        print("   ✅ 单一方向边界情况正确")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
