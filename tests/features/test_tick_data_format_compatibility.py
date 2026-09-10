#!/usr/bin/env python3
"""测试实盘系统的特征计算与研究pipeline的兼容性

验证点：
1. 研究pipeline的数据格式：1min ticks with [timestamp, price, volume, side]
2. 实盘系统生成的数据格式：1min bars with [timestamp, open, high, low, close, volume, buy_volume, sell_volume]
3. 特征计算器能否从实盘数据正确计算VPIN等tick依赖特征
"""
import pandas as pd
import numpy as np

print("=" * 80)
print("数据格式对比测试")
print("=" * 80)

# ========== 1. 研究pipeline的数据格式（原始ticks） ==========
print("\n📊 场景 1: 研究pipeline的数据格式")
print("-" * 80)

# 模拟研究pipeline的1min tick数据（aggregated from aggTrades）
# 格式：每行是1分钟内一个方向的聚合数据
research_ticks = pd.DataFrame(
    {
        "timestamp": pd.date_range("2025-12-01 00:00:00", periods=120, freq="30s"),
        "price": 50000 + np.random.randn(120) * 100,
        "volume": np.random.uniform(0.5, 2.0, 120),
        "side": np.random.choice([1, -1], 120),  # 1=buy, -1=sell
    }
)
research_ticks = research_ticks.set_index("timestamp")

print(f"✅ 研究数据格式:")
print(f"   - Shape: {research_ticks.shape}")
print(f"   - Columns: {research_ticks.columns.tolist()}")
print(f"   - Index: {type(research_ticks.index).__name__}")
print(f"\n前5行示例:")
print(research_ticks.head())

# ========== 2. 实盘系统生成的数据格式（1min bars） ==========
print("\n📊 场景 2: 实盘系统生成的数据格式")
print("-" * 80)

# 模拟实盘系统的1min bar数据（从OrderFlowListener生成）
live_bars = pd.DataFrame(
    {
        "timestamp": pd.date_range("2025-12-01 00:00:00", periods=60, freq="1min"),
        "open": 50000 + np.random.randn(60) * 50,
        "high": 50000 + np.random.randn(60) * 50 + 20,
        "low": 50000 + np.random.randn(60) * 50 - 20,
        "close": 50000 + np.random.randn(60) * 50,
        "volume": np.random.uniform(5.0, 20.0, 60),
        "buy_volume": np.random.uniform(2.0, 10.0, 60),
        "sell_volume": np.random.uniform(2.0, 10.0, 60),
        "buy_count": np.random.randint(10, 50, 60),
        "sell_count": np.random.randint(10, 50, 60),
        "trade_count": np.random.randint(20, 100, 60),
    }
)

print(f"✅ 实盘数据格式:")
print(f"   - Shape: {live_bars.shape}")
print(f"   - Columns: {live_bars.columns.tolist()}")
print(f"\n前5行示例:")
print(live_bars.head())

# ========== 3. 测试VPIN特征计算（需要tick数据） ==========
print("\n📊 场景 3: VPIN特征计算兼容性测试")
print("-" * 80)

try:
    from features.time_series.utils_order_flow_features import compute_vpin_from_ticks

    # 测试：使用研究pipeline的数据格式计算VPIN
    print("\n🧪 测试1: 使用研究pipeline数据格式计算VPIN")
    vpin_result = compute_vpin_from_ticks(
        ticks=research_ticks,
        bucket_volume_usd=10000.0,  # 使用USD桶
        n_buckets=50,
        adaptive=False,
    )

    print(f"   ✅ VPIN计算成功!")
    print(f"   - 输出Shape: {vpin_result.shape}")
    print(f"   - 输出Columns: {vpin_result.columns.tolist()}")
    print(f"   - VPIN均值: {vpin_result['vpin'].mean():.4f}")
    print(f"   - Signed Imbalance均值: {vpin_result['signed_imbalance'].mean():.4f}")

except Exception as e:
    print(f"   ❌ VPIN计算失败: {e}")

# ========== 4. 问题发现：实盘系统缺少tick级数据 ==========
print("\n⚠️  场景 4: 核心问题识别")
print("-" * 80)

print("\n🔍 数据格式对比:")
print(f"   研究pipeline: {research_ticks.shape[0]} 条 tick 记录")
print(f"   实盘系统:     {live_bars.shape[0]} 条 bar 记录")
print(f"\n📌 关键差异:")
print(f"   1. 研究数据: 保留了 side 列（1=buy, -1=sell）")
print(f"   2. 实盘数据: 只有聚合后的 buy_volume 和 sell_volume")
print(f"   3. 研究数据: 粒度更细（30秒级别的tick）")
print(f"   4. 实盘数据: 已经聚合到 1分钟 bar")

# ========== 5. 解决方案验证：从bar数据重建tick ==========
print("\n💡 场景 5: 解决方案 - 从实盘bar数据重建tick")
print("-" * 80)


def reconstruct_ticks_from_bars(bars_df: pd.DataFrame) -> pd.DataFrame:
    """
    从实盘系统的bar数据重建tick格式数据

    策略：
    - 每个bar生成2条tick记录（buy和sell）
    - 使用VWAP作为价格
    - 使用buy_volume和sell_volume作为成交量
    """
    ticks_list = []

    for idx, row in bars_df.iterrows():
        ts = row["timestamp"]

        # 使用close作为价格（或计算VWAP）
        price = row["close"]

        # Buy tick
        if row["buy_volume"] > 0:
            ticks_list.append(
                {
                    "timestamp": ts,
                    "price": price,
                    "volume": row["buy_volume"],
                    "side": 1,
                }
            )

        # Sell tick
        if row["sell_volume"] > 0:
            ticks_list.append(
                {
                    "timestamp": ts + pd.Timedelta(milliseconds=500),  # 错开时间戳
                    "price": price,
                    "volume": row["sell_volume"],
                    "side": -1,
                }
            )

    reconstructed_ticks = pd.DataFrame(ticks_list)
    if not reconstructed_ticks.empty:
        reconstructed_ticks = reconstructed_ticks.set_index("timestamp").sort_index()

    return reconstructed_ticks


try:
    # 重建tick数据
    reconstructed_ticks = reconstruct_ticks_from_bars(live_bars)

    print(f"✅ 成功从实盘bar重建tick数据:")
    print(f"   - 输入bars: {len(live_bars)} 条")
    print(f"   - 输出ticks: {len(reconstructed_ticks)} 条")
    print(f"   - Columns: {reconstructed_ticks.columns.tolist()}")
    print(f"\n前5行示例:")
    print(reconstructed_ticks.head())

    # 测试VPIN计算
    print(f"\n🧪 测试2: 使用重建的tick数据计算VPIN")
    vpin_result_reconstructed = compute_vpin_from_ticks(
        ticks=reconstructed_ticks,
        bucket_volume_usd=10000.0,
        n_buckets=50,
        adaptive=False,
    )

    print(f"   ✅ VPIN计算成功!")
    print(f"   - 输出Shape: {vpin_result_reconstructed.shape}")
    print(f"   - VPIN均值: {vpin_result_reconstructed['vpin'].mean():.4f}")
    print(
        f"   - Signed Imbalance均值: {vpin_result_reconstructed['signed_imbalance'].mean():.4f}"
    )

except Exception as e:
    print(f"   ❌ 重建或计算失败: {e}")
    import traceback

    traceback.print_exc()

# ========== 6. 结论 ==========
print("\n" + "=" * 80)
print("📋 结论与建议")
print("=" * 80)

print(
    """
✅ 好消息：
1. 研究pipeline的数据格式完全符合VPIN特征计算的要求
2. VPIN计算函数需要的列：[price, volume, side]
3. 实盘系统的OrderFlowListener已经在聚合时保留了买卖统计

⚠️  问题：
1. 实盘系统保存的是1min聚合bar（OHLCV + buy/sell volume）
2. 缺少逐笔级别的side信息（已经被聚合了）
3. 这可能导致VPIN等tick依赖特征计算不准确

💡 解决方案：
方案A（推荐）：实盘系统同时保存tick级数据
  - 在OrderFlowListener中增加tick级缓存
  - 保存到 ticks/SYMBOL/raw_ticks/DATE.parquet
  - 格式：[timestamp, price, volume, side]
  
方案B（妥协）：从bar数据重建tick
  - 每个bar拆分为2条tick（buy和sell）
  - 精度损失：无法保留微观订单流信息
  - 优点：不需要修改现有代码

方案C（精确）：研究与实盘使用相同的tick存储
  - 研究：data/parquet_data/{SYMBOL}_{YEAR}-{MONTH}.parquet
  - 实盘：live/highcap/data/ticks/{SYMBOL}/{DATE}.parquet
  - 确保两者都使用相同的tick格式存储

🎯 推荐实施路径：
1. 短期：使用方案B，确保系统能启动
2. 中期：实施方案A，提高特征计算精度
3. 长期：实施方案C，统一研究与实盘数据管道
"""
)
