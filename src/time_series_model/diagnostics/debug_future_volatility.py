"""
调试未来波动率标签计算问题
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.pipeline.training.label_utils import future_volatility_label
from src.data_tools.data_utils import load_raw_data


def debug_future_volatility():
    """调试未来波动率标签计算"""
    print("=" * 60)
    print("调试未来波动率标签计算")
    print("=" * 60)

    # 加载数据
    print("\n1️⃣ 加载数据...")
    df = load_raw_data(
        data_path=Path("data/parquet_data"),
        symbol="BTCUSDT",
        timeframe="240T",
    )
    print(f"   数据长度: {len(df)}")
    print(f"   价格列: {df['close'].head(10).tolist()}")

    # 计算未来波动率标签
    print("\n2️⃣ 计算未来波动率标签...")
    future_vol = future_volatility_label(df["close"], horizon=10)

    print(f"   标签长度: {len(future_vol)}")
    print(f"   非NaN数量: {future_vol.notna().sum()}")
    print(f"   NaN数量: {future_vol.isna().sum()}")

    if future_vol.notna().sum() > 0:
        print(f"\n   统计信息:")
        print(f"     均值: {future_vol.mean():.8f}")
        print(f"     中位数: {future_vol.median():.8f}")
        print(f"     标准差: {future_vol.std():.8f}")
        print(f"     最小值: {future_vol.min():.8f}")
        print(f"     最大值: {future_vol.max():.8f}")

        print(f"\n   前20个非NaN值:")
        non_nan_values = future_vol.dropna().head(20)
        for idx, val in non_nan_values.items():
            print(f"     {idx}: {val:.8f}")

        # 检查是否有0值
        zero_count = (future_vol == 0.0).sum()
        print(f"\n   零值数量: {zero_count}")
        if zero_count > 0:
            print(f"   ⚠️  发现 {zero_count} 个零值")
            zero_indices = future_vol[future_vol == 0.0].index[:10]
            print(f"   前10个零值索引: {zero_indices.tolist()}")

            # 检查这些索引对应的价格数据
            for idx in zero_indices[:5]:
                idx_pos = df.index.get_loc(idx)
                if idx_pos + 10 < len(df):
                    future_window = df["close"].iloc[idx_pos + 1 : idx_pos + 11]
                    returns = future_window.pct_change().dropna()
                    print(f"\n     索引 {idx} (位置 {idx_pos}):")
                    print(f"       未来10期价格: {future_window.tolist()}")
                    print(f"       未来收益率: {returns.tolist()}")
                    print(f"       收益率平方均值: {np.mean(np.square(returns)):.8f}")
                    print(f"       RMS: {np.sqrt(np.mean(np.square(returns))):.8f}")
    else:
        print("   ⚠️  所有标签都是NaN")

    # 检查收益率计算
    print("\n3️⃣ 检查收益率计算...")
    returns = df["close"].pct_change()
    print(f"   收益率统计:")
    print(f"     非NaN数量: {returns.notna().sum()}")
    print(f"     均值: {returns.mean():.8f}")
    print(f"     标准差: {returns.std():.8f}")
    print(f"     最小值: {returns.min():.8f}")
    print(f"     最大值: {returns.max():.8f}")

    # 手动计算一个窗口的未来波动率
    print("\n4️⃣ 手动计算示例...")
    test_idx = 100
    if test_idx + 10 < len(df):
        test_window = returns.iloc[test_idx + 1 : test_idx + 11]
        manual_vol = np.sqrt(np.mean(np.square(test_window.dropna())))
        print(f"   测试索引: {test_idx}")
        print(f"   未来10期收益率: {test_window.tolist()}")
        print(f"   手动计算的波动率: {manual_vol:.8f}")
        print(f"   函数计算的波动率: {future_vol.iloc[test_idx]:.8f}")
        print(
            f"   是否匹配: {np.isclose(manual_vol, future_vol.iloc[test_idx], rtol=1e-6)}"
        )

    # 检查在模型对比脚本中的使用
    print("\n5️⃣ 检查数据对齐问题...")
    print(f"   df索引类型: {type(df.index)}")
    print(f"   future_vol索引类型: {type(future_vol.index)}")
    print(f"   索引是否匹配: {df.index.equals(future_vol.index)}")

    # 模拟模型对比脚本中的使用
    print("\n6️⃣ 模拟模型对比脚本中的使用...")
    df_test = df.iloc[-1000:].copy()  # 模拟测试集
    future_vol_test = future_volatility_label(df_test["close"], horizon=10)

    print(f"   测试集长度: {len(df_test)}")
    print(f"   测试集未来波动率标签长度: {len(future_vol_test)}")
    print(f"   测试集未来波动率标签非NaN数量: {future_vol_test.notna().sum()}")

    if future_vol_test.notna().sum() > 0:
        print(f"   测试集未来波动率标签均值: {future_vol_test.mean():.8f}")
        print(f"   测试集未来波动率标签中位数: {future_vol_test.median():.8f}")

    # 检查索引对齐
    print(f"\n   测试集索引对齐检查:")
    print(f"     df_test索引类型: {type(df_test.index)}")
    print(f"     future_vol_test索引类型: {type(future_vol_test.index)}")
    print(f"     索引是否匹配: {df_test.index.equals(future_vol_test.index)}")

    # 检查在模型对比脚本中如何添加标签
    print("\n7️⃣ 检查标签添加方式...")
    df_test_with_vol = df_test.copy()
    df_test_with_vol["future_volatility"] = future_vol_test

    print(f"   添加标签后，future_volatility列:")
    print(f"     非NaN数量: {df_test_with_vol['future_volatility'].notna().sum()}")
    if df_test_with_vol["future_volatility"].notna().sum() > 0:
        print(f"     均值: {df_test_with_vol['future_volatility'].mean():.8f}")
        print(f"     中位数: {df_test_with_vol['future_volatility'].median():.8f}")

    # 检查是否有问题：如果标签是在整个数据集上计算的，但只添加到测试集
    print("\n8️⃣ 检查可能的索引对齐问题...")
    # 模拟：在整个数据集上计算，但只添加到测试集
    full_future_vol = future_volatility_label(df["close"], horizon=10)
    df_test_with_full_vol = df_test.copy()
    df_test_with_full_vol["future_volatility"] = full_future_vol.loc[df_test.index]

    print(f"   使用全数据集计算的标签（对齐到测试集）:")
    print(f"     非NaN数量: {df_test_with_full_vol['future_volatility'].notna().sum()}")
    if df_test_with_full_vol["future_volatility"].notna().sum() > 0:
        print(f"     均值: {df_test_with_full_vol['future_volatility'].mean():.8f}")
        print(f"     中位数: {df_test_with_full_vol['future_volatility'].median():.8f}")

    print("\n" + "=" * 60)
    print("调试完成")
    print("=" * 60)


if __name__ == "__main__":
    debug_future_volatility()
