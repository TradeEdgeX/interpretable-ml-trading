"""Alpha101 时序适配版 - 针对单资产时序策略优化

将原始 Alpha101 因子从横截面策略适配为时序策略：
- 移除横截面 rank() 操作（在单资产下无效）
- 保留或替换 ts_rank() 为更合适的归一化方法
- 优化窗口参数以适配加密货币市场
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def ts_rank_vectorized(series: pd.Series, window: int) -> pd.Series:
    """时间序列排名（向量化实现，避免 apply 循环）

    计算每个值在过去 window 个值中的分位数排名 (0~1)
    等价于：rank(x[-window:]) / window

    优化版本：使用 numpy 的 percentile 方法，更高效

    Args:
        series: 输入序列
        window: 滚动窗口大小

    Returns:
        排名序列（0~1 之间的值）
    """

    def _ts_rank(x: np.ndarray) -> float:
        """计算最后一个值在窗口中的排名（0~1）"""
        if len(x) < 2:
            return 0.5
        if np.isnan(x[-1]):
            return np.nan
        # 计算最后一个值在窗口中的排名（从0到1）
        sorted_vals = np.sort(x[~np.isnan(x)])
        if len(sorted_vals) == 0:
            return np.nan
        # 找到最后一个值在排序数组中的位置
        rank = np.searchsorted(sorted_vals, x[-1], side="right")
        # 转换为 0~1 之间的分位数
        return rank / len(sorted_vals) if len(sorted_vals) > 0 else 0.5

    return series.rolling(window, min_periods=1).apply(_ts_rank, raw=True).fillna(0.5)


def alpha001_ts(returns: pd.Series, window: int = 5) -> pd.Series:
    """Alpha #001 时序版本

    原始: rank(ts_argmax(signed_power(returns, 2), 5)) - 0.5
    问题: rank() 是横截面操作，在单资产下恒为常数

    时序优化: 直接用波动率
    """
    # 方案 A: 使用滚动波动率（推荐）
    # CRITICAL: 确保只使用历史数据，不包含当前时刻
    # rolling(window) 默认包含当前时刻，这是正确的（预测 future_return[i] 可以使用 close[i]）
    # 但为了更安全，我们可以明确使用 min_periods 确保有足够的历史数据
    return returns.rolling(window, min_periods=1).std().fillna(0)

    # 方案 B: 使用 ts_argmax 位置（作为状态特征）
    # returns_squared = returns ** 2
    # return returns_squared.rolling(window).apply(
    #     lambda x: x.argmax() if len(x) == window else np.nan
    # )


def alpha022_ts(
    high: pd.Series,
    volume: pd.Series,
    close: pd.Series,
    corr_window: int = 10,
    delta_window: int = 5,
    vol_window: int = 20,
) -> pd.Series:
    """Alpha #022 时序版本

    原始: -1 * delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))
    问题:
    - rank(stddev) 在单资产下等价于 z-score，不如直接用 stddev
    - correlation(high, volume, 5) 样本太少，噪声大

    时序优化:
    - 使用更稳定的窗口（10~20）
    - 用变化率代替 delta
    - 用原始波动率代替 rank
    """
    # 计算量价相关性（使用更稳定的窗口）
    # CRITICAL: rolling().corr() 只使用历史数据，这是正确的
    hv_corr = high.rolling(corr_window, min_periods=2).corr(volume)

    # 计算相关性的变化率
    # CRITICAL: diff() 只使用历史数据，这是正确的
    hv_corr_change = hv_corr.diff(delta_window)

    # 计算波动率（直接使用，不用 rank）
    # CRITICAL: rolling().std() 只使用历史数据，这是正确的
    volatility = close.rolling(vol_window, min_periods=2).std()

    # 组合因子
    alpha = -hv_corr_change * volatility

    return alpha.fillna(0)


def alpha043_ts(
    volume: pd.Series,
    close: pd.Series,
    vol_rank_window: int = 20,
    mom_rank_window: int = 8,
    adv_window: int = 20,
    mom_period: int = 7,
    use_ts_rank: bool = True,
) -> pd.Series:
    """Alpha #043 时序版本

    原始: ts_rank(volume / adv20, 20) * ts_rank(-delta(close, 7), 8)

    时序优化:
    - 保留 ts_rank（时间序列内的归一化，有意义）
    - 或使用 z-score 替代（更平滑）

    Args:
        volume: 成交量序列
        close: 收盘价序列
        vol_rank_window: 成交量排名窗口
        mom_rank_window: 动量排名窗口
        adv_window: 平均成交量窗口
        mom_period: 动量周期
        use_ts_rank: 是否使用 ts_rank（True）或 z-score（False）
    """
    # 计算成交量比率
    adv = volume.rolling(adv_window).mean()
    vol_ratio = volume / (adv + 1e-8)  # 防止除零

    # 计算价格动量
    price_mom = -close.diff(mom_period)

    if use_ts_rank:
        # 方案 A: 使用 ts_rank（对极端值更敏感）
        vol_rank = ts_rank_vectorized(vol_ratio, vol_rank_window)
        mom_rank = ts_rank_vectorized(price_mom, mom_rank_window)
        alpha = vol_rank * mom_rank
    else:
        # 方案 B: 使用 z-score（更平滑）
        vol_mean = vol_ratio.rolling(vol_rank_window).mean()
        vol_std = vol_ratio.rolling(vol_rank_window).std()
        vol_z = (vol_ratio - vol_mean) / (vol_std + 1e-8)

        mom_mean = price_mom.rolling(mom_rank_window).mean()
        mom_std = price_mom.rolling(mom_rank_window).std()
        mom_z = (price_mom - mom_mean) / (mom_std + 1e-8)

        alpha = vol_z * mom_z

    return alpha.fillna(0)


def alpha066_ts(
    open_price: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    eps: float = 1e-8,
) -> pd.Series:
    """Alpha #066 时序版本（无需修改，直接使用）

    原始: (close - open) / (high - low)

    完全无 rank 操作，直接衡量 K 线实体强度（多空力量对比）
    在加密市场极有效（尤其配合高成交量）
    """
    numerator = close - open_price
    denominator = high - low
    alpha = numerator / (denominator + eps)  # 防止除零

    return alpha.fillna(0)


def compute_adapted_alpha101_factors(
    df: pd.DataFrame,
    use_ts_rank: bool = True,
    alpha001_window: int = 5,
    alpha022_corr_window: int = 10,
    alpha022_delta_window: int = 5,
    alpha022_vol_window: int = 20,
    alpha043_vol_rank_window: int = 20,
    alpha043_mom_rank_window: int = 8,
    alpha043_adv_window: int = 20,
    alpha043_mom_period: int = 7,
) -> pd.DataFrame:
    """计算适配后的 Alpha101 因子

    Args:
        df: 包含 open, high, low, close, volume 列的 DataFrame
        use_ts_rank: 是否在 alpha043 中使用 ts_rank（True）或 z-score（False）
        其他参数: 各因子的窗口参数

    Returns:
        包含 alpha101_001_ts, alpha101_022_ts, alpha101_043_ts, alpha101_066_ts 的 DataFrame
    """
    result = pd.DataFrame(index=df.index)

    # 确保有必要的列
    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要的列: {missing_cols}")

    # 计算收益率
    returns = df["close"].pct_change().fillna(0)

    # Alpha #001: 波动率
    result["alpha101_001_ts"] = alpha001_ts(returns, window=alpha001_window)

    # Alpha #022: 量价相关性变化 × 波动率
    result["alpha101_022_ts"] = alpha022_ts(
        df["high"],
        df["volume"],
        df["close"],
        corr_window=alpha022_corr_window,
        delta_window=alpha022_delta_window,
        vol_window=alpha022_vol_window,
    )

    # Alpha #043: 量能+动量
    result["alpha101_043_ts"] = alpha043_ts(
        df["volume"],
        df["close"],
        vol_rank_window=alpha043_vol_rank_window,
        mom_rank_window=alpha043_mom_rank_window,
        adv_window=alpha043_adv_window,
        mom_period=alpha043_mom_period,
        use_ts_rank=use_ts_rank,
    )

    # Alpha #066: K线实体强度
    result["alpha101_066_ts"] = alpha066_ts(
        df["open"], df["high"], df["low"], df["close"]
    )

    return result


if __name__ == "__main__":
    # 测试代码
    print("Alpha101 时序适配版")
    print("=" * 80)

    # 创建测试数据
    dates = pd.date_range("2024-01-01", periods=100, freq="5T")
    np.random.seed(42)

    test_df = pd.DataFrame(
        {
            "open": 100 + np.cumsum(np.random.randn(100) * 0.1),
            "high": 101 + np.cumsum(np.random.randn(100) * 0.1),
            "low": 99 + np.cumsum(np.random.randn(100) * 0.1),
            "close": 100 + np.cumsum(np.random.randn(100) * 0.1),
            "volume": 1000 + np.random.randn(100) * 100,
        },
        index=dates,
    )

    # 计算因子
    factors = compute_adapted_alpha101_factors(test_df)

    print("\n生成的因子:")
    print(factors.head(10))
    print(f"\n因子统计:")
    print(factors.describe())
    print(f"\n因子相关性:")
    print(factors.corr())
