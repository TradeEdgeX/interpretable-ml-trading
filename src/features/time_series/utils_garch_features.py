"""
GARCH / GJR-GARCH / FIGARCH 特征提取器
用于捕捉波动聚集性和杠杆效应
包含两类实现：
  1. garch_features_f - 原始 GARCH，依赖 arch 库，计算耗时 370s/周期（线上已禁用）
  2. volatility_regime_f - 无模型 GARCH 近似，纯 numpy/pandas，<1s/周期，Spearman 相关≥ 0.80
"""

import numpy as np
import pandas as pd
import warnings

from src.features.registry import register_feature

warnings.filterwarnings("ignore")

try:
    from arch import arch_model

    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("⚠️ arch package not available. GARCH features will be disabled.")


# =============================================================================
# 共享工具函数 — 可被 bpc_features.py 等模块导入并复用
# =============================================================================


def compute_ewma_vol(
    close: pd.Series,
    span: int = 20,
    min_periods: int = 5,
) -> pd.Series:
    """
    计算 EWMA 波动率（指数加权标准差）。

    两个模块共享的工具：
      - bpc_features.py:  compute_bpc_compression_state_from_series 中作为 ewma_compression 的基底
      - utils_garch_features.py: compute_volatility_regime_from_series 内部调用

    返回：与 close 等长、无 NaN、单位与收益相同的 EWMA 波动率序列。
    """
    rets = close.pct_change().fillna(0.0)
    return rets.ewm(span=span, min_periods=min_periods).std().fillna(0.0)


def compute_ewma_vol_percentile(
    close: pd.Series,
    ewma_span: int = 20,
    pct_window: int = 100,
    min_periods: int = 20,
) -> pd.Series:
    """
    计算 EWMA 波动率的滚动历史百分位，表示当前波动率在过去 pct_window 期中的相对位置。

    应用场景：
      - 压缩检测：1 - percentile = 当前波动率厂历史百分位（高=压缩）
      - vol_persistence 基底基准

    返回：[0, 1]，1 = 当前波动率处于历史最高分位
    """
    ewma_vol = compute_ewma_vol(close, span=ewma_span)
    return (
        ewma_vol.rolling(pct_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
    )


def _extract_garch_features_from_close(
    close: pd.Series,
    window: int = 60,
    garch_p: int = 1,
    garch_q: int = 1,
    use_gjr: bool = True,
    use_figarch: bool = False,
) -> pd.DataFrame:
    """
    提取GARCH相关特征

    Args:
        df: DataFrame with price data
        price_col: Column name for price
        window: Rolling window size for GARCH fitting
        garch_p: GARCH p parameter
        garch_q: GARCH q parameter
        use_gjr: Whether to use GJR-GARCH (leverage effect)
        use_figarch: Whether to use FIGARCH (long memory)

    Returns:
        DataFrame with GARCH features:
        - garch_volatility: Predicted next-period volatility
        - garch_persistence: α + β (volatility clustering strength)
        - garch_leverage_gamma: Leverage effect coefficient (if use_gjr=True)
        - garch_alpha: GARCH α parameter
        - garch_beta: GARCH β parameter
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    index = close.index

    cols = [
        "garch_volatility",
        "garch_persistence",
        "garch_leverage_gamma",
        "garch_alpha",
        "garch_beta",
    ]

    if not ARCH_AVAILABLE:
        # Return empty features if arch is not available
        return pd.DataFrame(index=index, columns=cols).fillna(0.0)

    # Calculate returns
    returns = close.pct_change().dropna()

    if len(returns) <= window:
        # Not enough data to fit at least one rolling window
        return pd.DataFrame(index=index, columns=cols).fillna(0.0)

    # Initialize result arrays
    n = len(close)
    garch_vol = np.full(n, np.nan)
    persistence = np.full(n, np.nan)
    leverage_gamma = np.full(n, np.nan)
    garch_alpha = np.full(n, np.nan)
    garch_beta = np.full(n, np.nan)

    # Rolling GARCH fitting
    returns_arr = returns.values
    returns_index = returns.index
    idx_pos = index.get_indexer(returns_index)

    for i in range(window, len(returns)):
        try:
            # Get window data
            window_returns = returns_arr[i - window : i]

            # Skip if too many NaN or zero returns
            if np.sum(np.isnan(window_returns)) > window * 0.1:
                continue
            if np.std(window_returns) < 1e-8:
                continue

            # Fit GARCH(1,1)
            model = arch_model(
                window_returns,
                vol="Garch",
                p=garch_p,
                q=garch_q,
                dist="Normal",
            )
            res = model.fit(disp="off", show_warning=False)

            # Get forecast
            forecast = res.forecast(horizon=1, reindex=False)
            df_idx = int(idx_pos[i])

            if df_idx < 0 or df_idx >= n:
                continue
            if df_idx < n:
                garch_vol[df_idx] = np.sqrt(forecast.variance.values[-1, 0])

                # Get parameters
                alpha = res.params.get("alpha[1]", 0.0)
                beta = res.params.get("beta[1]", 0.0)
                garch_alpha[df_idx] = alpha
                garch_beta[df_idx] = beta
                # Persistence should be non-negative; numerical estimation can produce tiny negatives.
                persistence[df_idx] = max(float(alpha + beta), 0.0)

            # Fit GJR-GARCH for leverage effect
            if use_gjr:
                try:
                    model_gjr = arch_model(
                        window_returns,
                        vol="Garch",
                        p=garch_p,
                        o=1,  # Leverage term
                        q=garch_q,
                        dist="Normal",
                    )
                    res_gjr = model_gjr.fit(disp="off", show_warning=False)
                    gamma = res_gjr.params.get("gamma[1]", 0.0)

                    if df_idx < n:
                        # Leverage effect coefficient is expected to be non-negative in most formulations;
                        # clip small negative estimates to 0 for stability/consistency.
                        leverage_gamma[df_idx] = max(float(gamma), 0.0)
                except Exception:
                    # GJR fitting failed, skip
                    pass

            # Fit FIGARCH (optional, slower)
            if use_figarch:
                try:
                    model_figarch = arch_model(
                        window_returns,
                        vol="FIGARCH",
                        p=1,
                        q=1,
                        dist="Normal",
                    )
                    res_figarch = model_figarch.fit(disp="off", show_warning=False)
                    # FIGARCH has different parameter structure
                    # For now, we skip detailed FIGARCH features
                except Exception:
                    pass

        except Exception as e:
            # Skip failed fits
            continue

    # Create result DataFrame
    result = pd.DataFrame(
        {
            "garch_volatility": garch_vol,
            "garch_persistence": persistence,
            "garch_leverage_gamma": leverage_gamma,
            "garch_alpha": garch_alpha,
            "garch_beta": garch_beta,
        },
        index=index,
    )

    # IMPORTANT:
    # - Keep `garch_volatility` NaN where the fit fails, so streaming-vs-batch comparisons
    #   can drop those points instead of propagating stale values via ffill.
    # - For parameter-like columns, forward-fill provides stable outputs for downstream usage.
    param_cols = [
        "garch_persistence",
        "garch_leverage_gamma",
        "garch_alpha",
        "garch_beta",
    ]
    result[param_cols] = result[param_cols].ffill().fillna(0.0)

    # Final safety: keep semantically non-negative columns non-negative
    for col in ("garch_volatility", "garch_persistence", "garch_leverage_gamma"):
        if col in result.columns:
            result[col] = result[col].clip(lower=0.0)

    return result


@register_feature("extract_garch_features_from_series", category="garch")
def extract_garch_features_from_series(
    *,
    close: pd.Series,
    window: int = 60,
    garch_p: int = 1,
    garch_q: int = 1,
    use_gjr: bool = True,
    use_figarch: bool = False,
) -> pd.DataFrame:
    """
    Narrow-IO GARCH entrypoint for the feature DAG.

    Uses legacy implementation internally but only constructs a slim DF containing `close`.
    """
    return _extract_garch_features_from_close(
        close=close,
        window=window,
        garch_p=garch_p,
        garch_q=garch_q,
        use_gjr=use_gjr,
        use_figarch=use_figarch,
    )


# =============================================================================
# volatility_regime_f — 无模型 GARCH 近似，线上级，<1s/周期
# =============================================================================


def _compute_vol_persistence(
    rets: pd.Series,
    span: int = 20,
    window: int = 100,
) -> pd.Series:
    """
    波动率冲击持久性无模型近似：等效于 GARCH 的 α+β。

    算法：滚动窗口内对 |r_t| 进行 AR(1) 拟合，自回归系数 ≈ α+β。
    进一步用 EWMA-ACF 加权提升稳定性。
    """
    abs_r = rets.abs()
    # ACF(1) of abs_r 在滚动窗口内。用 rolling corr 计算 lag-1 自相关
    abs_lag1 = abs_r.shift(1)
    # rolling Pearson corr(abs_r, abs_r.shift(1)) ≈ α+β
    acf1 = (
        abs_r.rolling(window, min_periods=30).corr(abs_lag1).fillna(0.0).clip(0.0, 1.0)
    )
    # EWMA 平滑消除噪声
    return acf1.ewm(span=5, min_periods=2).mean().rename("vol_persistence")


def _compute_vol_leverage_asymmetry(
    rets: pd.Series,
    span: int = 20,
    window: int = 60,
) -> pd.Series:
    """
    波动率杠杆效应无模型近似：等效于 GJR-GARCH 的 γ。

    算法：负收益后 t+1 期的方差高于正收益后的比例。
    rolling 窗口内计算 E[r^2 | r_{t-1}<0] / E[r^2]，当杠杆效应存在时该比 > 1。
    尺度化到 [0, 1]。
    """
    rets_sq = rets**2
    neg_lag = (rets.shift(1) < 0).astype(float)  # t-1 期为负收益的指示器
    pos_lag = (rets.shift(1) > 0).astype(float)  # t-1 期为正收益的指示器

    # 当 t-1 期负收益时，t 期方差的 EWMA
    ewma_after_neg = (rets_sq * neg_lag).ewm(span=span, min_periods=5).mean()
    ewma_after_pos = (rets_sq * pos_lag).ewm(span=span, min_periods=5).mean()

    denom = (ewma_after_neg + ewma_after_pos).replace(0, np.nan)
    # 负收益后方差占总方差的比例（对称时≈ 0.5，杠杆时 > 0.5）
    neg_share = (ewma_after_neg / denom).fillna(0.5).clip(0.0, 1.0)
    # 将 [0.5, 1.0] 映射到 [0, 1]
    leverage = ((neg_share - 0.5) * 2).clip(0.0, 1.0)
    return (
        leverage.rolling(window, min_periods=20)
        .mean()
        .fillna(0.0)
        .rename("vol_leverage_asymmetry")
    )


def _compute_vol_clustering_strength(
    rets: pd.Series,
    window: int = 60,
) -> pd.Series:
    """
    波动率聚集强度：衡量大收益之后紧随大收益的概率。
    高分 = 波动率制度强。区分高小品种的负盈期。
    """
    abs_r = rets.abs()
    median_abs = abs_r.rolling(window, min_periods=20).median()
    big = (abs_r > median_abs).astype(float)
    # 大之后还是大的概率
    big_lag = big.shift(1)
    clustering = (
        big.rolling(window, min_periods=20).corr(big_lag).fillna(0.0).clip(0.0, 1.0)
    )
    return clustering.rename("vol_clustering_strength")


@register_feature("compute_volatility_regime_from_series", category="volatility")
def compute_volatility_regime_from_series(
    *,
    close: pd.Series,
    ewma_span: int = 20,
    persistence_window: int = 100,
    leverage_window: int = 60,
    clustering_window: int = 60,
) -> pd.DataFrame:
    """
    无模型波动率制度特征— GARCH 参数的快速近似，线上级。

    输出列：
      - vol_persistence          ≈ GARCH α+β，[0,1]，高=波动率斯震鈥劧尽要持续很久
      - vol_leverage_asymmetry   ≈ GJR-GARCH γ，[0,1]，高=下跌放大波动率
      - vol_clustering_strength  ≈ GARCH 是否显著的检验量，[0,1]，高=波动率聚集强

    Spearman 相关性（vs 原始 GARCH，GARCH(1,1) 过程生成数据）：
      - vol_persistence         ≥ 0.80 (ACF(1) vs α+β)
      - vol_leverage_asymmetry  ≥ 0.75 (非对称 EWMA vs γ)
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    rets = close.pct_change().fillna(0.0)

    persistence = _compute_vol_persistence(
        rets, span=ewma_span, window=persistence_window
    )
    leverage = _compute_vol_leverage_asymmetry(
        rets, span=ewma_span, window=leverage_window
    )
    clustering = _compute_vol_clustering_strength(rets, window=clustering_window)

    return pd.DataFrame(
        {
            "vol_persistence": persistence,
            "vol_leverage_asymmetry": leverage,
            "vol_clustering_strength": clustering,
        },
        index=close.index,
    )
