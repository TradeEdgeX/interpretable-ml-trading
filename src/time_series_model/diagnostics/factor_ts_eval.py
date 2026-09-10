#!/usr/bin/env python3
"""Single-factor time-series evaluation helper with IC decay and HTML reports."""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp, skew, kurtosis


def format_pvalue(p_value: float) -> str:
    """
    Format p-value for display with proper handling of very small values.

    Args:
        p_value: P-value to format

    Returns:
        Formatted string:
        - If p_value < 0.0001: returns "< 0.0001" (very significant)
        - If p_value >= 1.0: returns "1.0000" (not significant)
        - Otherwise: returns formatted with 4 decimal places
    """
    if p_value is None or np.isnan(p_value):
        return "N/A"

    p_value = float(p_value)

    # Very small p-values (highly significant)
    if p_value < 0.0001:
        return "< 0.0001"

    # Large p-values (not significant) - cap at 1.0
    if p_value >= 1.0:
        return "1.0000"

    # Normal range: display with 4 decimal places
    return f"{p_value:.4f}"


# Ensure project root on sys.path so we can reuse existing modules
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]  # src/diagnostics -> src -> project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import train_strategy_pipeline as strategy_runner  # noqa: E402
from src.data_tools.data_utils import load_raw_data  # noqa: E402
from src.features.loader.strategy_feature_loader import (
    StrategyFeatureLoader,
)  # noqa: E402
from src.time_series_model.strategy_config import StrategyConfigLoader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate individual factors via the strategy pipeline"
    )
    parser.add_argument("--strategy-config", required=True, help="Path to strategy dir")
    parser.add_argument("--symbol", required=True)
    parser.add_argument(
        "--factors",
        nargs="+",
        default=None,
        help="Factor columns to evaluate. If not specified, will use requested_features from strategy config's features.yaml",
    )
    parser.add_argument("--data-path", default="data/parquet_data")
    parser.add_argument("--timeframe", default="15T")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--quantile", type=float, default=0.2, help="Top/Bottom quantile"
    )
    parser.add_argument(
        "--feature-mode",
        choices=["strategy", "only", "append"],
        default="strategy",
        help="How to handle feature pipeline: use strategy defaults, only requested factors, or append requested factors.",
    )
    # Default output dir is dynamically resolved at runtime:
    # if user keeps the legacy default ("results/factor_ts_eval"), we will redirect to:
    #   results/pools/<strategy_name>/pool_b
    parser.add_argument("--output-dir", default="results/factor_ts_eval")
    parser.add_argument(
        "--ic-decay-lags",
        type=str,
        default="1,3,5,10,20",
        help="Comma-separated forward bars for IC decay analysis (e.g., '1,3,5,10,20')",
    )
    parser.add_argument(
        "--generate-html",
        action="store_true",
        default=True,
        help="Generate HTML report (default: True)",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        default=False,
        help="Automatically open HTML report in browser after generation",
    )
    parser.add_argument(
        "--export-yaml",
        type=str,
        default=None,
        help=(
            "Write the exported features.yaml to PATH (instead of the default output path). "
            "The exported YAML includes BOTH requested_features and feature_pipeline.invert_features "
            "(from strong negative factors), so no separate direction file is needed. "
            "(requires --generate-html so qualified factors are computed)."
        ),
    )
    parser.add_argument(
        "--remove-correlated",
        action="store_true",
        default=False,
        help="Remove highly correlated features (correlation threshold: 0.9)",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.9,
        help="Correlation threshold for removing redundant features (default: 0.9)",
    )
    parser.add_argument(
        "--filter-by-best-lag",
        action="store_true",
        default=False,
        help="Filter features by best lag (only keep features with best lag matching target lag)",
    )
    parser.add_argument(
        "--target-lag",
        type=int,
        default=None,
        help="Target lag for filtering (if not specified, will be inferred from label config max_holding_bars)",
    )
    parser.add_argument(
        "--lag-tolerance",
        type=int,
        default=5,
        help="Tolerance for best lag filtering: keep features if |best_lag - target_lag| <= tolerance (default: 5)",
    )
    return parser.parse_args()


def _default_pool_b_dir(strategy_dir_name: str) -> Path:
    # Convention: Pool B artifacts live under results/pools/<strategy_dir_name>/pool_b
    # We use the strategy DIRECTORY name (config/strategies/<dir_name>) as the stable identifier.
    return Path("results") / "pools" / str(strategy_dir_name) / "pool_b"


def _default_pool_b_yaml_path(strategy_dir_name: str) -> Path:
    # Convention: Pool B YAML is always named features_pool_b.yaml
    return _default_pool_b_dir(strategy_dir_name) / "features_pool_b.yaml"


def _invert_features_from_negative(
    qualified_factors: Dict[str, List[str]],
) -> List[str]:
    neg = (qualified_factors or {}).get("negative") or []
    out: List[str] = []
    for name in neg:
        if isinstance(name, str) and name.strip():
            out.append(name.strip())
    return sorted(set(out))


def _compute_requested_features(
    df_raw: pd.DataFrame,
    feature_loader: StrategyFeatureLoader,
    requested: List[str],
    ensure_signal_cfg,
) -> pd.DataFrame:
    print(f"   🔧 Computing requested features: {requested}")
    print(f"   📊 Input DataFrame columns: {list(df_raw.columns)[:10]}...")

    df_features = feature_loader.load_features_from_requested(
        df_raw,
        requested_features=requested,
        fit=True,
    )

    # Check which columns were actually computed
    computed_cols = [c for c in df_features.columns if c not in df_raw.columns]
    missing_features = [f for f in requested if f not in df_features.columns]

    print(
        f"   ✅ Computed {len(computed_cols)} new columns: {computed_cols[:15]}{'...' if len(computed_cols) > 15 else ''}"
    )

    if missing_features:
        print(f"⚠️  Warning: Some requested factors were not found: {missing_features}")

        # Check each missing feature's output_columns configuration
        features_config = feature_loader.feature_deps.get("features", {})
        for feature_name in missing_features:
            if feature_name in features_config:
                feature_info = features_config[feature_name]
                expected_outputs = feature_info.get("output_columns", [feature_name])
                found_outputs = [
                    col for col in expected_outputs if col in df_features.columns
                ]

                if found_outputs:
                    print(
                        f"   ✅ Feature '{feature_name}' outputs found: {found_outputs} (expected: {expected_outputs})"
                    )
                else:
                    print(
                        f"   ❌ Feature '{feature_name}' not found. Expected outputs: {expected_outputs}"
                    )
                    print(
                        f"      Description: {feature_info.get('description', 'N/A')}"
                    )
                    if feature_info.get("required_columns"):
                        print(
                            f"      Required columns: {feature_info.get('required_columns')}"
                        )
                        missing_req = [
                            c
                            for c in feature_info.get("required_columns", [])
                            if c not in df_raw.columns
                        ]
                        if missing_req:
                            print(f"      ⚠️  Missing required columns: {missing_req}")
                        else:
                            print(f"      ✅ All required columns present")
                    # Check if compute function exists
                    compute_func_name = feature_info.get("compute_func", "")
                    print(f"      Compute function: {compute_func_name}")
            else:
                print(
                    f"   ❌ Feature '{feature_name}' not found in feature dependencies configuration"
                )
                print(
                    f"      Available features: {list(features_config.keys())[:30]}{'...' if len(features_config) > 30 else ''}"
                )

    return strategy_runner.ensure_signal_column(df_features, ensure_signal_cfg)


def prepare_dataset(args: argparse.Namespace, strategy_cfg) -> pd.DataFrame:
    df_raw = load_raw_data(
        data_path=args.data_path,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
    )

    feature_loader = StrategyFeatureLoader()
    feature_mode = args.feature_mode
    extra_factors = args.factors or []

    # Get strategy's requested features if available
    strategy_requested = []
    if hasattr(strategy_cfg.features, "requested_features"):
        strategy_requested = strategy_cfg.features.requested_features or []
    elif isinstance(strategy_cfg.features, dict):
        strategy_requested = strategy_cfg.features.get("requested_features", [])

    # Check if requested factors are in strategy config
    # MACD outputs multiple columns: macd, macd_signal, macd_histogram
    # So if user requests "macd", we need to check if any MACD column exists
    missing_from_strategy = []
    if extra_factors and feature_mode == "strategy":
        strategy_requested_set = set(strategy_requested)

        for factor in extra_factors:
            factor_found = False

            # Direct match
            if factor in strategy_requested_set:
                factor_found = True
            else:
                # Check MACD variants (macd outputs: macd, macd_signal, macd_histogram)
                macd_variants = ["macd", "macd_signal", "macd_histogram", "macd_hist"]
                if factor in macd_variants:
                    # If user requests "macd", check if any MACD column is in strategy
                    if any(
                        variant in strategy_requested_set for variant in macd_variants
                    ):
                        factor_found = True
                    # Or if strategy has "macd" as a feature name (which generates all MACD columns)
                    if "macd" in strategy_requested_set:
                        factor_found = True

            if not factor_found:
                missing_from_strategy.append(factor)

        # Auto-switch to append mode if factors are missing
        if missing_from_strategy:
            print(
                f"⚠️  Warning: Factors {missing_from_strategy} not in strategy config '{strategy_cfg.name}'. Auto-switching to 'append' mode."
            )
            print(f"   Strategy has: {strategy_requested}")
            feature_mode = "append"

    if feature_mode == "strategy":
        df_features = strategy_runner.run_feature_pipeline(
            df_raw,
            feature_loader=feature_loader,
            pipeline_cfg=strategy_cfg.features,
            fit=True,
        )
    elif feature_mode == "only":
        if not extra_factors:
            raise ValueError("--feature-mode=only requires --factors to be specified")
        df_features = _compute_requested_features(
            df_raw,
            feature_loader,
            extra_factors,
            strategy_cfg.features.ensure_signal,
        )
    elif feature_mode == "append":
        if not extra_factors:
            raise ValueError("--feature-mode=append requires --factors to be specified")
        base_features = strategy_runner.run_feature_pipeline(
            df_raw,
            feature_loader=feature_loader,
            pipeline_cfg=strategy_cfg.features,
            fit=True,
        )
        requested_df = _compute_requested_features(
            df_raw,
            feature_loader,
            extra_factors,
            strategy_cfg.features.ensure_signal,
        )

        # Merge requested features into base features
        # Ensure all output columns from requested features are included
        features_config = feature_loader.feature_deps.get("features", {})
        for col in requested_df.columns:
            if col in base_features.columns:
                # Update existing column if it's a requested factor or its output
                if col in extra_factors:
                    base_features[col] = requested_df[col]
            else:
                base_features[col] = requested_df[col]

        # For each requested feature, ensure all its output_columns are included
        for feature_name in extra_factors:
            if feature_name in features_config:
                feature_info = features_config[feature_name]
                output_cols = feature_info.get("output_columns", [feature_name])
                for output_col in output_cols:
                    if (
                        output_col in requested_df.columns
                        and output_col not in base_features.columns
                    ):
                        base_features[output_col] = requested_df[output_col]

        df_features = base_features
    else:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")

    label_func = strategy_runner.import_callable(
        strategy_cfg.labels.generator.module,
        strategy_cfg.labels.generator.function,
    )
    target_col = strategy_cfg.labels.target_column
    df_features[target_col] = label_func(
        df_features.copy(), **strategy_cfg.labels.generator.params
    )

    df_filtered = strategy_runner.apply_filters(
        df_features, strategy_cfg.labels.filters
    )
    # For factor-eval, ignore ensure_feature_non_null filter (too aggressive for research)
    # This allows evaluation of factors even when some features have NaN values
    # The factor-eval will check each factor individually for sufficient samples
    post_label_filters_for_eval = [
        f
        for f in strategy_cfg.labels.post_label_filters
        if not (isinstance(f, dict) and f.get("ensure_feature_non_null"))
        and f != "ensure_feature_non_null"
    ]
    df_filtered = strategy_runner.apply_post_label_filters(
        df_filtered,
        post_label_filters_for_eval,
        list(df_filtered.columns),
    )
    return df_filtered


def compute_ic_series(
    df: pd.DataFrame, factor: str, target_col: str, window: int = 60
) -> pd.Series:
    """Compute rolling IC series."""
    valid = df[[factor, target_col]].dropna()
    if len(valid) < window:
        return pd.Series(dtype=float)

    ic_series = []
    for i in range(window, len(valid) + 1):
        window_data = valid.iloc[i - window : i]
        factor_ranks = window_data[factor].rank(pct=True)
        target_ranks = window_data[target_col].rank(pct=True)
        ic = spearmanr(factor_ranks, target_ranks).correlation
        if not np.isnan(ic):
            ic_series.append(ic)
        else:
            ic_series.append(0.0)

    return pd.Series(ic_series, index=valid.index[window - 1 :])


def compute_ic_decay(
    df: pd.DataFrame,
    factor: str,
    target_col: str,
    decay_lags: List[int],
) -> Dict[str, float]:
    """Compute IC for different forward horizons (decay analysis)."""
    decay_metrics = {}
    base_target = df[target_col].copy()

    for lag in decay_lags:
        # Shift target forward by lag bars
        lag_target = base_target.shift(-lag)
        lag_target.name = f"{target_col}_lag{lag}"
        valid = df[[factor]].join(lag_target).dropna()

        if len(valid) < 50:
            decay_metrics[f"ic_lag_{lag}"] = np.nan
            decay_metrics[f"ic_lag_{lag}_pearson"] = np.nan
            continue

        factor_ranks = valid[factor].rank(pct=True)
        target_ranks = valid[lag_target.name].rank(pct=True)
        rank_ic = spearmanr(factor_ranks, target_ranks).correlation
        pearson_ic = np.corrcoef(valid[factor], valid[lag_target.name])[0, 1]

        # Keep NaN as NaN, don't convert to 0.0, so Best Lag calculation can skip invalid values
        decay_metrics[f"ic_lag_{lag}"] = (
            float(rank_ic) if not np.isnan(rank_ic) else np.nan
        )
        decay_metrics[f"ic_lag_{lag}_pearson"] = (
            float(pearson_ic) if not np.isnan(pearson_ic) else np.nan
        )

    return decay_metrics


def compute_factor_metrics(
    df: pd.DataFrame,
    factor: str,
    target_col: str,
    quantile: float,
    ic_decay_lags: List[int] = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Compute comprehensive factor metrics including IC decay."""
    metrics: Dict[str, float] = {}
    ic_series_df = pd.DataFrame()

    # Check if factor exists
    if factor not in df.columns:
        metrics["error"] = "factor_missing"
        available = sorted(
            [
                c
                for c in df.columns
                if c
                not in [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "timestamp",
                    "_symbol",
                    "signal",
                    "future_return",
                ]
            ]
        )[:30]
        metrics["available_columns"] = available
        print(f"   ❌ Factor '{factor}' not found in DataFrame columns")
        print(f"      Available feature columns: {available[:15]}...")
        return metrics, ic_series_df

    # ------------------------------------------------------------------
    # Strict numeric checks for factor & target
    # - 因子列：如果大部分都能转成数值，则使用转换后的结果；否则直接标记为 non_numeric_factor
    # - 目标列：必须是数值列，否则直接报错 non_numeric_target
    # ------------------------------------------------------------------
    factor_series = df[factor]
    target_series = df[target_col]

    # Factor: allow coercion but要求足够高的非 NaN 比例
    if not pd.api.types.is_numeric_dtype(factor_series):
        factor_numeric = pd.to_numeric(factor_series, errors="coerce")
        non_na_ratio = float(factor_numeric.notna().mean())
        if non_na_ratio < 0.8:
            metrics["error"] = "non_numeric_factor"
            metrics["non_na_ratio"] = non_na_ratio
            sample_values = factor_series.dropna().astype(str).unique().tolist()[:10]
            metrics["sample_values"] = sample_values
            print(
                f"   ❌ Factor '{factor}' is non-numeric (non-NaN ratio after coercion={non_na_ratio:.2f})."
            )
            print(f"      Example raw values: {sample_values}")
            return metrics, ic_series_df
    else:
        factor_numeric = pd.to_numeric(factor_series, errors="coerce")

    # Target: should already be numeric; if不是，直接报错
    if not pd.api.types.is_numeric_dtype(target_series):
        target_numeric = pd.to_numeric(target_series, errors="coerce")
        non_na_ratio_tgt = float(target_numeric.notna().mean())
        if non_na_ratio_tgt < 0.95:
            metrics["error"] = "non_numeric_target"
            metrics["non_na_ratio"] = non_na_ratio_tgt
            sample_values_tgt = (
                target_series.dropna().astype(str).unique().tolist()[:10]
            )
            metrics["sample_values_target"] = sample_values_tgt
            print(
                f"   ❌ Target '{target_col}' is non-numeric (non-NaN ratio after coercion={non_na_ratio_tgt:.2f})."
            )
            print(f"      Example raw target values: {sample_values_tgt}")
            return metrics, ic_series_df
    else:
        target_numeric = pd.to_numeric(target_series, errors="coerce")

    # Build a numeric-only DataFrame for downstream computations
    df_numeric = df.copy()
    df_numeric[factor] = factor_numeric
    df_numeric[target_col] = target_numeric

    valid = df_numeric[[factor, target_col]].dropna()
    if len(valid) < 50:
        metrics["error"] = "insufficient_samples"
        metrics["n_samples"] = int(len(valid))
        # Debug: Add detailed info for first few factors with zero samples
        if len(valid) == 0:
            factor_non_null = factor_numeric.notna().sum()
            target_non_null = target_numeric.notna().sum()
            # Use a module-level counter to limit debug output
            if not hasattr(compute_factor_metrics, "_debug_count"):
                compute_factor_metrics._debug_count = 0
            if compute_factor_metrics._debug_count < 5:
                print(f"   ⚠️  Factor '{factor}': valid samples = 0")
                print(
                    f"      factor_non_null after numeric conversion: {factor_non_null} / {len(df)}"
                )
                print(
                    f"      target_non_null after numeric conversion: {target_non_null} / {len(df)}"
                )
                compute_factor_metrics._debug_count += 1
        return metrics, ic_series_df

    # Basic IC metrics
    factor_ranks = valid[factor].rank(pct=True)
    target_ranks = valid[target_col].rank(pct=True)
    rank_ic = spearmanr(factor_ranks, target_ranks).correlation
    pearson = np.corrcoef(valid[factor], valid[target_col])[0, 1]

    # Rolling IC series for IR calculation
    ic_series = compute_ic_series(df_numeric, factor, target_col, window=60)
    if len(ic_series) > 0:
        metrics["ic_mean"] = float(ic_series.mean())
        metrics["ic_std"] = float(ic_series.std())
        metrics["ic_ir"] = (
            metrics["ic_mean"] / metrics["ic_std"] if metrics["ic_std"] > 0 else 0.0
        )
        metrics["ic_positive_ratio"] = float((ic_series > 0).mean())
        ic_series_df = pd.DataFrame({"ic": ic_series})
    else:
        metrics["ic_mean"] = float(rank_ic) if not np.isnan(rank_ic) else 0.0
        metrics["ic_std"] = 0.0
        metrics["ic_ir"] = 0.0
        metrics["ic_positive_ratio"] = 0.0

    # IC decay analysis
    if ic_decay_lags:
        decay_metrics = compute_ic_decay(df_numeric, factor, target_col, ic_decay_lags)
        metrics.update(decay_metrics)

    # Quantile analysis
    high_cut = valid[factor].quantile(1 - quantile)
    low_cut = valid[factor].quantile(quantile)

    long_mask = valid[factor] >= high_cut
    short_mask = valid[factor] <= low_cut

    long_returns = valid.loc[long_mask, target_col]
    short_returns = valid.loc[short_mask, target_col]

    win_rate_long = float((long_returns > 0).mean()) if len(long_returns) else 0.0
    win_rate_short = float((short_returns < 0).mean()) if len(short_returns) else 0.0

    # Simple long-short backtest
    position = pd.Series(0.0, index=valid.index)
    position[long_mask] = 1.0
    position[short_mask] = -1.0
    strategy_ret = position.shift().fillna(0.0) * valid[target_col]
    equity_curve = strategy_ret.cumsum()
    total_return = float(equity_curve.iloc[-1])
    max_dd = float((equity_curve.cummax() - equity_curve).max())

    # Calculate max drawdown duration
    dd_series = equity_curve.cummax() - equity_curve
    max_dd_duration = 0
    if max_dd > 0:
        in_dd = dd_series > 0
        if in_dd.any():
            dd_periods = (in_dd != in_dd.shift()).cumsum()
            max_dd_duration = (
                int(dd_periods.value_counts().max()) if len(dd_periods) > 0 else 0
            )

    # Risk-adjusted returns
    strategy_ret_clean = strategy_ret[strategy_ret != 0]
    if len(strategy_ret_clean) > 0:
        ret_std = float(strategy_ret_clean.std())
        ret_mean = float(strategy_ret_clean.mean())
        sharpe_ratio = (ret_mean / ret_std * np.sqrt(252)) if ret_std > 1e-8 else 0.0
        calmar_ratio = (total_return / abs(max_dd)) if max_dd < -1e-8 else 0.0
        # Sortino ratio (downside deviation)
        downside_returns = strategy_ret_clean[strategy_ret_clean < 0]
        downside_std = (
            float(downside_returns.std()) if len(downside_returns) > 0 else 0.0
        )
        sortino_ratio = (
            (ret_mean / downside_std * np.sqrt(252)) if downside_std > 1e-8 else 0.0
        )
    else:
        sharpe_ratio = 0.0
        calmar_ratio = 0.0
        sortino_ratio = 0.0

    # Quantile spread (long-short return difference)
    quantile_spread = (
        float(long_returns.mean() - short_returns.mean())
        if len(long_returns) > 0 and len(short_returns) > 0
        else 0.0
    )

    # Statistical significance of IC
    # Note: p-value is NOT related to the sign of IC (positive/negative correlation).
    # It only measures whether IC is significantly different from 0.
    # For negative IC (negative correlation), if it's significant, p-value will still be small (< 0.05).
    if len(ic_series) > 1:
        try:
            ic_t_stat, ic_p_value = ttest_1samp(ic_series, 0)
            # Handle edge cases: NaN, inf, or invalid values
            # Note: if std = 0, t-stat will be inf, but p-value will be 0.0 (valid)
            # We should handle NaN values, but allow inf for t-stat and 0.0 for p-value
            if np.isnan(ic_t_stat) or np.isnan(ic_p_value):
                # If t-stat or p-value is NaN, set defaults
                metrics["ic_t_stat"] = 0.0
                metrics["ic_p_value"] = 1.0
            else:
                metrics["ic_t_stat"] = float(ic_t_stat)
                # Ensure p-value is valid: if p-value is inf, set to 1.0 (shouldn't happen, but be safe)
                if np.isinf(ic_p_value):
                    metrics["ic_p_value"] = 1.0
                else:
                    metrics["ic_p_value"] = float(ic_p_value)
        except Exception as e:
            # If t-test fails for any reason, set defaults
            print(f"   ⚠️  Warning: t-test failed for factor '{factor}': {e}")
            metrics["ic_t_stat"] = 0.0
            metrics["ic_p_value"] = 1.0
    else:
        # Cannot perform statistical test with <= 1 samples
        # Set defaults: ic_t_stat = 0.0 (no significance), p_value = 1.0 (null hypothesis cannot be rejected)
        metrics["ic_t_stat"] = 0.0
        metrics["ic_p_value"] = 1.0

    # Statistical significance of quantile returns
    if len(long_returns) > 1:
        long_t_stat, long_p_value = ttest_1samp(long_returns, 0)
        metrics["long_return_t_stat"] = float(long_t_stat)
        metrics["long_return_p_value"] = float(long_p_value)
    else:
        metrics["long_return_t_stat"] = 0.0
        metrics["long_return_p_value"] = 1.0

    if len(short_returns) > 1:
        short_t_stat, short_p_value = ttest_1samp(short_returns, 0)
        metrics["short_return_t_stat"] = float(short_t_stat)
        metrics["short_return_p_value"] = float(short_p_value)
    else:
        metrics["short_return_t_stat"] = 0.0
        metrics["short_return_p_value"] = 1.0

    # Quantile return statistics
    if len(long_returns) > 0:
        metrics["long_return_std"] = float(long_returns.std())
        metrics["long_return_skew"] = (
            float(skew(long_returns)) if len(long_returns) > 2 else 0.0
        )
        metrics["long_return_kurtosis"] = (
            float(kurtosis(long_returns)) if len(long_returns) > 3 else 0.0
        )
    else:
        metrics["long_return_std"] = 0.0
        metrics["long_return_skew"] = 0.0
        metrics["long_return_kurtosis"] = 0.0

    if len(short_returns) > 0:
        metrics["short_return_std"] = float(short_returns.std())
        metrics["short_return_skew"] = (
            float(skew(short_returns)) if len(short_returns) > 2 else 0.0
        )
        metrics["short_return_kurtosis"] = (
            float(kurtosis(short_returns)) if len(short_returns) > 3 else 0.0
        )
    else:
        metrics["short_return_std"] = 0.0
        metrics["short_return_skew"] = 0.0
        metrics["short_return_kurtosis"] = 0.0

    # Factor distribution stats
    factor_stats = valid[factor].describe()
    factor_skewness = float(skew(valid[factor])) if len(valid) > 2 else 0.0
    factor_kurt = float(kurtosis(valid[factor])) if len(valid) > 3 else 0.0

    # Factor autocorrelation (stability measure)
    factor_autocorr = float(valid[factor].autocorr(lag=1)) if len(valid) > 1 else 0.0

    # Turnover calculation (position change rate)
    position_changes = (
        (position.diff().abs().sum() / len(position)) if len(position) > 1 else 0.0
    )
    turnover = float(position_changes)

    # Factor-target correlation distribution stats
    target_stats = valid[target_col].describe()

    metrics.update(
        {
            "n_samples": int(len(valid)),
            "rank_ic": float(rank_ic) if not np.isnan(rank_ic) else 0.0,
            "pearson": float(pearson) if not np.isnan(pearson) else 0.0,
            "win_rate_long": win_rate_long,
            "win_rate_short": win_rate_short,
            "avg_return_long": float(long_returns.mean()) if len(long_returns) else 0.0,
            "avg_return_short": (
                float(short_returns.mean()) if len(short_returns) else 0.0
            ),
            "quantile_spread": quantile_spread,
            "total_return": total_return,
            "max_drawdown": max_dd,
            "max_drawdown_duration": max_dd_duration,
            "sharpe_ratio": sharpe_ratio,
            "calmar_ratio": calmar_ratio,
            "sortino_ratio": sortino_ratio,
            "turnover": turnover,
            "factor_mean": float(factor_stats["mean"]),
            "factor_std": float(factor_stats["std"]),
            "factor_min": float(factor_stats["min"]),
            "factor_max": float(factor_stats["max"]),
            "factor_skewness": factor_skewness,
            "factor_kurtosis": factor_kurt,
            "factor_autocorr": factor_autocorr,
            "target_mean": float(target_stats["mean"]),
            "target_std": float(target_stats["std"]),
        }
    )
    return metrics, ic_series_df


def generate_html_report(
    results: Dict[str, Dict[str, float]],
    ic_series_data: Dict[str, pd.DataFrame],
    strategy_name: str,
    symbol: str,
    output_dir: Path,
    ic_decay_lags: List[int],
    diagnostics: Dict[str, Any] | None = None,
) -> Path:
    """Generate comprehensive HTML report."""
    html_path = output_dir / f"ts_eval_{strategy_name}_{symbol}.html"

    diagnostics = diagnostics or {}
    factor_resolution = diagnostics.get("factor_resolution", {})
    unknown_factors = factor_resolution.get("unknown_factors", [])
    missing_feature_outputs = factor_resolution.get("missing_feature_outputs", [])
    factor_mappings = factor_resolution.get("mappings", {})
    error_factors = diagnostics.get("error_factors", {})
    # Get lag_info for best lag display
    lag_filtering_info = diagnostics.get("best_lag_filtering", {})
    lag_info = lag_filtering_info.get("lag_info", {})

    # Build summary table
    summary_rows = []
    for factor, metrics in results.items():
        if "error" in metrics:
            summary_rows.append(
                f"""
                <tr>
                    <td><strong>{factor}</strong></td>
                    <td colspan="12" style="color: red;">Error: {metrics['error']}</td>
                </tr>
                """
            )
            continue

        # Find best lag (highest IC value, not absolute value)
        # For positive IC: higher is better
        # For negative IC: less negative (closer to 0) is better, but we want the highest IC value
        best_lag = None
        best_ic = None
        for lag in ic_decay_lags:
            ic_val = metrics.get(f"ic_lag_{lag}", None)
            # Skip None, NaN
            if ic_val is None:
                continue
            # Check for NaN (can be float('nan') or np.nan)
            if isinstance(ic_val, float) and np.isnan(ic_val):
                continue
            # Convert to float and check if valid
            try:
                ic_val_float = float(ic_val)
                if np.isnan(ic_val_float):
                    continue
            except (ValueError, TypeError):
                continue

            # Update best if this IC value is higher (not absolute value)
            # This means: for positive IC, higher is better; for negative IC, less negative is better
            if best_ic is None or ic_val_float > best_ic:
                best_ic = ic_val_float
                best_lag = lag

        # Fallback: if no valid IC found in decay lags, use rank_ic as best_ic
        if best_lag is None:
            rank_ic = metrics.get("rank_ic", 0.0)
            if rank_ic is not None:
                try:
                    rank_ic_float = float(rank_ic)
                    if not np.isnan(rank_ic_float) and rank_ic_float != 0.0:
                        best_ic = rank_ic_float
                        best_lag = 0  # Use 0 to indicate current period
                except (ValueError, TypeError):
                    pass

        # Color coding for key metrics
        ic_ir = metrics.get("ic_ir", 0.0)
        ic_ir_class = "good" if ic_ir > 0.5 else ("warning" if ic_ir > 0 else "bad")

        sharpe = metrics.get("sharpe_ratio", 0.0)
        sharpe_class = "good" if sharpe > 1.0 else ("warning" if sharpe > 0 else "bad")

        quantile_spread = metrics.get("quantile_spread", 0.0)
        spread_class = "good" if quantile_spread > 0 else "bad"

        best_lag_display = f"{best_lag}" if best_lag is not None else "N/A"
        best_ic_display = f"{best_ic:.4f}" if best_ic is not None else "N/A"
        best_ic_class = (
            "good"
            if (best_ic is not None and best_ic > 0.05)
            else ("warning" if (best_ic is not None and best_ic > 0) else "bad")
        )

        summary_rows.append(
            f"""
            <tr>
                <td><strong>{factor}</strong></td>
                <td>{metrics.get('n_samples', 0)}</td>
                <td>{metrics.get('rank_ic', 0.0):.4f}</td>
                <td>{metrics.get('pearson', 0.0):.4f}</td>
                <td>{metrics.get('ic_mean', 0.0):.4f}</td>
                <td>{metrics.get('ic_std', 0.0):.4f}</td>
                <td class="{ic_ir_class}">{metrics.get('ic_ir', 0.0):.4f}</td>
                <td>{metrics.get('ic_positive_ratio', 0.0):.2%}</td>
                <td>{metrics.get('ic_t_stat', 0.0):.2f}</td>
                <td>{format_pvalue(metrics.get('ic_p_value', 1.0))}</td>
                <td class="{best_ic_class}"><strong>{best_lag_display}</strong></td>
                <td class="{best_ic_class}">{best_ic_display}</td>
                <td>{metrics.get('quantile_spread', 0.0):.4f}</td>
                <td>{metrics.get('win_rate_long', 0.0):.2%}</td>
                <td>{metrics.get('win_rate_short', 0.0):.2%}</td>
                <td>{metrics.get('avg_return_long', 0.0):.4f}</td>
                <td>{metrics.get('avg_return_short', 0.0):.4f}</td>
                <td>{metrics.get('long_return_t_stat', 0.0):.2f}</td>
                <td>{metrics.get('short_return_t_stat', 0.0):.2f}</td>
                <td class="{sharpe_class}">{metrics.get('sharpe_ratio', 0.0):.2f}</td>
                <td>{metrics.get('calmar_ratio', 0.0):.2f}</td>
                <td>{metrics.get('sortino_ratio', 0.0):.2f}</td>
                <td>{metrics.get('total_return', 0.0):.2f}</td>
                <td>{metrics.get('max_drawdown', 0.0):.2f}</td>
                <td>{metrics.get('max_drawdown_duration', 0)}</td>
                <td>{metrics.get('turnover', 0.0):.4f}</td>
                <td>{metrics.get('factor_autocorr', 0.0):.4f}</td>
            </tr>
            """
        )

    # Create detailed metrics section with IC decay charts for each factor
    detailed_sections = []
    chart_scripts = []

    for factor_idx, (factor, metrics) in enumerate(results.items()):
        if "error" in metrics:
            continue

        # Extract IC decay data
        ic_decay_values = []
        ic_decay_pearson = []
        best_lag = None
        best_ic = None

        for lag in ic_decay_lags:
            ic_val = metrics.get(f"ic_lag_{lag}", None)
            ic_pearson = metrics.get(f"ic_lag_{lag}_pearson", None)

            # For chart display, use 0.0 for NaN (Chart.js can handle it)
            ic_val_for_chart = 0.0
            ic_pearson_for_chart = 0.0
            if ic_val is not None:
                try:
                    ic_val_float = float(ic_val)
                    if not np.isnan(ic_val_float):
                        ic_val_for_chart = ic_val_float
                except (ValueError, TypeError):
                    pass
            if ic_pearson is not None:
                try:
                    ic_pearson_float = float(ic_pearson)
                    if not np.isnan(ic_pearson_float):
                        ic_pearson_for_chart = ic_pearson_float
                except (ValueError, TypeError):
                    pass

            ic_decay_values.append(ic_val_for_chart)
            ic_decay_pearson.append(ic_pearson_for_chart)

            # Find best lag (highest IC value, not absolute value) - skip NaN values
            if ic_val is not None:
                try:
                    ic_val_float = float(ic_val)
                    if not np.isnan(ic_val_float):
                        # Update best if this IC value is higher (not absolute value)
                        if best_ic is None or ic_val_float > best_ic:
                            best_ic = ic_val_float
                            best_lag = lag
                except (ValueError, TypeError):
                    pass

        # Fallback: if no valid IC found in decay lags, use rank_ic
        if best_lag is None:
            rank_ic = metrics.get("rank_ic", 0.0)
            if rank_ic is not None:
                try:
                    rank_ic_float = float(rank_ic)
                    if not np.isnan(rank_ic_float) and rank_ic_float != 0.0:
                        best_ic = rank_ic_float
                        best_lag = 0
                except (ValueError, TypeError):
                    pass

        best_lag_display = f"{best_lag}" if best_lag is not None else "N/A"
        best_ic_display = f"{best_ic:.4f}" if best_ic is not None else "N/A"

        # Chart ID for this factor
        chart_id = f"icDecayChart_{factor_idx}"

        # Create chart HTML
        chart_html = f"""
        <div class="chart-container">
            <canvas id="{chart_id}"></canvas>
        </div>
        """

        # Create chart script
        chart_script = f"""
        // Chart for {factor}
        const ctx_{factor_idx} = document.getElementById('{chart_id}');
        if (ctx_{factor_idx}) {{
            new Chart(ctx_{factor_idx}, {{
                type: 'line',
                data: {{
                    labels: {ic_decay_lags},
                    datasets: [{{
                        label: 'Rank IC (Spearman)',
                        data: {ic_decay_values},
                        borderColor: 'rgb(33, 150, 243)',
                        backgroundColor: 'rgba(33, 150, 243, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    }}, {{
                        label: 'Pearson IC',
                        data: {ic_decay_pearson},
                        borderColor: 'rgb(76, 175, 80)',
                        backgroundColor: 'rgba(76, 175, 80, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        borderDash: [5, 5]
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'IC Decay Analysis - Best Lag: {best_lag_display} (IC={best_ic_display})',
                            font: {{
                                size: 16,
                                weight: 'bold'
                            }}
                        }},
                        legend: {{
                            display: true,
                            position: 'top'
                        }},
                        tooltip: {{
                            mode: 'index',
                            intersect: false
                        }}
                    }},
                    scales: {{
                        x: {{
                            title: {{
                                display: true,
                                text: 'Forward Bars (Lag)'
                            }},
                            grid: {{
                                color: 'rgba(0, 0, 0, 0.05)'
                            }}
                        }},
                        y: {{
                            title: {{
                                display: true,
                                text: 'IC Value'
                            }},
                            grid: {{
                                color: 'rgba(0, 0, 0, 0.05)'
                            }},
                            zeroLine: {{
                                color: 'rgba(0, 0, 0, 0.2)',
                                width: 2
                            }}
                        }}
                    }},
                    interaction: {{
                        mode: 'nearest',
                        axis: 'x',
                        intersect: false
                    }}
                }}
            }});
        }}
        """
        chart_scripts.append(chart_script)

        detailed_html = f"""
        <div class="factor-detail">
            <h3>📊 {factor} - Detailed Metrics</h3>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">IC Statistics</div>
                    <div class="metric-value">{metrics.get('rank_ic', 0.0):.4f}</div>
                    <div>Rank IC | IR: {metrics.get('ic_ir', 0.0):.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Best Lag</div>
                    <div class="metric-value" style="color: {'#4CAF50' if (best_ic is not None and best_ic > 0) else '#f44336'}">{best_lag if best_lag else 'N/A'}</div>
                    <div>IC: {best_ic_display} | Bars forward</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Risk-Adjusted Returns</div>
                    <div class="metric-value">{metrics.get('sharpe_ratio', 0.0):.2f}</div>
                    <div>Sharpe | Calmar: {metrics.get('calmar_ratio', 0.0):.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Quantile Spread</div>
                    <div class="metric-value">{metrics.get('quantile_spread', 0.0):.4f}</div>
                    <div>Long: {metrics.get('avg_return_long', 0.0):.4f} | Short: {metrics.get('avg_return_short', 0.0):.4f}</div>
                </div>
            </div>

            <h4>📉 IC Decay Analysis</h4>
            {chart_html}
        </div>
        """
        detailed_sections.append(detailed_html)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Time-Series Factor Evaluation: {symbol}</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 95%;
                width: 100%;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #555;
                margin-top: 30px;
                border-left: 4px solid #2196F3;
                padding-left: 10px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 13px;
                table-layout: auto;
            }}
            th, td {{
                white-space: nowrap;
                padding: 8px 10px;
            }}
            th {{
                background-color: #2196F3;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: 600;
            }}
            td {{
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .metric-card {{
                display: inline-block;
                background: #f8f9fa;
                border-left: 4px solid #4CAF50;
                padding: 15px;
                margin: 10px;
                border-radius: 4px;
                min-width: 200px;
            }}
            .metric-label {{
                font-size: 12px;
                color: #666;
                text-transform: uppercase;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #333;
            }}
            .good {{ color: #4CAF50; }}
            .bad {{ color: #f44336; }}
            .warning {{ color: #ff9800; }}
            .info-box {{
                background: #e3f2fd;
                border-left: 4px solid #2196F3;
                padding: 15px;
                margin: 20px 0;
                border-radius: 4px;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .factor-detail {{
                margin: 30px 0;
                padding: 20px;
                background: #f9f9f9;
                border-radius: 8px;
                border-left: 4px solid #4CAF50;
            }}
            .chart-container {{
                position: relative;
                height: 400px;
                width: 100%;
                margin: 20px 0;
                background: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h4 {{
                color: #555;
                margin-top: 20px;
                margin-bottom: 15px;
                font-size: 18px;
            }}
            .footer {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #666;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Time-Series Factor Evaluation Report</h1>

            <div class="info-box">
                <strong>Strategy:</strong> {strategy_name}<br>
                <strong>Symbol:</strong> {symbol}<br>
                <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                <strong>Factors Evaluated:</strong> {len(results)}
            </div>

            <h2>🧪 Diagnostics</h2>
            <div class="info-box">
                <p>Automatic checks from this run. Use them to identify bad factors, config issues, or data problems.</p>
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Non-numeric / invalid factors</div>
                    <div class="metric-value {'bad' if error_factors else 'good'}">{len(error_factors)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Unknown factors in config</div>
                    <div class="metric-value {'bad' if unknown_factors else 'good'}">{len(unknown_factors)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Features with missing output columns</div>
                    <div class="metric-value {'warning' if missing_feature_outputs else 'good'}">{len(missing_feature_outputs)}</div>
                </div>
            </div>

            <h3>Factor Resolution (Config → DataFrame Columns)</h3>
            <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>DataFrame Column</th>
                        <th>Requested As</th>
                    </tr>
                </thead>
                <tbody>
    """

    # Render factor mappings (limit number of rows for readability)
    mapping_items = list(sorted(factor_mappings.items()))
    for col, sources in mapping_items[:200]:
        html_content += f"""
                    <tr>
                        <td><code>{col}</code></td>
                        <td>{", ".join(sorted(sources))}</td>
                    </tr>
        """

    if len(mapping_items) > 200:
        html_content += f"""
                    <tr>
                        <td colspan="2" style="color: #999;">... {len(mapping_items) - 200} more mappings not shown ...</td>
                    </tr>
        """

    html_content += """
                </tbody>
            </table>
            </div>

            <h3>Unknown Factors in Config</h3>
    """

    if unknown_factors:
        html_content += "<ul>"
        for f in unknown_factors:
            html_content += f"<li><code>{f}</code></li>"
        html_content += "</ul>"
    else:
        html_content += "<p>No unknown factors. All requested factors were resolved to DataFrame columns or features.</p>"

    # Missing feature outputs
    html_content += """

            <h3>Features With Missing Output Columns</h3>
    """
    if missing_feature_outputs:
        html_content += "<ul>"
        for item in missing_feature_outputs:
            feature = item.get("feature")
            expected = item.get("expected", [])
            html_content += f"<li><code>{feature}</code> &mdash; expected columns not found in DataFrame: {', '.join(expected)}</li>"
        html_content += "</ul>"
    else:
        html_content += "<p>All requested features that were recognized had at least one output column present in the DataFrame.</p>"

    # Correlation removal info
    correlation_removal = (
        diagnostics.get("correlation_removal") if diagnostics else None
    )
    if correlation_removal:
        html_content += """
            <h3>Correlation-Based Feature Removal</h3>
            <p>Removed highly correlated features to reduce redundancy.</p>
            <ul>
                <li><strong>Threshold:</strong> {threshold}</li>
                <li><strong>Kept:</strong> {kept_count} features</li>
                <li><strong>Removed:</strong> {removed_count} features</li>
            </ul>
        """.format(
            threshold=correlation_removal["threshold"],
            kept_count=len(correlation_removal["kept"]),
            removed_count=len(correlation_removal["removed"]),
        )
        if correlation_removal.get("removed"):
            html_content += "<h4>Removed Features:</h4><ul>"
            removal_reasons = correlation_removal.get("removal_reasons", {})
            for feat in correlation_removal["removed"][:20]:  # Limit to 20
                reason = removal_reasons.get(
                    feat, "Highly correlated with other features"
                )
                html_content += f"<li><code>{feat}</code> &mdash; {reason}</li>"
            html_content += "</ul>"
            if len(correlation_removal["removed"]) > 20:
                html_content += (
                    f"<p>... and {len(correlation_removal['removed']) - 20} more</p>"
                )

    # Best lag filtering info
    best_lag_filtering = diagnostics.get("best_lag_filtering") if diagnostics else None
    if best_lag_filtering:
        html_content += """
            <h3>Best Lag Filtering</h3>
            <p>Filtered features based on their best lag (IC peak horizon).</p>
            <ul>
                <li><strong>Target Lag:</strong> {target_lag} bars</li>
                <li><strong>Tolerance:</strong> ±{tolerance} bars</li>
                <li><strong>Kept:</strong> {kept_count} features</li>
                <li><strong>Removed:</strong> {removed_count} features</li>
            </ul>
        """.format(
            target_lag=best_lag_filtering["target_lag"],
            tolerance=best_lag_filtering["tolerance"],
            kept_count=len(best_lag_filtering["kept"]),
            removed_count=len(best_lag_filtering["removed"]),
        )
        if best_lag_filtering.get("removed"):
            html_content += "<h4>Removed Features:</h4><ul>"
            lag_info = best_lag_filtering.get("lag_info", {})
            for feat in best_lag_filtering["removed"][:20]:  # Limit to 20
                info = lag_info.get(feat, {})
                best_lag = info.get("best_lag", "N/A")
                best_ic = info.get("best_ic", "N/A")
                if isinstance(best_ic, (int, float)):
                    best_ic_str = f"{best_ic:.4f}"
                else:
                    best_ic_str = str(best_ic)
                if (
                    isinstance(best_lag, (int, float))
                    and best_lag_filtering["target_lag"] is not None
                ):
                    lag_diff = abs(best_lag - best_lag_filtering["target_lag"])
                else:
                    lag_diff = "N/A"
                html_content += f"<li><code>{feat}</code> &mdash; best_lag={best_lag}, best_ic={best_ic_str}, diff={lag_diff}</li>"
            html_content += "</ul>"
            if len(best_lag_filtering["removed"]) > 20:
                html_content += (
                    f"<p>... and {len(best_lag_filtering['removed']) - 20} more</p>"
                )

    # Error factors (non-numeric, insufficient samples, etc.)
    html_content += """

            <h3>Problematic Factors (Errors During Evaluation)</h3>
    """

    if error_factors:
        html_content += """
            <table>
                <thead>
                    <tr>
                        <th>Factor</th>
                        <th>Error Type</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
        """
        for factor, m in error_factors.items():
            error_type = m.get("error", "unknown")
            details_parts = []
            if error_type == "non_numeric_factor":
                nnr = m.get("non_na_ratio")
                if nnr is not None:
                    details_parts.append(f"non-NaN ratio after coercion = {nnr:.2f}")
                samples = m.get("sample_values") or []
                if samples:
                    details_parts.append(
                        f"sample raw values: {', '.join(map(str, samples[:10]))}"
                    )
            elif error_type == "non_numeric_target":
                nnr = m.get("non_na_ratio")
                if nnr is not None:
                    details_parts.append(
                        f"target non-NaN ratio after coercion = {nnr:.2f}"
                    )
                samples = m.get("sample_values_target") or []
                if samples:
                    details_parts.append(
                        f"sample raw target values: {', '.join(map(str, samples[:10]))}"
                    )
            elif error_type == "factor_missing":
                avail = m.get("available_columns") or []
                if avail:
                    details_parts.append(
                        f"first available feature columns: {', '.join(map(str, avail[:15]))}"
                    )
            elif error_type == "insufficient_samples":
                n = m.get("n_samples")
                if n is not None:
                    details_parts.append(f"valid samples after cleaning: {n}")

            details = "; ".join(details_parts) if details_parts else ""
            html_content += f"""
                    <tr>
                        <td><code>{factor}</code></td>
                        <td>{error_type}</td>
                        <td>{details}</td>
                    </tr>
            """

        html_content += """
                </tbody>
            </table>
        """
    else:
        html_content += "<p>No evaluation errors. All evaluated factors passed basic numeric and sample-size checks.</p>"

    # ------------------------------------------------------------------
    # Auto summary of top factors (based on IC IR + Sharpe)
    # ------------------------------------------------------------------

    # Build a list of factors with valid metrics (no error)
    ranked_factors = []
    for factor, m in results.items():
        if "error" in m:
            continue
        ic_ir_val = float(m.get("ic_ir", 0.0) or 0.0)
        sharpe_val = float(m.get("sharpe_ratio", 0.0) or 0.0)
        ic_mean_val = float(m.get("ic_mean", 0.0) or 0.0)
        ic_t_stat = float(m.get("ic_t_stat", 0.0) or 0.0)
        ic_p_value = float(m.get("ic_p_value", 1.0) or 1.0)
        # Get best lag from lag_info if available
        best_lag_info = lag_info.get(factor, {})
        best_lag_val = best_lag_info.get("best_lag")

        ranked_factors.append(
            {
                "name": factor,
                "ic_ir": ic_ir_val,
                "sharpe": sharpe_val,
                "ic_mean": ic_mean_val,
                "ic_t_stat": ic_t_stat,
                "ic_p_value": ic_p_value,
                "best_lag": best_lag_val,  # Add best_lag to factor dict
            }
        )

    # Positive alpha candidates: IC Mean > 0, IC t-stat > 1.96 (statistically significant)
    # Optional: IC IR > 0 or Sharpe > 0 (loose constraint to keep more candidates)
    # Note: p-value should be < 0.05 if ic_t_stat > 1.96 (they are consistent)
    # If p-value = 1.0, it means ic_series length <= 1, which is inconsistent with ic_t_stat > 1.96
    top_positive = sorted(
        [
            f
            for f in ranked_factors
            if f["ic_mean"] > 0
            and f["ic_t_stat"] > 1.96
            and (f["ic_ir"] > 0 or f["sharpe"] > 0)
            # Additional check: if ic_t_stat > 1.96, p-value should be < 0.05 (consistent)
            # Exclude factors with p-value = 1.0 as they indicate data inconsistency
            and f["ic_p_value"] < 1.0
        ],
        key=lambda x: (x["ic_ir"], x["sharpe"]),
        reverse=True,
    )
    # No truncation - show all qualified factors

    # Strong negative factors (potentially useful when reversed):
    # IC Mean < 0, IC t-stat < -1.96 (statistically significant)
    # Optional: IC IR < 0 or Sharpe < 0
    # Note: p-value should be < 0.05 if ic_t_stat < -1.96 (they are consistent)
    # If p-value = 1.0, it means ic_series length <= 1, which is inconsistent with ic_t_stat < -1.96
    top_negative = sorted(
        [
            f
            for f in ranked_factors
            if f["ic_mean"] < 0
            and f["ic_t_stat"] < -1.96
            and (f["ic_ir"] < 0 or f["sharpe"] < 0)
            # Additional check: if ic_t_stat < -1.96, p-value should be < 0.05 (consistent)
            # Exclude factors with p-value = 1.0 as they indicate data inconsistency
            and f["ic_p_value"] < 1.0
        ],
        key=lambda x: (abs(x["ic_ir"]), abs(x["sharpe"])),
        reverse=True,
    )
    # No truncation - show all qualified factors

    html_content += """

            <h2>📊 Auto Summary: Qualified Factors</h2>
            <div class="info-box">
                <p><strong>Selection Criteria:</strong></p>
                <ul>
                    <li><strong>Positive Alpha Candidates:</strong> IC Mean > 0, IC t-stat > 1.96 (statistically significant), and (IC IR > 0 or Sharpe > 0)</li>
                    <li><strong>Strong Negative Factors:</strong> IC Mean < 0, IC t-stat < -1.96 (statistically significant), and (IC IR < 0 or Sharpe < 0)</li>
                </ul>
                <p><strong>Interpretation:</strong></p>
                <ul>
                    <li><strong>IC Mean:</strong> Average predictive power (direction and strength). Positive = higher factor value predicts higher future return.</li>
                    <li><strong>IC t-stat:</strong> Statistical significance. |t-stat| > 1.96 means p-value < 0.05 (95% confidence). Higher |t-stat| = more reliable signal.</li>
                    <li><strong>IC IR (Information Ratio):</strong> IC Mean / IC Std. Measures stability of predictive power. Higher = more consistent.</li>
                    <li><strong>Sharpe Ratio:</strong> Risk-adjusted return of simple long-short strategy. > 1.0 is good, > 0 is acceptable.</li>
                </ul>
                <p><strong>Usage:</strong></p>
                <ul>
                    <li><strong>Positive factors:</strong> Use directly as long signals. Higher factor value → higher expected return.</li>
                    <li><strong>Negative factors:</strong> Two options:
                        <ul>
                            <li><strong>Reverse:</strong> Multiply by -1 to create a positive signal (high negative factor → low expected return → reverse to get long signal)</li>
                            <li><strong>Risk filter:</strong> Use as-is to avoid trading when negative factors are extreme (e.g., reduce position size or skip trades)</li>
                        </ul>
                    </li>
                </ul>
                <p><strong>Note:</strong> All qualified factors are shown below (not limited to top 10). A features.yaml file with all qualified factors has been generated for iterative training.</p>
            </div>

            <div class="metrics-grid">
                <div>
                    <h3>Top Positive Alpha Candidates</h3>
    """

    if top_positive:
        html_content += f"""
                    <table>
                        <thead>
                            <tr>
                                <th>Factor</th>
                                <th>IC Mean</th>
                                <th>IC t-stat</th>
                                <th>IC p-value</th>
                                <th>IC IR</th>
                                <th>Sharpe</th>
                                <th>Best Lag</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        for f in top_positive:
            best_lag_display = (
                f"{f.get('best_lag', 'N/A')}"
                if f.get("best_lag") is not None
                else "N/A"
            )
            html_content += f"""
                            <tr>
                                <td><code>{f['name']}</code></td>
                                <td>{f['ic_mean']:.4f}</td>
                                <td>{f['ic_t_stat']:.2f}</td>
                                <td>{format_pvalue(f['ic_p_value'])}</td>
                                <td>{f['ic_ir']:.3f}</td>
                                <td>{f['sharpe']:.2f}</td>
                                <td>{best_lag_display}</td>
                            </tr>
            """
        html_content += """
                        </tbody>
                    </table>
        """
    else:
        html_content += "<p>No positive-alpha candidates (IC Mean > 0, IC t-stat > 1.96, and (IC IR > 0 or Sharpe > 0)) were found in this run.</p>"

    html_content += """
                </div>
                <div>
                    <h3>Strong Negative Factors (Consider Reversing)</h3>
    """

    if top_negative:
        html_content += f"""
                    <table>
                        <thead>
                            <tr>
                                <th>Factor</th>
                                <th>IC Mean</th>
                                <th>IC t-stat</th>
                                <th>IC p-value</th>
                                <th>IC IR</th>
                                <th>Sharpe</th>
                                <th>Best Lag</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        for f in top_negative:
            best_lag_display = (
                f"{f.get('best_lag', 'N/A')}"
                if f.get("best_lag") is not None
                else "N/A"
            )
            html_content += f"""
                            <tr>
                                <td><code>{f['name']}</code></td>
                                <td>{f['ic_mean']:.4f}</td>
                                <td>{f['ic_t_stat']:.2f}</td>
                                <td>{format_pvalue(f['ic_p_value'])}</td>
                                <td>{f['ic_ir']:.3f}</td>
                                <td>{f['sharpe']:.2f}</td>
                                <td>{best_lag_display}</td>
                            </tr>
            """
        html_content += """
                        </tbody>
                    </table>
        """
    else:
        html_content += "<p>No strong negative factors (IC Mean < 0, IC t-stat < -1.96, and (IC IR < 0 or Sharpe < 0)) were detected.</p>"

    html_content += """
                </div>
            </div>

            <!-- Main summary table -->

            <h2>📈 Summary Metrics</h2>
            <div class="info-box">
                <strong>Color Coding:</strong>
                <span class="good">Green</span> = Good (IC IR > 0.5, Sharpe > 1.0) |
                <span class="warning">Orange</span> = Warning (IC IR > 0, Sharpe > 0) |
                <span class="bad">Red</span> = Poor
            </div>
            <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Factor</th>
                        <th>N</th>
                        <th>Rank IC</th>
                        <th>Pearson</th>
                        <th>IC Mean</th>
                        <th>IC Std</th>
                        <th>IC IR</th>
                        <th>IC+ %</th>
                        <th>IC t-stat</th>
                        <th>IC p-value</th>
                        <th>Best Lag</th>
                        <th>Best IC</th>
                        <th>Q Spread</th>
                        <th>WR Long</th>
                        <th>WR Short</th>
                        <th>Ret Long</th>
                        <th>Ret Short</th>
                        <th>Long t-stat</th>
                        <th>Short t-stat</th>
                        <th>Sharpe</th>
                        <th>Calmar</th>
                        <th>Sortino</th>
                        <th>Tot Ret</th>
                        <th>Max DD</th>
                        <th>DD Dur</th>
                        <th>Turnover</th>
                        <th>F Autocorr</th>
                    </tr>
                </thead>
                <tbody>
    """

    # Inject summary rows
    html_content += "".join(summary_rows)

    html_content += """
                </tbody>
            </table>
            </div>

    """

    # Inject detailed sections for each factor
    html_content += "".join(detailed_sections)

    html_content += """
            <h2>📚 Metrics Explanation</h2>
            <div class="info-box">
                <h4>IC (Information Coefficient) Metrics:</h4>
                <ul>
                    <li><strong>Rank IC:</strong> Spearman rank correlation between factor and target</li>
                    <li><strong>Pearson IC:</strong> Pearson correlation between factor and target</li>
                    <li><strong>IC Mean/Std:</strong> Mean and std of rolling IC series (60-bar window)</li>
                    <li><strong>IC IR:</strong> Information Ratio = IC Mean / IC Std (higher is better, >0.5 is good)</li>
                    <li><strong>IC+ Ratio:</strong> Percentage of periods with positive IC</li>
                    <li><strong>IC t-stat/p-value:</strong> Statistical significance test of IC (p < 0.05 is significant)</li>
                    <li><strong>IC Lag N:</strong> IC at N bars forward (decay analysis - shows how IC decays over time)</li>
                </ul>

                <h4>Quantile Analysis:</h4>
                <ul>
                    <li><strong>Q Spread:</strong> Quantile spread = Long return - Short return (higher is better)</li>
                    <li><strong>Ret Long/Short:</strong> Average return of top/bottom quantile positions</li>
                    <li><strong>Long/Short t-stat:</strong> Statistical significance of quantile returns</li>
                    <li><strong>WR Long/Short:</strong> Win rate of long/short positions</li>
                </ul>

                <h4>Risk-Adjusted Returns:</h4>
                <ul>
                    <li><strong>Sharpe Ratio:</strong> (Mean return / Std) * sqrt(252) (higher is better, >1.0 is good)</li>
                    <li><strong>Calmar Ratio:</strong> Total return / Max drawdown (higher is better)</li>
                    <li><strong>Sortino Ratio:</strong> Downside risk-adjusted Sharpe (uses only negative returns)</li>
                </ul>

                <h4>Factor Characteristics:</h4>
                <ul>
                    <li><strong>Turnover:</strong> Position change rate (lower = more stable signals)</li>
                    <li><strong>F Autocorr:</strong> Factor autocorrelation (higher = more persistent/stable)</li>
                    <li><strong>Max DD Duration:</strong> Maximum drawdown duration in periods</li>
                </ul>

                <h4>📝 Using the Generated features.yaml:</h4>
                <p>This evaluation has generated a <code>features_suggested.yaml</code> file containing all qualified factors (positive + negative).</p>
                <ol>
                    <li><strong>Review the qualified factors</strong> in the "Auto Summary" section above</li>
                    <li><strong>Locate the generated file:</strong> <code>results/factor_ts_eval/{strategy_name}_{symbol}_features_suggested.yaml</code></li>
                    <li><strong>Replace your strategy's features.yaml:</strong> Copy the generated file to your strategy config directory:
                        <pre>cp results/factor_ts_eval/{strategy_name}_{symbol}_features_suggested.yaml config/strategies/{strategy_name}/features.yaml</pre>
                    </li>
                    <li><strong>Iterative refinement:</strong>
                        <ul>
                            <li>Train your model with the new features.yaml</li>
                            <li>Evaluate performance using <code>ts-strategy-feature-compare</code></li>
                            <li>Re-run <code>ts-factor-eval</code> to refine factor selection</li>
                            <li>Repeat until optimal factor combination is found</li>
                        </ul>
                    </li>
                </ol>
                <p><strong>Tip:</strong> Start with all qualified factors, then use ablation studies (<code>ts-strategy-feature-compare</code>) to identify the optimal subset.</p>
            </div>

            <div class="footer">
                <p>Generated by ts-factor-eval | Time-Series Factor Evaluation Tool</p>
            </div>
        </div>

        <script>
            // Initialize all IC decay charts
        </script>
    </body>
    </html>
    """

    # Inject chart scripts after f-string is defined (since chart_scripts is generated in the loop above)
    html_content = html_content.replace(
        "// Initialize all IC decay charts",
        f"// Initialize all IC decay charts\n            {''.join(chart_scripts)}",
    )

    html_path.write_text(html_content, encoding="utf-8")

    # Return qualified factors for YAML export
    qualified_factors = {
        "positive": [f["name"] for f in top_positive],
        "negative": [f["name"] for f in top_negative],
    }
    return html_path, qualified_factors


def export_features_yaml(
    qualified_factors: Dict[str, List[str]],
    strategy_name: str,
    symbol: str,
    output_dir: Path,
    strategy_config_path: Path | None = None,
    error_factors: Dict[str, Any] | None = None,
    output_path: Path | None = None,
    invert_features: List[str] | None = None,
) -> Path:
    """Export qualified factors to a features.yaml file that can be used for training."""
    import yaml
    from src.features.loader.strategy_feature_loader import StrategyFeatureLoader

    # Combine positive and negative factors (negative factors can be used as-is or reversed)
    all_factors = qualified_factors["positive"] + qualified_factors["negative"]

    # Check if any dl_seq_f* feature is in qualified factors, if so add all dl_seq_f* features
    dl_seq_features_included = False
    dl_seq_feature_prefixes = [
        "dl_seq_f",
        "dl_sequence",
    ]  # Common prefixes for deep learning sequence features
    for factor in all_factors:
        if any(factor.startswith(prefix) for prefix in dl_seq_feature_prefixes):
            dl_seq_features_included = True
            break

    # Add all dl_seq_f* features if any is qualified
    if dl_seq_features_included:
        feature_loader_temp = StrategyFeatureLoader()
        feature_definitions_temp = feature_loader_temp.feature_deps.get("features", {})
        all_dl_seq_features = [
            feat_name
            for feat_name in feature_definitions_temp.keys()
            if any(feat_name.startswith(prefix) for prefix in dl_seq_feature_prefixes)
        ]
        print(
            f"   📦 Found qualified dl_seq_f* features, adding all {len(all_dl_seq_features)} dl_seq_f* features"
        )
        # Add to all_factors but avoid duplicates
        for dl_seq_feat in all_dl_seq_features:
            if dl_seq_feat not in all_factors:
                all_factors.append(dl_seq_feat)

    if not all_factors:
        # Still export an empty YAML when output_path is requested.
        # This is important for automation (e.g. Pool-B + semantic search scripts) so downstream
        # tooling can proceed deterministically even when factor-eval finds no qualified factors.
        print("   ⚠️  No qualified factors to export. Writing an empty exported YAML.")
        out_path = output_path
        if out_path is None:
            out_path = output_dir / "features_pool_b.yaml"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        obj = {
            "name": f"{strategy_name}__pool_b_empty",
            "description": (
                "Pool-B export (empty): factor-eval found no qualified factors for this window/symbol. "
                "Check label availability, filters, and evaluation window."
            ),
            "feature_pipeline": {
                "requested_features": [],
                "invert_features": list(invert_features or []),
            },
            "factor_eval": {
                "strategy": strategy_name,
                "symbol": symbol,
                "status": "empty",
                "reason": "no_qualified_factors",
            },
        }
        out_path.write_text(
            yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        return out_path

    # Collect non-numeric / invalid factors from error_factors (to be added at the end)
    # NOTE: These factors failed numeric conversion or had insufficient samples during IC evaluation.
    # They are included in the YAML for manual review and verification, but should NOT be used
    # directly in model training without further investigation:
    #   - non_numeric_factor: May be categorical/string features that need encoding
    #   - insufficient_samples: May need more data or different evaluation window
    # These factors are placed at the end of requested_features list for easy identification.
    non_numeric_factors = []
    if error_factors is not None:
        for factor_name, error_info in error_factors.items():
            error_type = error_info.get("error", "")
            # Include non-numeric factors and other invalid factors
            if error_type in ["non_numeric_factor", "insufficient_samples"]:
                non_numeric_factors.append(factor_name)
        if non_numeric_factors:
            print(
                f"   📋 Adding {len(non_numeric_factors)} non-numeric/invalid factors (will be placed at the end)"
            )
            print(
                f"      ⚠️  WARNING: These factors require manual verification before use in training"
            )

    # Initialize non_numeric_mapped early to ensure it exists even if error_factors is None or empty
    # This set will contain feature names that need manual verification before use in training
    non_numeric_mapped = set()

    # Map output columns to feature compute function names (with _f suffix)
    # Some factors are output columns (e.g., bb_lower, bb_middle) that need to be mapped to feature compute functions (e.g., bb_width_f)
    feature_loader = StrategyFeatureLoader()
    feature_definitions = feature_loader.feature_deps.get("features", {})

    # Build mapping: output_column -> feature_compute_function_name (with _f suffix)
    output_to_feature = {}
    for feat_name, feat_info in feature_definitions.items():
        output_cols = feat_info.get("output_columns", [])
        for col in output_cols:
            if col not in output_to_feature:
                output_to_feature[col] = feat_name

    # Map factors to feature compute function names (only keep feature compute function names, not output columns)
    # This ensures the generated YAML only contains feature compute function names that can be loaded
    mapped_features = set()
    unmapped_factors = []
    mapped_outputs = []  # Track which output columns were mapped

    # First, process qualified factors (positive + negative)
    qualified_factor_names = set(all_factors)

    for factor in all_factors:
        # Check if it's already a feature compute function name (with _f suffix)
        if factor in feature_definitions:
            mapped_features.add(factor)
        # Check if it's an output column - map to source feature compute function
        elif factor in output_to_feature:
            source_feature = output_to_feature[factor]
            mapped_features.add(source_feature)
            mapped_outputs.append((factor, source_feature))
        else:
            # Not found - check if it's an old name (without _f suffix)
            potential_old_name = factor
            potential_new_name = f"{factor}_f"
            if potential_new_name in feature_definitions:
                # This is an old name without _f suffix - raise error
                raise ValueError(
                    f"Feature compute function name must end with '_f' suffix. "
                    f"Got: '{factor}'. Did you mean '{potential_new_name}'?"
                )
            # Not found - keep it anyway (might be a raw column or deprecated feature)
            unmapped_factors.append(factor)
            mapped_features.add(factor)

    # Now add non-numeric/invalid factors at the end (but map them first)
    # NOTE: These factors require manual verification before use in training.
    # They are placed at the end of requested_features for easy identification.
    # See notes section in generated YAML for detailed warning information.
    if non_numeric_factors:
        for factor in non_numeric_factors:
            # Check if it's already a feature compute function name
            if factor in feature_definitions:
                non_numeric_mapped.add(factor)
            # Check if it's an output column - map to source feature
            elif factor in output_to_feature:
                source_feature = output_to_feature[factor]
                non_numeric_mapped.add(source_feature)
                mapped_outputs.append((factor, source_feature))
            else:
                # Not found - keep it anyway (might be a raw column or deprecated feature)
                unmapped_factors.append(factor)
                non_numeric_mapped.add(factor)

        # Add non-numeric factors to mapped_features (they will be sorted and placed at the end)
        mapped_features.update(non_numeric_mapped)

    if mapped_outputs:
        print(
            f"   📋 Mapped {len(mapped_outputs)} feature output columns to feature compute functions:"
        )
        for output_col, source_feat in sorted(mapped_outputs)[:5]:
            print(f"      {output_col} → {source_feat}")
        if len(mapped_outputs) > 5:
            print(f"      ... and {len(mapped_outputs) - 5} more")

    if unmapped_factors:
        print(f"   ⚠️  {len(unmapped_factors)} factors not found in feature registry:")
        for factor in sorted(unmapped_factors)[:10]:
            print(f"      - {factor}")
        if len(unmapped_factors) > 10:
            print(f"      ... and {len(unmapped_factors) - 10} more")

    # Load base strategy config if available to preserve other settings
    base_config = {}
    if strategy_config_path and strategy_config_path.exists():
        try:
            with open(strategy_config_path, "r", encoding="utf-8") as f:
                base_config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"   ⚠️  Could not load base strategy config: {e}")

    # Build features.yaml structure
    features_config = {
        "name": base_config.get("name", strategy_name),
        "description": base_config.get(
            "description",
            f"Feature pipeline with qualified factors from factor evaluation ({symbol})",
        ),
        "feature_pipeline": {
            "requested_features": (
                sorted(
                    [f for f in mapped_features if f not in non_numeric_mapped]
                )  # Qualified factors first (positive + negative)
                + sorted(
                    non_numeric_mapped
                )  # Non-numeric/invalid factors at the end (⚠️ requires manual verification - see notes)
            ),
            # Optional: output-column names to multiply by -1 BEFORE training/inference.
            # These come from "Strong Negative Factors (Consider Reversing)" in the report.
            # NOTE: requested_features are compute functions; invert_features are output columns.
            **(
                {"invert_features": sorted(set(invert_features))}
                if invert_features
                else {}
            ),
            "ensure_signal_column": base_config.get("feature_pipeline", {}).get(
                "ensure_signal_column",
                {"name": "signal", "default_value": 0},
            ),
        },
        "notes": f"""
Auto-generated from factor evaluation results.

Factor counts (from HTML report):
- Positive factors ({len(qualified_factors['positive'])}): Direct alpha signals
- Negative factors ({len(qualified_factors['negative'])}): Consider reversing (multiply by -1) or use as risk filters
- Total qualified factors: {len(qualified_factors['positive']) + len(qualified_factors['negative'])}
- Non-numeric/invalid factors ({len(non_numeric_factors)}): Added for manual verification (placed at the end)

Feature function counts (in this YAML):
- Qualified feature functions: {len([f for f in mapped_features if f not in non_numeric_mapped])}
- Non-numeric/invalid feature functions: {len(non_numeric_mapped)}
- Total feature functions: {len(mapped_features)}

📊 Why the counts differ:
  The HTML report shows individual factor columns (e.g., "bb_lower", "bb_middle", "bb_upper"), 
  but these output columns are mapped to their source feature compute functions (e.g., "bb_width_f").
  When multiple output columns from the same feature function are qualified, they are merged into 
  a single feature function entry in this YAML file. This is correct behavior - requesting the 
  feature function once will generate all its output columns.

  - {len(mapped_outputs)} output columns were mapped to {len(set(source_feat for _, source_feat in mapped_outputs))} unique feature functions

Selection criteria:
- Positive: IC Mean > 0, IC t-stat > 1.96, (IC IR > 0 or Sharpe > 0)
- Negative: IC Mean < 0, IC t-stat < -1.96, (IC IR < 0 or Sharpe < 0)
- Non-numeric/invalid: Factors that failed numeric conversion or had insufficient samples

⚠️  IMPORTANT: Non-numeric/invalid factors require further verification before use:
  - These factors failed IC evaluation due to non-numeric data or insufficient samples
  - They may be categorical/string features that need proper encoding
  - They may need more data or different evaluation parameters
  - DO NOT use these factors directly in model training without investigation
  - Review the factor evaluation HTML report for detailed error information

Special handling:
- dl_seq_f* features: If any dl_seq_f* feature is qualified, all dl_seq_f* features are included

Note: Feature output columns (e.g., bb_lower, bb_middle) have been mapped to their source feature compute functions (e.g., bb_width_f).
Only feature compute function names (with _f suffix) are included in requested_features for cleaner configuration.

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """.strip(),
    }

    # Write YAML file
    yaml_path = (
        Path(output_path)
        if output_path is not None
        else (output_dir / f"{strategy_name}_{symbol}_features_suggested.yaml")
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(
            features_config,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    return yaml_path


def main() -> None:
    args = parse_args()
    strategy_config_path = Path(args.strategy_config)

    # Support both directory and file paths
    # If it's a file (e.g., features_all.yaml), use its parent directory as strategy config
    # and temporarily use that file as features.yaml
    features_file_override = None
    if strategy_config_path.is_file():
        # It's a features.yaml file, extract the directory
        features_file_override = strategy_config_path
        strategy_config_dir = strategy_config_path.parent
        print(f"📁 Detected features file: {features_file_override}")
        print(f"   Using strategy config directory: {strategy_config_dir}")
    else:
        # It's a directory, use default features.yaml
        strategy_config_dir = strategy_config_path

    loader = StrategyConfigLoader(strategy_config_dir)
    strategy_cfg = loader.load()

    # ------------------------------------------------------------------
    # Convention-by-default:
    # - If user did not override --output-dir (i.e., kept legacy default),
    #   redirect outputs to results/pools/<strategy_name>/pool_b
    # - If user did not set --export-yaml, default export path to:
    #   results/pools/<strategy_name>/pool_b/features_pool_b.yaml
    # ------------------------------------------------------------------
    legacy_default_out = "results/factor_ts_eval"
    strategy_dir_id = strategy_config_dir.name
    if str(args.output_dir) == legacy_default_out:
        args.output_dir = str(_default_pool_b_dir(strategy_dir_id))
    if args.export_yaml is None:
        args.export_yaml = str(_default_pool_b_yaml_path(strategy_dir_id))

    # If a features file was specified, directly override the features config
    # features_all.yaml should be self-contained with all dependencies included
    if features_file_override:
        import yaml
        from src.time_series_model.strategy_config import FeaturePipelineConfig

        with open(features_file_override, "r", encoding="utf-8") as f:
            features_override = yaml.safe_load(f)

        # Directly override strategy_cfg.features with features_all.yaml content
        if "feature_pipeline" in features_override:
            override_requested = features_override["feature_pipeline"].get(
                "requested_features", []
            )

            # Create new FeaturePipelineConfig from override file
            strategy_cfg.features = FeaturePipelineConfig(
                requested_features=override_requested,
                invert_features=features_override["feature_pipeline"].get(
                    "invert_features", []
                ),
                post_processors=[],  # features_all.yaml typically doesn't have post_processors
                selector=None,
                ensure_signal=features_override["feature_pipeline"].get(
                    "ensure_signal_column",
                    strategy_cfg.features.ensure_signal,
                ),
            )

            print(
                f"   ✅ Overridden features config with {features_file_override.name}"
            )
            print(
                f"      Using {len(override_requested)} features (self-contained, all dependencies included)"
            )
            print(
                f"      Note: features_all.yaml should be self-contained and not require features.yaml"
            )

            # Set factors to use the overridden features
            args.factors = None  # Will use strategy_cfg.features.requested_features
            args.feature_mode = (
                "strategy"  # Use strategy mode (features_all.yaml is self-contained)
            )

    # If factors not specified, use requested_features from strategy config
    if args.factors is None:
        strategy_requested = strategy_cfg.features.requested_features or []
        if not strategy_requested:
            raise ValueError(
                f"No factors specified and strategy config '{args.strategy_config}' "
                f"has no requested_features in features.yaml. "
                f"Please either specify --factors or add requested_features to the config."
            )
        args.factors = strategy_requested
        print(f"📋 Using factors from strategy config: {len(args.factors)} factors")
        print(
            f"   Factors: {', '.join(args.factors[:10])}{'...' if len(args.factors) > 10 else ''}"
        )

    df = prepare_dataset(args, strategy_cfg)
    target_col = strategy_cfg.labels.target_column

    # Debug: Check dataset after prepare_dataset
    print(f"\n📊 Dataset after prepare_dataset:")
    print(f"   Shape: {df.shape}")
    if target_col in df.columns:
        label_non_null = df[target_col].notna().sum()
        print(f"   Label ({target_col}) non-null: {label_non_null} / {len(df)}")
        if label_non_null == 0:
            print(f"   ⚠️  WARNING: Label column is all NaN!")
    else:
        print(f"   ❌ ERROR: Label column '{target_col}' not found in DataFrame!")

    # Check feature columns
    feature_cols = [
        c
        for c in df.columns
        if c
        not in [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "timestamp",
            "_symbol",
            "signal",
            target_col,
        ]
    ]
    if feature_cols:
        sample_feature = feature_cols[0] if feature_cols else None
        if sample_feature:
            feat_non_null = df[sample_feature].notna().sum()
            print(
                f"   Sample feature ({sample_feature}) non-null: {feat_non_null} / {len(df)}"
            )
            if target_col in df.columns:
                valid_samples = df[[sample_feature, target_col]].dropna()
                print(
                    f"   Joint valid samples ({sample_feature} + {target_col}): {len(valid_samples)}"
                )

    # Parse IC decay lags
    ic_decay_lags = [int(x.strip()) for x in args.ic_decay_lags.split(",") if x.strip()]

    results: Dict[str, Dict[str, float]] = {}
    ic_series_data: Dict[str, pd.DataFrame] = {}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Resolve factors:
    # - 如果因子名是 DataFrame 中的列名：直接使用该列
    # - 如果因子名是特征名（存在于 feature_dependencies.yaml）：
    #     使用其 output_columns 中「实际存在于 DataFrame」的列
    # - 同一个列名只计算一次（即使既写了特征名又写了列名）
    # ------------------------------------------------------------------
    feature_loader = StrategyFeatureLoader()
    features_config = feature_loader.feature_deps.get("features", {})

    # Diagnostics containers
    diagnostics: Dict[str, Any] = {}
    factor_resolution: Dict[str, Any] = {
        "mappings": {},  # col_name -> set(sources)
        "unknown_factors": [],
        "missing_feature_outputs": [],
    }

    # col_name -> set(sources)  例如: "hilbert_price_env" <- {"hilbert_phase", "hilbert_price_env"}
    cols_to_evaluate: Dict[str, set] = {}

    # Build reverse mapping: output_column -> feature_compute_function_name
    output_to_feature = {}
    for feat_name, feat_info in features_config.items():
        output_cols = feat_info.get("output_columns", [])
        for col in output_cols:
            if col not in output_to_feature:
                output_to_feature[col] = feat_name

    # Track features already reported as missing to avoid duplicates
    missing_features_reported = set()

    for factor in args.factors:
        # 1. Check if it's a DataFrame column
        if factor in df.columns:
            cols_to_evaluate.setdefault(factor, set()).add(factor)
            continue

        # 2. Check if it's a feature compute function name (with _f suffix)
        if factor in features_config:
            feature_info = features_config[factor]
            output_cols = feature_info.get("output_columns", [factor])
            existing_cols = [c for c in output_cols if c in df.columns]

            if not existing_cols:
                # Only report once per feature
                if factor not in missing_features_reported:
                    msg = {
                        "feature": factor,
                        "expected": output_cols,
                    }
                    factor_resolution["missing_feature_outputs"].append(msg)
                    missing_features_reported.add(factor)
                    print(
                        f"   ❌ Feature compute function '{factor}' found in config, but none of its output columns "
                        f"exist in DataFrame. Expected: {output_cols}"
                    )
                continue

            for col in existing_cols:
                cols_to_evaluate.setdefault(col, set()).add(factor)
        # 3. Check if it's a feature output column name (reverse lookup)
        elif factor in output_to_feature:
            source_feature = output_to_feature[factor]
            # Check if this output column exists in DataFrame
            if factor in df.columns:
                cols_to_evaluate.setdefault(factor, set()).add(source_feature)
            else:
                # Output column doesn't exist in DataFrame
                # Only report once per feature
                if source_feature not in missing_features_reported:
                    feature_info = features_config[source_feature]
                    output_cols = feature_info.get("output_columns", [])
                    msg = {
                        "feature": source_feature,
                        "expected": output_cols,
                    }
                    factor_resolution["missing_feature_outputs"].append(msg)
                    missing_features_reported.add(source_feature)
                    print(
                        f"   ❌ Feature output column '{factor}' (from feature compute function '{source_feature}') "
                        f"does not exist in DataFrame. Expected output columns: {output_cols}"
                    )
        else:
            # 4. Not found - check if it's an old feature name (without _f suffix)
            potential_old_name = factor
            potential_new_name = f"{factor}_f"
            if potential_new_name in features_config:
                factor_resolution["unknown_factors"].append(factor)
                print(
                    f"   ❌ Factor '{factor}' is an old feature compute function name (without '_f' suffix). "
                    f"Did you mean '{potential_new_name}'? "
                    f"Or if you want to use the output column, please check if '{factor}' is in the output columns of '{potential_new_name}'."
                )
            else:
                factor_resolution["unknown_factors"].append(factor)
                print(
                    f"   ❌ Factor '{factor}' is neither a DataFrame column nor a known feature compute function "
                    f"(with '_f' suffix) or feature output column in feature_dependencies.yaml"
                )

    if not cols_to_evaluate:
        print("   ❌ No valid factors/columns to evaluate. Exiting.")
        return

    # 打印映射关系，方便调试
    print("\n   🔍 Factor resolution (config entry -> actual DataFrame columns):")
    for col, sources in sorted(cols_to_evaluate.items()):
        print(f"      - {col}: requested as {sorted(sources)}")

    # 保存映射到 diagnostics (convert sets to lists for JSON serialization)
    factor_resolution["mappings"] = {
        col: sorted(list(sources)) for col, sources in cols_to_evaluate.items()
    }
    diagnostics["factor_resolution"] = factor_resolution

    # Create subdirectory for IC series CSV files
    ic_series_dir = output_dir / "ic_series"
    ic_series_dir.mkdir(parents=True, exist_ok=True)

    # 逐列计算指标（每个实际列只计算一次）
    for col in cols_to_evaluate.keys():
        if col not in df.columns:
            # 理论上不会发生，这里仅做保护
            print(f"   ❌ Column '{col}' not found in DataFrame (skipped).")
            continue

        metrics, ic_series_df = compute_factor_metrics(
            df, col, target_col, args.quantile, ic_decay_lags
        )
        results[col] = metrics
        if not ic_series_df.empty:
            ic_series_data[col] = ic_series_df
            # Save IC series to CSV in subdirectory
            ic_csv_path = ic_series_dir / f"ic_series_{col}_{args.symbol}.csv"
            ic_series_df.to_csv(ic_csv_path)
            print(f"   💾 Saved IC series to {ic_csv_path}")

    # Collect error factors for diagnostics
    error_factors = {f: m for f, m in results.items() if "error" in m}
    diagnostics["error_factors"] = error_factors

    # Apply correlation-based removal if requested
    if args.remove_correlated and len(results) > 1:
        print(
            f"\n🔗 Removing highly correlated features (threshold: {args.correlation_threshold})..."
        )
        # Get valid factors (non-error, numeric)
        valid_factors = {
            col: metrics
            for col, metrics in results.items()
            if "error" not in metrics and col in df.columns
        }

        if len(valid_factors) > 1:
            # Get factor data for correlation calculation
            factor_cols = list(valid_factors.keys())
            factor_df = df[factor_cols].dropna()

            if len(factor_df) > 50:  # Need sufficient samples
                # Calculate correlation matrix
                corr_matrix = factor_df.corr().abs()

                # Sort by IC IR (if available) or IC Mean
                factor_scores = {}
                for col in factor_cols:
                    metrics = results[col]
                    # Use IC IR as priority, fallback to IC Mean
                    score = (
                        metrics.get("ic_ir", 0.0)
                        if metrics.get("ic_ir", 0.0) is not None
                        else metrics.get("ic_mean", 0.0)
                    )
                    factor_scores[col] = abs(score) if score is not None else 0.0

                # Sort by score (descending)
                sorted_factors = sorted(
                    factor_cols, key=lambda x: factor_scores.get(x, 0.0), reverse=True
                )

                # Keep features with low correlation to already selected ones
                kept_factors = []
                removed_factors = []
                removal_reasons = {}

                for factor in sorted_factors:
                    # Check correlation with already kept factors
                    should_keep = True
                    removal_reason = None

                    for kept in kept_factors:
                        corr = corr_matrix.loc[factor, kept]
                        if corr >= args.correlation_threshold:
                            # Factor is highly correlated with a kept factor
                            # Keep the one with higher score
                            if factor_scores[factor] <= factor_scores[kept]:
                                should_keep = False
                                removal_reason = f"Highly correlated with {kept} (r={corr:.3f}), lower score ({factor_scores[factor]:.4f} vs {factor_scores[kept]:.4f})"
                                break

                    if should_keep:
                        kept_factors.append(factor)
                    else:
                        removed_factors.append(factor)
                        if removal_reason:
                            removal_reasons[factor] = removal_reason

                print(
                    f"   ✅ Kept {len(kept_factors)} features, removed {len(removed_factors)} redundant features"
                )
                if removed_factors:
                    print(
                        f"   Removed factors: {', '.join(removed_factors[:10])}{'...' if len(removed_factors) > 10 else ''}"
                    )

                # Update results to only include kept factors
                results = {
                    col: metrics
                    for col, metrics in results.items()
                    if col in kept_factors
                }

                # Save removal info to diagnostics
                diagnostics["correlation_removal"] = {
                    "removed": removed_factors,
                    "kept": kept_factors,
                    "threshold": args.correlation_threshold,
                    "removal_reasons": removal_reasons,
                }
            else:
                print(
                    f"   ⚠️  Insufficient data for correlation analysis (need >50 samples, got {len(factor_df)})"
                )
        else:
            print(
                f"   ⚠️  Too few valid factors for correlation removal ({len(valid_factors)})"
            )

    # Always compute best lag info for display in HTML report, even if not filtering
    # This ensures best lag is shown in "Top Positive Alpha Candidates" and "Strong Negative Factors" tables
    lag_info = {}
    for col, metrics in results.items():
        if "error" in metrics:
            continue

        # Find best lag
        best_lag = None
        best_ic = None

        for lag in ic_decay_lags:
            ic_val = metrics.get(f"ic_lag_{lag}", None)
            if ic_val is not None:
                try:
                    ic_val_float = float(ic_val)
                    if not np.isnan(ic_val_float):
                        if best_ic is None or ic_val_float > best_ic:
                            best_ic = ic_val_float
                            best_lag = lag
                except (ValueError, TypeError):
                    continue

        # Fallback to rank_ic if no valid lag found
        if best_lag is None:
            rank_ic = metrics.get("rank_ic", None)
            if rank_ic is not None:
                try:
                    rank_ic_float = float(rank_ic)
                    if not np.isnan(rank_ic_float) and rank_ic_float != 0.0:
                        best_lag = 0  # Current period
                        best_ic = rank_ic_float
                except (ValueError, TypeError):
                    pass

        lag_info[col] = {"best_lag": best_lag, "best_ic": best_ic}

    # Apply best lag filtering if requested or if target_lag is explicitly specified
    # If user specifies target_lag, it implies they want to filter by best lag
    should_filter_by_lag = args.filter_by_best_lag or args.target_lag is not None

    if should_filter_by_lag and len(results) > 0:
        print(f"\n⏱️  Filtering features by best lag...")

        # Determine target lag
        target_lag = args.target_lag
        if target_lag is None:
            # Try to infer from strategy config
            try:
                max_holding_bars = strategy_cfg.labels.generator.params.get(
                    "max_holding_bars", None
                )
                if max_holding_bars:
                    # max_holding_bars 是标签生成时的最大扫描周期（上限）
                    # 实际持仓周期通常小于这个值（一旦触达止盈/止损就会提前平仓）
                    # 对于 SR Reversal 策略，典型持仓周期通常是 max_holding_bars 的 30-50%
                    # 我们使用 40% 作为典型持仓周期的估计
                    typical_holding_bars = int(max_holding_bars * 0.4)
                    # 但不要超过 max_holding_bars 的一半，也不要小于 10
                    target_lag = max(
                        10, min(typical_holding_bars, max_holding_bars // 2)
                    )
                    print(
                        f"   Inferred target lag from max_holding_bars ({max_holding_bars}): "
                        f"target_lag={target_lag} (estimated typical holding period = {max_holding_bars} * 0.4)"
                    )
                    print(
                        f"   Note: max_holding_bars is the label generation upper bound, "
                        f"while target_lag represents typical holding period for feature selection"
                    )
                else:
                    # Default to middle of ic_decay_lags
                    target_lag = sorted(ic_decay_lags)[len(ic_decay_lags) // 2]
                    print(
                        f"   Using default target lag (middle of IC decay lags): {target_lag}"
                    )
            except Exception as e:
                print(f"   ⚠️  Could not infer target lag: {e}, using default: 10")
                target_lag = 10

        tolerance = args.lag_tolerance
        print(f"   Target lag: {target_lag}, Tolerance: ±{tolerance}")

        kept_factors = []
        removed_factors = []
        # lag_info already computed above, reuse it for filtering

        for col, metrics in results.items():
            if "error" in metrics:
                # Keep error factors for reporting
                kept_factors.append(col)
                continue

            # Get best lag from pre-computed lag_info
            best_lag_info = lag_info.get(col, {})
            best_lag = best_lag_info.get("best_lag")

            # Check if best lag is within tolerance
            if best_lag is not None:
                lag_diff = abs(best_lag - target_lag)
                if lag_diff <= tolerance:
                    kept_factors.append(col)
                else:
                    removed_factors.append((col, best_lag, lag_diff))
            else:
                # No valid best lag found, remove it
                removed_factors.append((col, None, None))

        print(
            f"   ✅ Kept {len(kept_factors)} features, removed {len(removed_factors)} features"
        )
        if removed_factors:
            removed_display = [
                (
                    f"{col} (best_lag={lag})"
                    if lag is not None
                    else f"{col} (no valid lag)"
                )
                for col, lag, _ in removed_factors[:10]
            ]
            print(
                f"   Removed factors: {', '.join(removed_display)}{'...' if len(removed_factors) > 10 else ''}"
            )

        # Update results
        results = {
            col: metrics for col, metrics in results.items() if col in kept_factors
        }

        # Save lag filtering info to diagnostics
        diagnostics["best_lag_filtering"] = {
            "target_lag": target_lag,
            "tolerance": tolerance,
            "kept": kept_factors,
            "removed": [col for col, _, _ in removed_factors],
            "lag_info": lag_info,
        }
    else:
        # Even if not filtering, save lag_info to diagnostics for HTML report display
        diagnostics["best_lag_filtering"] = {
            "target_lag": None,
            "tolerance": None,
            "kept": list(results.keys()),
            "removed": [],
            "lag_info": lag_info,
        }

    # Save JSON summary
    summary_path = output_dir / f"ts_eval_{strategy_cfg.name}_{args.symbol}.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "strategy": strategy_cfg.name,
                "symbol": args.symbol,
                "ic_decay_lags": ic_decay_lags,
                "results": results,
                "diagnostics": diagnostics,
            },
            fh,
            indent=2,
        )

    print(f"✅ Saved time-series factor evaluation to {summary_path}")

    # Generate HTML report
    qualified_factors = None
    if args.generate_html:
        html_path, qualified_factors = generate_html_report(
            results,
            ic_series_data,
            strategy_cfg.name,
            args.symbol,
            output_dir,
            ic_decay_lags,
            diagnostics=diagnostics,
        )
        print(f"✅ Generated HTML report: {html_path}")

        # Export features.yaml with qualified factors
        if qualified_factors:
            strategy_config_path = Path(args.strategy_config) / "features.yaml"
            error_factors = diagnostics.get("error_factors", {})
            invert_features = _invert_features_from_negative(qualified_factors)
            yaml_path = export_features_yaml(
                qualified_factors,
                strategy_cfg.name,
                args.symbol,
                output_dir,
                strategy_config_path=strategy_config_path,
                error_factors=error_factors,
                output_path=Path(args.export_yaml) if args.export_yaml else None,
                invert_features=invert_features,
            )
            if yaml_path:
                print(f"✅ Exported qualified factors to: {yaml_path}")
                print(f"   - Positive factors: {len(qualified_factors['positive'])}")
                print(f"   - Negative factors: {len(qualified_factors['negative'])}")
                print(
                    f"   - Total: {len(qualified_factors['positive']) + len(qualified_factors['negative'])}"
                )
                print(
                    f"   💡 You can use this file to replace your strategy's features.yaml for iterative training."
                )

            if args.export_yaml:
                print(
                    "   ✅ Included feature_pipeline.invert_features in the exported YAML "
                    f"(n={len(invert_features)})"
                )

        # Open in browser if requested (only in non-Docker/local environment)
        if args.open_browser:
            # Check if running in Docker (common indicators)
            in_docker = (
                os.path.exists("/.dockerenv")
                or os.environ.get("DOCKER_CONTAINER") == "1"
                or os.environ.get("DEV_CONTAINER") == "1"
            )

            if in_docker:
                print(f"⚠️  Running in Docker - cannot open browser automatically")
                print(f"   Please open manually: {html_path}")
                # Try to print a local path if mounted
                local_path = str(html_path).replace("/workspace/", "")
                if local_path != str(html_path):
                    print(f"   Local path (if mounted): {local_path}")
            else:
                try:
                    # Convert to absolute path and use file:// URL
                    abs_path = html_path.resolve()
                    file_url = f"file://{abs_path}"
                    webbrowser.open(file_url)
                    print(f"🌐 Opened report in browser: {file_url}")
                except Exception as e:
                    print(f"⚠️  Could not open browser automatically: {e}")
                    print(f"   Please open manually: {html_path}")


if __name__ == "__main__":
    main()
