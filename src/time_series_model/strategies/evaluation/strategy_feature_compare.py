#!/usr/bin/env python3
"""Compare strategy performance across different feature configurations."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[
    3
]  # CURRENT_DIR=evaluation -> parents[0]=strategies -> [1]=time_series_model -> [2]=src -> [3]=project_root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("strategy_feature_compare")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
logger.setLevel(logging.INFO)

# Import train_strategy_pipeline using importlib to avoid import path issues
import importlib.util

train_strategy_pipeline_path = PROJECT_ROOT / "scripts" / "train_strategy_pipeline.py"
spec = importlib.util.spec_from_file_location(
    "train_strategy_pipeline", train_strategy_pipeline_path
)
strategy_runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy_runner)
from src.data_tools.data_utils import load_raw_data  # noqa: E402
from src.features.loader.strategy_feature_loader import (
    StrategyFeatureLoader,
)  # noqa: E402
from src.time_series_model.strategy_config import (
    StrategyConfig,
    StrategyConfigLoader,
)  # noqa: E402
from src.data_tools.tick_loader import (
    list_tick_files,
    serialize_tick_loader_params,
)  # noqa: E402
from src.time_series_model.strategies.evaluation.reports.strategy_feature_compare_report import (
    write_strategy_feature_compare_reports,
)

VENDOR_DIR = PROJECT_ROOT / "vendor"
if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare strategy feature configurations."
    )
    parser.add_argument(
        "--strategy-config", required=True, help="Base strategy directory"
    )
    parser.add_argument(
        "--feature-overrides",
        nargs="*",
        default=[],
        help="List of variant definitions in the form name=path/to/features.yaml",
    )
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--data-path", default="data/parquet_data")
    parser.add_argument("--timeframe", default="240T")
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional inclusive start date (e.g. 2022-01-01)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional inclusive end date (e.g. 2023-01-01)",
    )
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--output-dir", default="results/strategy_compare")
    parser.add_argument("--run-rolling", action="store_true")
    parser.add_argument("--rolling-train-bars", type=int, default=5000)
    parser.add_argument("--rolling-test-bars", type=int, default=1000)
    parser.add_argument("--rolling-step-bars", type=int, default=1000)
    parser.add_argument("--rolling-max-windows", type=int, default=5)
    parser.add_argument(
        "--test-warmup-bars",
        type=int,
        default=200,
        help="Extra bars before the test split for feature warm-up",
    )
    parser.add_argument(
        "--calibrate-proba",
        choices=["none", "platt", "isotonic"],
        default="none",
        help="Optional probability calibration (binary only).",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def dump_yaml(path: Path, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


def merge_features(base_path: Path, override_path: Path, variant_name: str) -> Dict:
    base_data = load_yaml(base_path)
    override_data = load_yaml(override_path)
    merged = dict(base_data)
    for key, value in override_data.items():
        if key == "feature_pipeline" and isinstance(value, dict):
            merged.setdefault("feature_pipeline", {})
            merged["feature_pipeline"].update(value)
        else:
            merged[key] = value
    merged["name"] = f"{base_data.get('name', 'strategy')}_{variant_name}"
    return merged


def update_meta(meta_path: Path, variant_name: str) -> None:
    data = load_yaml(meta_path)
    strategy_info = data.get("strategy", {})
    base_name = strategy_info.get("name", variant_name)
    strategy_info["name"] = f"{base_name}_{variant_name}"
    data["strategy"] = strategy_info
    dump_yaml(meta_path, data)


@dataclass
class VariantSpec:
    name: str
    config_dir: Path
    is_temp: bool = False


def build_variants(
    base_dir: Path, overrides: List[str]
) -> Tuple[List[VariantSpec], List[Path]]:
    variants = [VariantSpec(name="base", config_dir=base_dir, is_temp=False)]
    temp_dirs: List[Path] = []

    for entry in overrides:
        if "=" in entry:
            variant_name, override_path = entry.split("=", 1)
        else:
            variant_name = Path(entry).stem
            override_path = entry
        variant_name = variant_name.strip()
        override_path = override_path.strip()

        # Resolve path relative to base_dir if not absolute
        override_path = Path(override_path)
        if not override_path.is_absolute():
            # Try relative to base_dir first
            candidate_path = base_dir / override_path
            if candidate_path.exists():
                override_path = candidate_path.resolve()
            else:
                # Fall back to current working directory
                fallback_path = Path.cwd() / override_path
                if fallback_path.exists():
                    override_path = fallback_path.resolve()
                else:
                    override_path = override_path.resolve()

        if not override_path.exists():
            # Provide more helpful error message with attempted paths
            attempted_paths = [
                (
                    str(base_dir / override_path.name)
                    if not override_path.is_absolute()
                    else None
                ),
                (
                    str(Path.cwd() / override_path.name)
                    if not override_path.is_absolute()
                    else None
                ),
                str(override_path),
            ]
            attempted_paths = [p for p in attempted_paths if p]
            raise FileNotFoundError(
                f"Override file not found: {override_path} (resolved from '{entry}'). "
                f"Tried paths: {', '.join(attempted_paths)}. "
                f"Base dir: {base_dir}"
            )

        temp_dir = Path(tempfile.mkdtemp(prefix=f"strategy_variant_{variant_name}_"))
        shutil.copytree(base_dir, temp_dir, dirs_exist_ok=True)
        merged_features = merge_features(
            base_dir / "features.yaml", override_path, variant_name
        )
        dump_yaml(temp_dir / "features.yaml", merged_features)
        meta_path = temp_dir / "meta.yaml"
        if meta_path.exists():
            update_meta(meta_path, variant_name)
        variants.append(
            VariantSpec(name=variant_name, config_dir=temp_dir, is_temp=True)
        )
        temp_dirs.append(temp_dir)

    return variants, temp_dirs


def _configure_vpin_ticks(
    feature_loader: StrategyFeatureLoader,
    symbol: str,
    data_path: str | Path,
    start_ts: Optional[str],
    end_ts: Optional[str],
) -> None:
    """
    Configure ticks_loader_json for features that require tick data.

    This project is based on tick data, so tick files must be available.
    If tick files are not found, this function will raise an exception
    indicating a configuration error.

    Args:
        feature_loader: StrategyFeatureLoader instance
        symbol: Trading symbol (e.g., 'BTCUSDT')
        data_path: Path to tick data directory
        start_ts: Start timestamp string
        end_ts: End timestamp string

    Raises:
        ValueError: If tick files are not found (configuration error)
        RuntimeError: If tick configuration fails due to other errors
    """
    if not start_ts or not end_ts:
        raise ValueError(
            f"Cannot configure VPIN ticks: missing start_ts or end_ts. "
            f"start_ts={start_ts}, end_ts={end_ts}"
        )

    features_cfg = feature_loader.feature_deps.get("features", {})
    if not features_cfg:
        raise ValueError(
            "Cannot configure VPIN ticks: features config is empty. "
            "This indicates a configuration loading error."
        )

    logger.info(
        "Configuring VPIN ticks for symbol=%s, data_path=%s, " "start_ts=%s, end_ts=%s",
        symbol,
        data_path,
        start_ts,
        end_ts,
    )

    # 检查是否已经有 ticks_loader_json（从 vpin_features 或其他特征）
    ticks_loader_json = None
    for feature_name, feature_cfg in features_cfg.items():
        compute_params = feature_cfg.get("compute_params", {})
        if compute_params.get("ticks_loader_json"):
            ticks_loader_json = compute_params["ticks_loader_json"]
            logger.info("Found existing ticks_loader_json from %s", feature_name)
            break

    # 如果还没有，创建新的
    if not ticks_loader_json:
        try:
            logger.debug(
                "Searching for tick files: symbol=%s, ticks_dir=%s, "
                "start_ts=%s, end_ts=%s, lookback_minutes=60",
                symbol,
                data_path,
                start_ts,
                end_ts,
            )

            tick_files = list_tick_files(
                symbol=symbol,
                start_ts=start_ts,
                end_ts=end_ts,
                ticks_dir=str(data_path),
                lookback_minutes=60,
            )

            if not tick_files:
                error_msg = (
                    f"VPIN tick files not found! This project is based on tick data, "
                    f"so this indicates a configuration error.\n"
                    f"  Symbol: {symbol}\n"
                    f"  Data path: {data_path}\n"
                    f"  Start timestamp: {start_ts}\n"
                    f"  End timestamp: {end_ts}\n"
                    f"  Tick directory: {data_path}\n"
                    f"\n"
                    f"Please check:\n"
                    f"  1. Tick data files exist in the specified directory\n"
                    f"  2. File naming convention matches expected format\n"
                    f"  3. Data path is correctly configured\n"
                    f"  4. Timestamps are within the available data range"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info(
                "Found %d tick file(s) for symbol %s in range [%s, %s]",
                len(tick_files),
                symbol,
                start_ts,
                end_ts,
            )

            tick_params = {
                "symbol": symbol,
                "tick_files": [str(Path(f)) for f in tick_files],
                "start_ts": start_ts,
                "end_ts": end_ts,
                "lookback_minutes": 60,
            }
            ticks_loader_json = serialize_tick_loader_params(tick_params)
            logger.info(
                "Successfully configured ticks_loader_json with %d file(s) for %s",
                len(tick_files),
                symbol,
            )
        except ValueError:
            # Re-raise ValueError (tick files not found)
            raise
        except Exception as e:
            error_msg = (
                f"Failed to configure VPIN ticks: {e}\n"
                f"  Symbol: {symbol}\n"
                f"  Data path: {data_path}\n"
                f"  Start timestamp: {start_ts}\n"
                f"  End timestamp: {end_ts}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    # 为所有需要 ticks 的特征设置 ticks_loader_json
    features_need_ticks = ["vpin_features", "footprint_basic"]
    configured_count = 0
    for feature_name in features_need_ticks:
        if feature_name in features_cfg:
            compute_params = features_cfg[feature_name].setdefault("compute_params", {})
            if not compute_params.get("ticks_loader_json"):
                compute_params["ticks_loader_json"] = ticks_loader_json
                configured_count += 1
                logger.debug(
                    "Configured ticks_loader_json for feature: %s", feature_name
                )

    if configured_count > 0:
        logger.info(
            "Successfully configured ticks_loader_json for %d feature(s): %s",
            configured_count,
            features_need_ticks,
        )
    else:
        logger.debug(
            "No features requiring ticks found in requested features, or already configured"
        )


def execute_single_run(
    strategy_cfg: StrategyConfig,
    df_train_raw: pd.DataFrame,
    df_test_raw: pd.DataFrame,
    test_warmup_bars: int = 0,
    variant_name: str = "unknown",
    symbol: Optional[str] = None,
    data_path: Optional[str] = None,
    calibrate_proba: str = "none",
    feature_loader: Optional[StrategyFeatureLoader] = None,
) -> Optional[Dict]:
    def _to_float_or_none(x: float) -> float | None:
        try:
            v = float(x)
        except Exception:
            return None
        return v if np.isfinite(v) else None

    def _build_pred_report(
        preds_arr: np.ndarray,
        y_arr: np.ndarray,
        backtest_params: Dict[str, Any],
        task_type: str,
        n_bins: int = 10,
    ) -> Dict[str, Any]:
        """
        Build a lightweight, JSON-safe report to answer:
        - Are predictions near-constant (e.g. ~0.3x)?
        - Do higher predictions actually imply higher realized win-rate?
        """
        preds_1d = np.asarray(preds_arr).reshape(-1)
        y_1d = np.asarray(y_arr).reshape(-1)

        valid = ~(np.isnan(preds_1d) | np.isnan(y_1d))
        pv = preds_1d[valid]
        yv = y_1d[valid]

        report: Dict[str, Any] = {
            "task_type": str(task_type),
            "n": int(len(preds_1d)),
            "n_valid": int(valid.sum()),
            "n_nan_pred": int(np.isnan(preds_1d).sum()),
            "n_nan_label": int(np.isnan(y_1d).sum()),
        }

        if len(pv) == 0:
            report["note"] = "No valid (pred,label) pairs to analyze."
            return report

        # Basic distribution stats
        qs = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
        qv = np.quantile(pv, qs)
        report["pred_stats"] = {
            "min": _to_float_or_none(pv.min()),
            "max": _to_float_or_none(pv.max()),
            "mean": _to_float_or_none(pv.mean()),
            "std": _to_float_or_none(pv.std()),
            "quantiles": {
                str(int(q * 100)): _to_float_or_none(v) for q, v in zip(qs, qv)
            },
        }

        # Label base-rate (binary only)
        if task_type == "binary":
            # y is expected to be 0/1; still tolerate floats
            y_bin = (yv >= 0.5).astype(int)
            report["label_stats"] = {
                "pos_rate": _to_float_or_none(y_bin.mean()),
                "pos": int(y_bin.sum()),
                "neg": int((1 - y_bin).sum()),
            }

        # Entry gating summary (for direction-fixed probability gating)
        try:
            entry_thr = backtest_params.get(
                "entry_threshold", backtest_params.get("long_entry_threshold", None)
            )
            entry_mode = str(backtest_params.get("entry_mode", "level")).lower()
            if entry_thr is not None:
                thr = float(entry_thr)
                entry_raw = pv >= thr
                if entry_mode == "cross":
                    prev = np.r_[False, entry_raw[:-1]]
                    entry = entry_raw & (~prev)
                else:
                    entry = entry_raw
                report["entry_gating"] = {
                    "entry_threshold": _to_float_or_none(thr),
                    "entry_mode": entry_mode,
                    "n_entry_raw": int(entry_raw.sum()),
                    "n_entry": int(entry.sum()),
                    "entry_rate_raw": _to_float_or_none(entry_raw.mean()),
                    "entry_rate": _to_float_or_none(entry.mean()),
                }
        except Exception:
            pass

        # Calibration by quantile bins (binary only)
        if task_type == "binary" and len(pv) >= max(n_bins, 2):
            y_bin = (yv >= 0.5).astype(int)
            # Guard: quantile edges can collapse if pv is near-constant
            edges = np.unique(np.quantile(pv, np.linspace(0, 1, n_bins + 1)))
            if len(edges) >= 3:
                # Make bins; include rightmost
                bin_ids = np.digitize(pv, edges[1:-1], right=True)
                bins = []
                for b in range(len(edges) - 1):
                    m = bin_ids == b
                    if not m.any():
                        continue
                    bins.append(
                        {
                            "bin": int(b),
                            "n": int(m.sum()),
                            "pred_mean": _to_float_or_none(pv[m].mean()),
                            "label_pos_rate": _to_float_or_none(y_bin[m].mean()),
                            "pred_min": _to_float_or_none(pv[m].min()),
                            "pred_max": _to_float_or_none(pv[m].max()),
                        }
                    )
                report["calibration_bins"] = bins
                # Brier score for a single-number sanity-check
                try:
                    report["brier"] = _to_float_or_none(
                        float(np.mean((pv - y_bin) ** 2))
                    )
                except Exception:
                    pass
            else:
                report["calibration_note"] = (
                    "Predictions near-constant; quantile bins collapsed."
                )

        return report

    if df_train_raw.empty or df_test_raw.empty:
        logger.error(
            "Variant %s has empty train/test split (train=%d, test=%d). "
            "This should not happen if data was loaded correctly. Check data loading logic.",
            variant_name,
            len(df_train_raw),
            len(df_test_raw),
        )
        return None

    logger.info(
        "Variant %s raw samples → train=%d, test=%d",
        variant_name,
        len(df_train_raw),
        len(df_test_raw),
    )

    # Use shared feature_loader if provided (for cross-variant memory cache reuse),
    # otherwise create a new one (backward compatibility).
    if feature_loader is None:
        feature_loader = StrategyFeatureLoader()

    # Configure tick loader if needed (for VPIN features)
    # This project is based on tick data, so tick files must be available
    if symbol and data_path and not df_train_raw.empty and not df_test_raw.empty:
        start_ts = str(df_train_raw.index.min())
        end_ts = str(df_test_raw.index.max())
        # This will raise an exception if tick files are not found
        _configure_vpin_ticks(feature_loader, symbol, data_path, start_ts, end_ts)

    df_train_features = strategy_runner.run_feature_pipeline(
        df_train_raw,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_cfg.features,
        fit=True,
    )
    df_test_features = strategy_runner.run_feature_pipeline(
        df_test_raw,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_cfg.features,
        fit=False,
    )

    # Debug: Check if OHLCV columns are preserved
    required_cols = ["high", "low", "open", "close"]
    missing_train = [c for c in required_cols if c not in df_train_features.columns]
    missing_test = [c for c in required_cols if c not in df_test_features.columns]
    if missing_train or missing_test:
        logger.warning(
            "Variant %s missing OHLCV columns after feature computation: train=%s, test=%s",
            variant_name,
            missing_train,
            missing_test,
        )

    if test_warmup_bars > 0 and len(df_test_features) > test_warmup_bars:
        df_test_features = df_test_features.iloc[test_warmup_bars:].copy()

    feature_cols = strategy_runner.determine_feature_columns(
        df_train_features, strategy_cfg.features
    )
    label_func = strategy_runner.import_callable(
        strategy_cfg.labels.generator.module, strategy_cfg.labels.generator.function
    )
    target_col = strategy_cfg.labels.target_column
    df_train_features[target_col] = label_func(
        df_train_features.copy(), **strategy_cfg.labels.generator.params
    )
    df_test_features[target_col] = label_func(
        df_test_features.copy(), **strategy_cfg.labels.generator.params
    )
    logger.info(
        "Variant %s labels computed → train targets=%d (NaN=%d) test targets=%d (NaN=%d)",
        variant_name,
        len(df_train_features),
        int(df_train_features[target_col].isna().sum()),
        len(df_test_features),
        int(df_test_features[target_col].isna().sum()),
    )

    df_train_filtered = strategy_runner.apply_filters(
        df_train_features, strategy_cfg.labels.filters
    )
    df_test_filtered = strategy_runner.apply_filters(
        df_test_features, strategy_cfg.labels.filters
    )
    logger.info(
        "Variant %s after label filters → train=%d, test=%d",
        variant_name,
        len(df_train_filtered),
        len(df_test_filtered),
    )

    # ------------------------------------------------------------------
    # Research-first NaN policy:
    # - Do NOT drop rows just because feature columns are NaN
    # - Instead: keep rows, warn when columns have high NaN ratio
    # - Ignore legacy post_label_filters entries with ensure_feature_non_null (too aggressive for research)
    #
    # We also ignore legacy post_label_filters entries with ensure_feature_non_null,
    # because it's too aggressive for research (it can collapse the test set to a few bars).
    # ------------------------------------------------------------------
    NAN_WARN_THRESHOLD = 0.20
    TOP_N = 20

    def _nan_ratio_report(
        df_in: pd.DataFrame, cols: List[str], top_n: int = 20
    ) -> Dict[str, Any]:
        if df_in is None or df_in.empty or not cols:
            return {
                "n_rows": int(len(df_in) if df_in is not None else 0),
                "warn_threshold": NAN_WARN_THRESHOLD,
                "top": [],
            }
        cols = [c for c in cols if c in df_in.columns]
        if not cols:
            return {
                "n_rows": int(len(df_in)),
                "warn_threshold": NAN_WARN_THRESHOLD,
                "top": [],
            }
        ratios = df_in[cols].isna().mean(axis=0).sort_values(ascending=False)
        warn = ratios[ratios >= NAN_WARN_THRESHOLD]
        top = [
            {"col": str(c), "nan_ratio": float(v), "nan_count": int(v * len(df_in))}
            for c, v in warn.head(top_n).items()
        ]
        return {
            "n_rows": int(len(df_in)),
            "warn_threshold": NAN_WARN_THRESHOLD,
            "top": top,
        }

    def _key_feature_cols(df_in: pd.DataFrame, cols: List[str]) -> List[str]:
        """
        Default key-feature heuristic:
        - key = columns with NaN ratio <= threshold on TRAIN
        This is reported for visibility only; we do NOT drop rows by default.
        """
        if df_in is None or df_in.empty:
            return []
        cols = [c for c in cols if c in df_in.columns]
        if not cols:
            return []
        ratios = df_in[cols].isna().mean(axis=0)
        key = [c for c, r in ratios.items() if float(r) <= NAN_WARN_THRESHOLD]
        return key

    nan_report_train = _nan_ratio_report(df_train_filtered, feature_cols, top_n=TOP_N)
    nan_report_test = _nan_ratio_report(df_test_filtered, feature_cols, top_n=TOP_N)

    key_cols = _key_feature_cols(df_train_filtered, feature_cols)

    # Apply post-label filters, but ignore ensure_feature_non_null in this command (research-first).
    post_filters = []
    for f in strategy_cfg.labels.post_label_filters or []:
        if isinstance(f, dict) and f.get("ensure_feature_non_null"):
            continue
        post_filters.append(f)
    df_train_filtered = strategy_runner.apply_post_label_filters(
        df_train_filtered, post_filters, feature_cols
    )
    df_test_filtered = strategy_runner.apply_post_label_filters(
        df_test_filtered, post_filters, feature_cols
    )

    logger.info(
        "Variant %s after post-label filters → train=%d, test=%d",
        variant_name,
        len(df_train_filtered),
        len(df_test_filtered),
    )

    if len(df_train_filtered) < 50 or len(df_test_filtered) < 10:
        logger.warning(
            "Variant %s skipped: insufficient samples after filters (train=%d, test=%d). "
            "Raw data: train=%d, test=%d. After label filters: train=%d, test=%d. "
            "This may indicate feature computation failures or overly aggressive filters.",
            variant_name,
            len(df_train_filtered),
            len(df_test_filtered),
            len(df_train_raw),
            len(df_test_raw),
            len(df_train_features),
            len(df_test_features),
        )
        return None

    trainer_func = strategy_runner.import_callable(
        strategy_cfg.model.trainer.module, strategy_cfg.model.trainer.function
    )
    trainer_params = dict(strategy_cfg.model.trainer.params)
    target_col = trainer_params.pop("target_col", target_col)
    model_type = trainer_params.get("model_type", "xgboost")
    task_type = trainer_params.get("task_type", "regression")

    models, avg_metric, cv_results, used_features, preprocessor = trainer_func(
        df_train_filtered,
        feature_cols=feature_cols,
        target_col=target_col,
        **trainer_params,
    )

    X_test = preprocessor.transform(df_test_filtered, feature_cols=used_features)
    y_test = df_test_filtered[target_col].values

    preds = strategy_runner.generate_predictions(
        models=models,
        model_type=model_type,
        task_type=task_type,
        X=X_test,
    )

    # Optional probability calibration (binary only)
    if calibrate_proba != "none" and task_type == "binary":
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.linear_model import LogisticRegression

            # Fit calibrator on train set predictions vs. labels (held-out CV inside calibrator).
            X_train_cal = preprocessor.transform(
                df_train_filtered, feature_cols=used_features
            )
            y_train_cal = df_train_filtered[target_col].values

            # Build a lightweight estimator wrapper using the first model if possible
            base_est = None
            if model_type == "xgboost":
                import xgboost as xgb

                # XGBoost: wrap Booster in XGBClassifier
                if models and len(models) > 0:
                    base_est = xgb.XGBClassifier()
                    base_est._Booster = models[0]
            elif model_type == "catboost":
                # CatBoost models are already sklearn-compatible
                base_est = models[0] if models else None
            elif model_type == "lightgbm":
                import lightgbm as lgb

                # LightGBM: wrap Booster in LGBMClassifier
                if models and len(models) > 0:
                    base_est = lgb.LGBMClassifier()
                    base_est._Booster = models[0]

            if base_est is not None:
                method = "isotonic" if calibrate_proba == "isotonic" else "sigmoid"
                calibrator = CalibratedClassifierCV(base_est, method=method, cv=3)
                calibrator.fit(X_train_cal, y_train_cal)
                preds = calibrator.predict_proba(X_test)[:, 1]
                logger.info(
                    f"Applied {calibrate_proba} calibration (method={method}) to test predictions"
                )
        except Exception as exc:
            logger.warning(f"Calibration skipped: {exc}")

    evaluation_results = strategy_runner.evaluate_predictions(
        preds, y_test, strategy_cfg.evaluation
    )

    # Add AUC (binary) and Rank IC (regression) for model validation
    additional_metrics = {}
    if task_type == "binary":
        # AUC for binary classification
        try:
            from sklearn.metrics import roc_auc_score

            # Filter out NaN labels/predictions
            valid_mask = ~(np.isnan(preds) | np.isnan(y_test))
            if valid_mask.sum() > 0:
                auc = roc_auc_score(y_test[valid_mask], preds[valid_mask])
                additional_metrics["auc"] = float(auc)
        except Exception as exc:
            logger.warning(f"AUC calculation skipped: {exc}")
    elif task_type == "regression":
        # Rank IC (Spearman correlation) for regression - especially useful for volatility regression
        try:
            from scipy.stats import spearmanr

            # Filter out NaN labels/predictions
            valid_mask = ~(np.isnan(preds) | np.isnan(y_test))
            if valid_mask.sum() > 1:
                rank_ic, _ = spearmanr(
                    preds[valid_mask], y_test[valid_mask], nan_policy="omit"
                )
                additional_metrics["rank_ic"] = (
                    float(rank_ic) if not np.isnan(rank_ic) else 0.0
                )
        except Exception as exc:
            logger.warning(f"Rank IC calculation skipped: {exc}")

    # Merge additional metrics into evaluation_results
    evaluation_results.update(additional_metrics)

    # Prediction report (distribution + calibration) — stored in diagnostics (not evaluation)
    try:
        bt_params = (
            (strategy_cfg.backtest.params or {}) if strategy_cfg.backtest else {}
        )
        pred_report = _build_pred_report(
            preds_arr=preds,
            y_arr=y_test,
            backtest_params=bt_params,
            task_type=task_type,
            n_bins=10,
        )
    except Exception as exc:  # noqa: BLE001
        pred_report = {"error": f"pred_report_failed: {exc}"}

    # Feature direction / inversion report — provided by trainer/preprocessor (config-driven).
    factor_direction: Dict[str, Any] = {
        "enabled": False,
        "source": "none",
        "inverted": [],
    }
    try:
        multipliers = getattr(preprocessor, "feature_multipliers", None)
        if isinstance(multipliers, dict):
            inverted = sorted([k for k, v in multipliers.items() if float(v) == -1.0])
            factor_direction = {
                "enabled": bool(inverted),
                "source": "trainer",
                "inverted": inverted,
            }
    except Exception:
        pass

    backtest_results = strategy_runner.run_vectorbt_backtest(
        df_test_filtered,
        preds,
        strategy_cfg.backtest,
        task_type,
        strategy_config=strategy_cfg,
    )

    logger.info(
        "Variant %s finished training with %d features, CV metric %.4f",
        variant_name,
        len(used_features),
        float(avg_metric) if avg_metric is not None else float("nan"),
    )

    # Feature pipeline debug stats (per-feature index mismatch etc.)
    feature_debug = {
        "train": (df_train_features.attrs.get("feature_debug_stats") or {}),
        "test": (df_test_features.attrs.get("feature_debug_stats") or {}),
    }

    return {
        "avg_cv_metric": float(avg_metric),
        "evaluation": evaluation_results,
        "backtest": backtest_results,
        "used_features": used_features,
        "n_train": int(len(df_train_filtered)),
        "n_test": int(len(df_test_filtered)),
        "diagnostics": {
            "nan_report": {
                "train": nan_report_train,
                "test": nan_report_test,
                "key_feature_policy": {
                    "threshold": NAN_WARN_THRESHOLD,
                    "n_key_features": int(len(key_cols)),
                    "note": "Key features are reported for visibility; rows are not dropped by default.",
                },
            },
            "feature_debug_stats": feature_debug,
            "pred_report": pred_report,
            "factor_direction": factor_direction,
        },
    }


def run_rolling_evaluation(
    strategy_cfg: StrategyConfig,
    df_raw: pd.DataFrame,
    params: argparse.Namespace,
    variant_name: str = "unknown",
    feature_loader: Optional[StrategyFeatureLoader] = None,
) -> Optional[Dict]:
    train_size = params.rolling_train_bars
    test_size = params.rolling_test_bars
    step = params.rolling_step_bars
    max_windows = params.rolling_max_windows

    windows: List[Dict] = []
    start = 0
    while start + train_size + test_size <= len(df_raw) and len(windows) < max_windows:
        train_raw = df_raw.iloc[start : start + train_size].copy()
        test_raw = df_raw.iloc[
            start + train_size : start + train_size + test_size
        ].copy()
        result = execute_single_run(
            strategy_cfg,
            train_raw,
            test_raw,
            variant_name=variant_name,
            symbol=params.symbol,
            data_path=params.data_path,
            calibrate_proba=getattr(params, "calibrate_proba", "none"),
            feature_loader=feature_loader,
        )
        if result:
            result["window_start"] = str(train_raw.index[0])
            result["window_end"] = str(test_raw.index[-1])
            windows.append(result)
        start += step

    if not windows:
        return None

    eval_keys = sorted({k for w in windows for k in w["evaluation"].keys()})
    aggregate_eval = {
        key: float(np.nanmean([w["evaluation"].get(key, np.nan) for w in windows]))
        for key in eval_keys
    }
    if any(w.get("backtest") for w in windows):
        # Only aggregate scalar numeric backtest metrics; skip nested/non-numeric fields (e.g. debug dict)
        def _to_float_or_nan(x: object) -> float:
            try:
                v = float(x)  # type: ignore[arg-type]
            except Exception:
                return float("nan")
            return v if np.isfinite(v) else float("nan")

        bt_keys = sorted(
            {
                k
                for w in windows
                if w.get("backtest") and isinstance(w.get("backtest"), dict)
                for k in w["backtest"].keys()
                if k != "debug"
            }
        )
        aggregate_bt = {}
        for key in bt_keys:
            vals = []
            for w in windows:
                bt = w.get("backtest")
                if not isinstance(bt, dict):
                    continue
                vals.append(_to_float_or_nan(bt.get(key, np.nan)))
            aggregate_bt[key] = float(np.nanmean(vals)) if vals else float("nan")
    else:
        aggregate_bt = None

    avg_cv = float(np.nanmean([w["avg_cv_metric"] for w in windows]))
    return {
        "windows": windows,
        "aggregate": {
            "avg_cv_metric": avg_cv,
            "evaluation": aggregate_eval,
            "backtest": aggregate_bt,
            "n_windows": len(windows),
        },
    }


def main() -> None:
    args = parse_args()
    base_dir = Path(args.strategy_config).resolve()
    variants, temp_dirs = build_variants(base_dir, args.feature_overrides)

    logger.info(
        "Loading data for %s [%s] from %s (%s → %s)",
        args.symbol,
        args.timeframe,
        args.data_path,
        args.start_date or "beginning",
        args.end_date or "latest",
    )

    df_raw = load_raw_data(
        data_path=args.data_path,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
    )
    logger.info("Loaded %d bars for %s", len(df_raw), args.symbol)
    split_idx = int(len(df_raw) * (1 - args.test_size))
    df_train_raw = df_raw.iloc[:split_idx].copy()
    test_warmup = min(args.test_warmup_bars, len(df_train_raw))
    df_test_raw = df_raw.iloc[split_idx - test_warmup :].copy()
    logger.info(
        "Split data → train=%d (%.1f%%) test=%d (%.1f%%)",
        len(df_train_raw),
        100 * (1 - args.test_size),
        len(df_test_raw),
        100 * args.test_size,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Report ranges (best-effort)
    def _range_str(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
        try:
            if df is None or df.empty:
                return None
            start = df.index[0]
            end = df.index[-1]
            return (str(start), str(end))
        except Exception:
            return None

    train_range = _range_str(df_train_raw)
    test_range = _range_str(df_test_raw)

    # Create a shared StrategyFeatureLoader to enable cross-variant memory cache reuse.
    # This significantly speeds up multi-variant comparisons when features overlap.
    shared_feature_loader = StrategyFeatureLoader()
    logger.info(
        "Created shared StrategyFeatureLoader for cross-variant memory cache reuse"
    )

    comparison_results = []
    try:
        for variant in variants:
            try:
                loader = StrategyConfigLoader(variant.config_dir)
                strategy_cfg = loader.load()
                logger.info("Running variant %s ...", variant.name)
                base_result = execute_single_run(
                    strategy_cfg,
                    df_train_raw,
                    df_test_raw,
                    test_warmup_bars=test_warmup,
                    variant_name=variant.name,
                    symbol=args.symbol,
                    data_path=args.data_path,
                    calibrate_proba=args.calibrate_proba,
                    feature_loader=shared_feature_loader,
                )
            except Exception as exc:
                logger.error(
                    "Variant %s failed with exception: %s",
                    variant.name,
                    exc,
                    exc_info=True,
                )
                base_result = None
            rolling_result = None
            if args.run_rolling:
                logger.info(
                    "Starting rolling evaluation for variant %s (%d windows max)",
                    variant.name,
                    args.rolling_max_windows,
                )
                rolling_result = run_rolling_evaluation(
                    strategy_cfg,
                    df_raw,
                    args,
                    variant_name=variant.name,
                    feature_loader=shared_feature_loader,
                )
            comparison_results.append(
                {
                    "variant": variant.name,
                    "base": base_result or {},
                    "rolling": rolling_result or {},
                }
            )
    finally:
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)

    paths = write_strategy_feature_compare_reports(
        comparison_results=comparison_results,
        symbol=args.symbol,
        timeframe=args.timeframe,
        test_size=args.test_size,
        start_date=args.start_date,
        end_date=args.end_date,
        train_range=train_range,
        test_range=test_range,
        test_warmup_bars=int(test_warmup),
        output_dir=output_dir,
        base_variant="base",
    )
    print(f"✅ Saved summary CSV to {paths['summary_csv']}")
    print(f"✅ Saved summary+ CSV to {paths['summary_plus_csv']}")
    if paths.get("rolling_windows_csv") and paths["rolling_windows_csv"].exists():
        print(f"✅ Saved rolling windows CSV to {paths['rolling_windows_csv']}")
    if paths.get("rolling_monthly_csv") and paths["rolling_monthly_csv"].exists():
        print(f"✅ Saved rolling monthly CSV to {paths['rolling_monthly_csv']}")
    print(f"✅ Saved summary JSON to {paths['detailed_json']}")
    print(f"✅ Saved HTML report to {paths['html_report']}")


if __name__ == "__main__":
    main()
