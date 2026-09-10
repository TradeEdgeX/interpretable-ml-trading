#!/usr/bin/env python3
"""
Config-driven rolling training for time-series strategies.

This script performs expanding window rolling training, where each test month
uses all previous months for training. It supports all four strategies via
config-driven approach.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path (repo root)
# __file__ = src/time_series_model/pipeline/rolling/rolling_train.py
# parents: [rolling, pipeline, time_series_model, src, repo_root]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.data_utils import load_raw_data
from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
from src.time_series_model.strategy_config import StrategyConfigLoader

# Import shared functions from train_strategy_pipeline
from scripts.train_strategy_pipeline import (
    BASE_DATA_COLUMNS,
    apply_filters,
    apply_post_label_filters,
    determine_feature_columns,
    evaluate_predictions,
    generate_predictions,
    import_callable,
    run_feature_pipeline,
    run_vectorbt_backtest,
)


def find_monthly_files(data_dir: str, symbol: str) -> List[Dict[str, Any]]:
    """Find all monthly data files for a symbol, sorted chronologically."""
    files = []
    data_path = Path(data_dir)

    if not data_path.exists():
        return files

    patterns = [
        f"{symbol}-aggTrades-*.parquet",
        f"{symbol}-aggTrades-*.zip",
        f"{symbol}-*.parquet",
        f"{symbol}-*.zip",
        f"{symbol}_*.parquet",  # 添加下划线格式（如 BTCUSDT_2024-01.parquet）
        f"{symbol}_*.zip",
    ]

    symbol_mapping = {
        "BTCUSDT": "BTC-USD",
        "ETHUSDT": "ETH-USD",
        "BNBUSDT": "BNB-USD",
        "ADAUSDT": "ADA-USD",
        "SOLUSDT": "SOL-USD",
    }
    file_symbol = symbol_mapping.get(symbol, symbol)

    for pattern in patterns:
        for file_path in data_path.glob(pattern):
            stem = file_path.stem
            date_patterns = [
                rf"{re.escape(symbol)}-aggTrades-(?P<year>\d{{4}})-(?P<month>\d{{2}})",
                rf"{re.escape(symbol)}_(?P<year>\d{{4}})-(?P<month>\d{{2}})",  # 添加原始 symbol 的下划线格式
                rf"{re.escape(file_symbol)}_(?P<year>\d{{4}})-(?P<month>\d{{2}})",
                rf"{re.escape(file_symbol)}-(?P<year>\d{{4}})-(?P<month>\d{{2}})",
                rf"(?P<year>\d{{4}})-(?P<month>\d{{2}})",
            ]

            match = None
            for pattern_re in date_patterns:
                match = re.search(pattern_re, stem)
                if match:
                    break

            if match:
                try:
                    year = int(match.group("year"))
                    month = int(match.group("month"))
                    files.append(
                        {
                            "path": str(file_path),
                            "year": year,
                            "month": month,
                            "month_str": f"{year}-{month:02d}",
                            "timestamp": pd.Timestamp(year, month, 1),
                        }
                    )
                except (ValueError, KeyError):
                    continue

    files.sort(key=lambda x: x["timestamp"])
    return files


def load_monthly_data(file_path: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Load a single monthly data file."""
    try:
        path = Path(file_path)
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".zip":
            import zipfile
            import tempfile

            temp_dir = Path(tempfile.gettempdir()) / f"rolling_{path.stem}"
            temp_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(path, "r") as archive:
                archive.extractall(temp_dir)
            csv_files = list(temp_dir.glob("*.csv"))
            if not csv_files:
                return None
            df = pd.read_csv(csv_files[0])
            if "transact_time" in df.columns:
                df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms")
            elif "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            else:
                return None
            df.set_index("timestamp", inplace=True)
        else:
            return None

        # Resample to timeframe if needed
        if timeframe:
            df = df.resample(timeframe).agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            df = df.dropna()

        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            return None

        return df
    except Exception as e:
        print(f"   ⚠️  Error loading {file_path}: {e}")
        return None


def train_single_month(
    config_dir: Path,
    train_files: List[Dict[str, Any]],
    test_file: Dict[str, Any],
    feature_loader: StrategyFeatureLoader,
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    """Train model for a single test month using expanding window."""
    loader = StrategyConfigLoader(config_dir)
    strategy_config = loader.load()

    print(f"\n{'=' * 80}")
    print(f"📂 Strategy: {strategy_config.name}")
    print(
        f"   Train: {train_files[0]['month_str']} to {train_files[-1]['month_str']} ({len(train_files)} months)"
    )
    print(f"   Test:  {test_file['month_str']}")

    # Load training data
    print(f"\n1. Loading training data...")
    train_data = []
    for file_info in train_files:
        print(f"   Loading {file_info['month_str']}")
        df = load_monthly_data(file_info["path"], args.timeframe)
        if df is not None and len(df) > 0:
            train_data.append(df)

    if not train_data:
        print("   ❌ No training data!")
        return None

    train_df_raw = pd.concat(train_data, axis=0).sort_index()
    print(f"   ✅ Training data: {len(train_df_raw):,} bars")

    # Load test data
    print(f"\n2. Loading test data...")
    test_df_raw = load_monthly_data(test_file["path"], args.timeframe)
    if test_df_raw is None or len(test_df_raw) == 0:
        print("   ❌ No test data!")
        return None
    print(f"   ✅ Test data: {len(test_df_raw):,} bars")

    # Engineer features
    print(f"\n3. Engineering features...")
    df_train_features = run_feature_pipeline(
        train_df_raw,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_config.features,
        fit=True,
    )
    df_test_features = run_feature_pipeline(
        test_df_raw,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_config.features,
        fit=False,
    )

    feature_cols = determine_feature_columns(
        df_train_features, strategy_config.features
    )
    print(f"   ✅ Features: {len(feature_cols)}")

    # Generate labels
    print(f"\n4. Generating labels...")
    label_func = import_callable(
        strategy_config.labels.generator.module,
        strategy_config.labels.generator.function,
    )

    df_train_features[strategy_config.labels.target_column] = label_func(
        df_train_features.copy(), **strategy_config.labels.generator.params
    )
    df_test_features[strategy_config.labels.target_column] = label_func(
        df_test_features.copy(), **strategy_config.labels.generator.params
    )

    # Apply filters
    df_train_filtered = apply_filters(df_train_features, strategy_config.labels.filters)
    df_test_filtered = apply_filters(df_test_features, strategy_config.labels.filters)

    df_train_filtered = apply_post_label_filters(
        df_train_filtered,
        strategy_config.labels.post_label_filters,
        feature_cols,
    )
    df_test_filtered = apply_post_label_filters(
        df_test_filtered,
        strategy_config.labels.post_label_filters,
        feature_cols,
    )

    print(
        f"   ✅ Valid samples - Train: {len(df_train_filtered)}, Test: {len(df_test_filtered)}"
    )
    if len(df_train_filtered) < 50:
        print("   ⚠️  Not enough training samples, skipping.")
        return None

    # Train model
    print(f"\n5. Training model...")
    trainer_func = import_callable(
        strategy_config.model.trainer.module,
        strategy_config.model.trainer.function,
    )
    trainer_params = dict(strategy_config.model.trainer.params)
    target_col = trainer_params.pop("target_col", strategy_config.labels.target_column)
    model_type = trainer_params.get("model_type", "xgboost")
    task_type = trainer_params.get("task_type", "regression")

    models, avg_metric, cv_results, used_features, preprocessor = trainer_func(
        df_train_filtered,
        feature_cols=feature_cols,
        target_col=target_col,
        **trainer_params,
    )

    print(f"   ✅ Average CV Metric: {avg_metric:.4f}")

    # Evaluate on test set
    print(f"\n6. Evaluating on test set...")
    X_test = preprocessor.transform(df_test_filtered, feature_cols=used_features)
    y_test = df_test_filtered[target_col].values

    preds = generate_predictions(
        models=models,
        model_type=model_type,
        task_type=task_type,
        X=X_test,
    )

    evaluation_results = evaluate_predictions(
        preds,
        y_test,
        strategy_config.evaluation,
    )

    for metric_name, score in evaluation_results.items():
        print(f"   ✅ {metric_name}: {score:.4f}")

    # Run backtest
    backtest_results = None
    if strategy_config.backtest.enabled:
        print(f"\n7. Running backtest...")
        backtest_results = run_vectorbt_backtest(
            df_test_filtered,
            preds,
            strategy_config.backtest,
            task_type,
        )
        if backtest_results:
            print(
                f"   ✅ Total Return: {backtest_results.get('total_return_pct', 0):.2f}%"
            )
            print(f"   ✅ Sharpe: {backtest_results.get('sharpe', 0):.4f}")

    # Save model and preprocessor
    output_dir = Path(args.output_root) / strategy_config.name / test_file["month_str"]
    output_dir.mkdir(parents=True, exist_ok=True)

    import joblib

    model_path = output_dir / "model.pkl"
    joblib.dump(models, model_path)
    print(f"   ✅ Model saved to {model_path}")

    preprocessor_path = output_dir / "preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"   ✅ Preprocessor saved to {preprocessor_path}")

    # Optionally save as ModelArtifact (unified format)
    try:
        from src.time_series_model.strategies.models.model_artifact import ModelArtifact

        artifact = ModelArtifact(
            model=models,
            preprocessor=preprocessor,
            used_features=used_features,
            feature_config=(
                strategy_config.features.__dict__
                if hasattr(strategy_config.features, "__dict__")
                else None
            ),
            metadata={
                "strategy": strategy_config.name,
                "model_type": model_type,
                "task_type": task_type,
                "test_month": test_file["month_str"],
                "train_months": [f["month_str"] for f in train_files],
                "cv_metric": float(avg_metric),
            },
        )
        artifact.save(output_dir)
        print(f"   ✅ ModelArtifact saved (unified format)")
    except Exception as exc:  # noqa: BLE001
        # Fallback: continue with individual saves if ModelArtifact fails
        print(f"   ⚠️  ModelArtifact save failed (using individual saves): {exc}")

    return {
        "test_month": test_file["month_str"],
        "train_months": [f["month_str"] for f in train_files],
        "strategy": strategy_config.name,
        "cv_metric": float(avg_metric),
        "evaluation": evaluation_results,
        "backtest": backtest_results,
        "train_samples": len(df_train_filtered),
        "test_samples": len(df_test_filtered),
        "features_used": len(used_features),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Config-driven rolling training for strategies"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to strategy config directory (e.g., config/strategies/sr_reversal_long)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Trading symbol (e.g., BTCUSDT)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/parquet_data",
        help="Directory containing monthly data files",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="15T",
        help="Data timeframe (e.g., 15T)",
    )
    parser.add_argument(
        "--initial-train-months",
        type=int,
        default=6,
        help="Initial training months before first test",
    )
    parser.add_argument(
        "--min-train-months",
        type=int,
        default=3,
        help="Minimum training months required",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/rolling",
        help="Root directory for output",
    )
    parser.add_argument(
        "--update-only",
        action="store_true",
        help="Only update from last trained month",
    )
    args = parser.parse_args()

    config_dir = Path(args.config)
    if not (config_dir / "features.yaml").exists():
        print(f"❌ Config directory not found or invalid: {config_dir}")
        sys.exit(1)

    # Find all monthly files
    print("🔍 Finding monthly data files...")
    all_files = find_monthly_files(args.data_dir, args.symbol)
    if not all_files:
        print(f"❌ No monthly files found for {args.symbol} in {args.data_dir}")
        sys.exit(1)

    # Filter by date range if specified
    if args.start:
        start_ts = pd.Timestamp(args.start)
        all_files = [f for f in all_files if f["timestamp"] >= start_ts]
    if args.end:
        end_ts = pd.Timestamp(args.end)
        all_files = [f for f in all_files if f["timestamp"] <= end_ts]

    print(f"   ✅ Found {len(all_files)} monthly files")
    print(f"   Range: {all_files[0]['month_str']} to {all_files[-1]['month_str']}")

    # Load existing results if update-only
    existing_results = []
    if args.update_only:
        output_dir = Path(args.output_root) / config_dir.name
        results_file = output_dir / "monthly_results.json"
        if results_file.exists():
            with open(results_file) as f:
                existing_results = json.load(f)
            print(f"   📂 Loaded {len(existing_results)} existing results")

    # Initialize feature loader
    feature_loader = StrategyFeatureLoader()

    # Determine starting point
    start_idx = args.initial_train_months
    if args.update_only and existing_results:
        last_trained = existing_results[-1].get("test_month")
        for idx, f in enumerate(all_files):
            if f["month_str"] == last_trained:
                start_idx = idx + 1
                break

    # Rolling training loop
    all_results = []
    print(f"\n{'=' * 80}")
    print(f"🔄 Starting Rolling Training")
    print(f"{'=' * 80}\n")

    for i in range(start_idx, len(all_files)):
        train_files = all_files[:i]
        test_file = all_files[i]

        if len(train_files) < args.min_train_months:
            print(
                f"⚠️  Skipping {test_file['month_str']}: insufficient training data ({len(train_files)} < {args.min_train_months})"
            )
            continue

        result = train_single_month(
            config_dir,
            train_files,
            test_file,
            feature_loader,
            args,
        )
        if result:
            all_results.append(result)

    # Save summary
    if all_results:
        output_dir = Path(args.output_root) / config_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        results_file = output_dir / "monthly_results.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'=' * 80}")
        print(f"✅ Rolling training complete!")
        print(f"   Results saved to {results_file}")
        print(f"   Total months trained: {len(all_results)}")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
