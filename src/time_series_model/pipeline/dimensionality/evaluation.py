"""Model evaluation utilities for dimensionality comparison."""

from __future__ import annotations

from typing import Dict, Optional, Tuple
from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from time_series_model.backtesting.vectorbot import (
    calculate_strategy_returns_from_predictions,
    calculate_financial_metrics_from_returns,
)


def sanitize_features(X: np.ndarray, clip_std: float = 5.0) -> np.ndarray:
    """Replace NaN/inf and clip outliers per feature to stabilize AE training."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # Clip per-column
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0) + 1e-8
    lower = means - clip_std * stds
    upper = means + clip_std * stds
    X = np.minimum(np.maximum(X, lower), upper)
    # Ensure finite again
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def _generate_shap_outputs(
    model: lgb.Booster,
    X: np.ndarray,
    feature_names: list[str],
    output_dir: str,
    prefix: str = "stage3",
    sample_size: int = 2000,
) -> Optional[str]:
    """Generate SHAP explainability artifacts for the provided LightGBM model."""
    try:
        import shap
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"   ⚠️ SHAP not available ({exc}); skipping SHAP analysis.")
        return None

    if X.size == 0 or len(feature_names) == 0:
        print("   ⚠️ SHAP skipped: no data or feature names available.")
        return None

    sample_size = int(min(sample_size, X.shape[0]))
    if sample_size <= 0:
        print("   ⚠️ SHAP skipped: insufficient samples.")
        return None

    rng = np.random.default_rng(42)
    sample_indices = rng.choice(X.shape[0], size=sample_size, replace=False)
    X_sample = X[sample_indices]

    # Check if model predictions are constant (would cause SHAP=0)
    model_info = {}
    is_constant_output = False

    try:
        predictions_sample = model.predict(X_sample)
        pred_std = np.std(predictions_sample)
        pred_mean = np.mean(predictions_sample)
        pred_min = np.min(predictions_sample)
        pred_max = np.max(predictions_sample)

        # Get model training info if available
        best_iteration = getattr(model, "best_iteration", None)
        current_iteration = getattr(model, "current_iteration", lambda: None)()
        if best_iteration is None and current_iteration is not None:
            best_iteration = current_iteration

        model_info = {
            "prediction_mean": float(pred_mean),
            "prediction_std": float(pred_std),
            "prediction_min": float(pred_min),
            "prediction_max": float(pred_max),
            "best_iteration": (
                int(best_iteration) if best_iteration is not None else None
            ),
            "sample_size": int(sample_size),
        }

        if pred_std < 1e-10:
            is_constant_output = True
            print(f"\n   {'='*70}")
            print(f"   ⚠️  CRITICAL WARNING: Model predictions are constant!")
            print(f"   {'='*70}")
            print(f"   Prediction statistics:")
            print(f"      Mean: {pred_mean:.6f}")
            print(f"      Std:  {pred_std:.2e} (should be > 0.01)")
            print(f"      Min:  {pred_min:.6f}")
            print(f"      Max:  {pred_max:.6f}")
            print(f"   Model training info:")
            print(f"      Best iteration: {best_iteration}")
            print(f"   ")
            print(f"   This means the model outputs the same value for all inputs.")
            print(f"   SHAP values will be 0 because there's no variation to explain.")
            print(f"   ")
            print(f"   Possible causes:")
            print(f"   1. ❌ Model only predicts one class (e.g., always predicts 1.0)")
            print(f"      → Check model training logs for 'best_iteration=1'")
            print(f"   2. ❌ Model didn't train properly (early stop at iteration 1)")
            print(f"      → Check training loss/validation loss curves")
            print(
                f"   3. ❌ Features don't contain enough information to distinguish classes"
            )
            print(f"      → Check feature-label correlation (should be > 0.1)")
            print(f"   4. ❌ Class imbalance too severe")
            print(f"      → Check label distribution (should be roughly balanced)")
            print(f"   ")
            print(f"   Diagnostic steps:")
            print(f"   - Check model training logs for warnings")
            print(f"   - Verify feature quality (feature-label correlation)")
            print(f"   - Check label distribution (should have both classes)")
            print(
                f"   - Review model parameters (min_data_in_leaf, learning_rate, etc.)"
            )
            print(f"   {'='*70}\n")

            # Still generate SHAP values but mark them as invalid
            # This allows the pipeline to continue while documenting the issue
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            except Exception as exc:
                print(f"   ⚠️ SHAP computation failed: {exc}")
                return None

            # Handle SHAP values for different task types
            # For multiclass, SHAP returns a list of arrays (one per class)
            # For binary, SHAP returns a list of 2 arrays or a single array
            # For regression, SHAP returns a single array
            if isinstance(shap_values, list):
                if len(shap_values) == 1:
                    # Single array (regression or binary with single output)
                    shap_array = shap_values[0]
                elif len(shap_values) == 2:
                    # Binary classification: use the positive class (class 1)
                    shap_array = shap_values[1]
                elif len(shap_values) >= 3:
                    # Multiclass: compute mean absolute SHAP across all classes
                    # This gives overall feature importance across all classes
                    # Alternative: use the most important class or weighted average
                    shap_array = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                    print(
                        f"   📊 Multiclass SHAP: Using mean absolute SHAP across {len(shap_values)} classes"
                    )
                else:
                    # Fallback: use the last array
                    shap_array = shap_values[-1]
            else:
                # Single array (regression or binary)
                shap_array = shap_values

            # Verify SHAP values are indeed all zeros
            shap_std = np.std(shap_array)
            shap_mean = np.mean(np.abs(shap_array))
            if shap_std < 1e-10:
                print(f"   ⚠️  Confirmed: All SHAP values are zero!")
                print(f"      SHAP std: {shap_std:.2e}")
                print(f"      Mean absolute SHAP: {shap_mean:.2e}")
                print(
                    f"      This confirms the model output is constant and cannot be explained."
                )
                model_info["shap_all_zero"] = True
                model_info["shap_std"] = float(shap_std)
                model_info["shap_mean_abs"] = float(shap_mean)
            else:
                model_info["shap_all_zero"] = False
                model_info["shap_std"] = float(shap_std)
                model_info["shap_mean_abs"] = float(shap_mean)
        else:
            # Normal case: predictions have variation
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            except Exception as exc:
                print(f"   ⚠️ SHAP computation failed: {exc}")
                return None

            # Handle SHAP values for different task types
            # For multiclass, SHAP returns a list of arrays (one per class)
            # For binary, SHAP returns a list of 2 arrays or a single array
            # For regression, SHAP returns a single array
            if isinstance(shap_values, list):
                if len(shap_values) == 1:
                    # Single array (regression or binary with single output)
                    shap_array = shap_values[0]
                elif len(shap_values) == 2:
                    # Binary classification: use the positive class (class 1)
                    shap_array = shap_values[1]
                elif len(shap_values) >= 3:
                    # Multiclass: compute mean absolute SHAP across all classes
                    # This gives overall feature importance across all classes
                    # Alternative: use the most important class or weighted average
                    shap_array = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                    print(
                        f"   📊 Multiclass SHAP: Using mean absolute SHAP across {len(shap_values)} classes"
                    )
                else:
                    # Fallback: use the last array
                    shap_array = shap_values[-1]
            else:
                # Single array (regression or binary)
                shap_array = shap_values
    except Exception as exc:
        print(f"   ⚠️ SHAP computation failed: {exc}")
        return None

    shap_dir = Path(output_dir) / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    try:
        shap.summary_plot(
            shap_array,
            X_sample,
            feature_names=feature_names,
            show=False,
            plot_type="bar",
            color_bar=True,
        )
        plt.tight_layout()
        plt.savefig(shap_dir / f"{prefix}_summary_bar.png", dpi=200)
        plt.close()

        shap.summary_plot(
            shap_array,
            X_sample,
            feature_names=feature_names,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(shap_dir / f"{prefix}_summary_beeswarm.png", dpi=200)
        plt.close()
    except Exception as exc:
        print(f"   ⚠️ Failed to render SHAP plots: {exc}")

    mean_abs_shap = np.abs(shap_array).mean(axis=0)

    # Check if all SHAP values are zero
    all_shap_zero = np.all(mean_abs_shap < 1e-10) or is_constant_output

    if all_shap_zero:
        print(f"   ⚠️  WARNING: All SHAP values are zero!")
        print(
            f"      This indicates the model output is constant and cannot be explained."
        )
        print(f"      SHAP importance ranking will be meaningless.")

        # Create enhanced ranking with diagnostic info
        shap_ranking = sorted(
            [
                {
                    "feature": feat,
                    "mean_abs_shap": float(val),
                    "rank": idx + 1,
                    "warning": "Model output is constant - SHAP values are invalid",
                    "diagnostic": {
                        "reason": "Model predictions are constant (all outputs identical)",
                        "impact": "SHAP values cannot be computed meaningfully",
                        "recommendation": "Fix model training issue before interpreting SHAP values",
                    },
                }
                for idx, (feat, val) in enumerate(
                    sorted(
                        zip(feature_names, mean_abs_shap),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )
                )
            ],
            key=lambda item: item["rank"],
        )

        # Add metadata at the top level
        shap_metadata = {
            "status": "invalid",
            "reason": "model_output_constant",
            "model_info": model_info,
            "warning": "All SHAP values are zero because model output is constant. This indicates a model training problem.",
            "features": shap_ranking,
        }
    else:
        shap_ranking = sorted(
            [
                {
                    "feature": feat,
                    "mean_abs_shap": float(val),
                    "rank": idx + 1,
                }
                for idx, (feat, val) in enumerate(
                    sorted(
                        zip(feature_names, mean_abs_shap),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )
                )
            ],
            key=lambda item: item["rank"],
        )

        # Add metadata for valid SHAP values
        shap_metadata = {
            "status": "valid",
            "model_info": model_info,
            "features": shap_ranking,
        }

    # Save both formats: legacy (features only) and enhanced (with metadata)
    with open(shap_dir / f"{prefix}_shap_importance.json", "w", encoding="utf-8") as f:
        json.dump(shap_ranking, f, indent=2)

    # Save enhanced version with metadata
    with open(
        shap_dir / f"{prefix}_shap_importance_enhanced.json", "w", encoding="utf-8"
    ) as f:
        json.dump(shap_metadata, f, indent=2)

    if all_shap_zero:
        print(f"   💾 SHAP summary saved to: {shap_dir}")
        print(f"      ⚠️  Note: SHAP values are invalid (model output constant)")
        print(
            f"      → Check {prefix}_shap_importance_enhanced.json for diagnostic info"
        )
    else:
        print(f"   💾 SHAP summary saved to: {shap_dir}")

    return str(shap_dir)


def compute_selection_score(
    perf: Dict,
    metric: str,
    *,
    max_dd_threshold: float = -20.0,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> float:
    """Compute selection score from a performance dictionary using the chosen metric."""
    fm = perf.get("financial_metrics", {}) if isinstance(perf, dict) else {}
    sharpe = float(fm.get("sharpe_ratio", 0.0))
    max_dd = float(fm.get("max_drawdown", 0.0))
    f1 = float(fm.get("f1", fm.get("directional_f1", 0.0)))
    if f1 == 0.0:
        # Fallback to classification metrics
        cls_metrics = perf.get("classification_metrics", {})
        f1 = float(cls_metrics.get("f1_macro", cls_metrics.get("f1_weighted", 0.0)))

    if metric == "sharpe":
        return sharpe
    if metric == "f1":
        return f1
    if metric == "r2":
        return float(perf.get("r2", 0.0))

    # Composite score: Sharpe - alpha * penalty(DD) - beta * (1 - F1)
    dd_penalty = 0.0
    if max_dd < max_dd_threshold:
        dd_penalty = abs(max_dd - max_dd_threshold)
    return sharpe - alpha * dd_penalty - beta * (1.0 - f1)


def calculate_financial_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """
    Calculate financial metrics for trading strategy evaluation.

    Args:
        y_true: True returns
        y_pred: Predicted returns
        risk_free_rate: Risk-free rate (annualized, default 0)

    Returns:
        Dictionary of financial metrics
    """
    metrics = {}

    try:
        # Strategy: use predicted returns as position signals
        # Simple strategy: long if pred > 0, short if pred < 0 (proportional to confidence)
        positions = np.sign(y_pred) * np.abs(y_pred)
        # Clip positions to reasonable range [-1, 1]
        positions = np.clip(positions, -1.0, 1.0)

        # Strategy returns: position * true return
        strategy_returns = positions * y_true

        # 1. Total return (cumulative)
        total_return = float(np.sum(strategy_returns))
        metrics["total_return"] = total_return

        # 2. Annualized return (assuming daily data)
        n_periods = len(strategy_returns)
        if n_periods > 0:
            # Simple annualization: multiply by ~252 trading days
            annualized_return = (
                total_return * (252.0 / n_periods) if n_periods < 252 else total_return
            )
            metrics["annualized_return"] = annualized_return
        else:
            metrics["annualized_return"] = 0.0

        # 3. Sharpe ratio
        returns_std = np.std(strategy_returns)
        if returns_std > 1e-8:
            # Annualized Sharpe: (mean_return - risk_free) / std_return * sqrt(252)
            daily_rf = risk_free_rate / 252.0
            sharpe_ratio = (
                (np.mean(strategy_returns) - daily_rf) / returns_std * np.sqrt(252.0)
            )
            metrics["sharpe_ratio"] = float(sharpe_ratio)
        else:
            metrics["sharpe_ratio"] = 0.0

        # 4. Maximum drawdown
        # Convert returns to cumulative equity (starting from 1.0)
        cumulative_equity = np.cumprod(1.0 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative_equity)
        drawdown = (cumulative_equity - running_max) / running_max
        max_drawdown_pct = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0
        # Also store absolute drawdown for backward compatibility
        max_drawdown_abs = (
            float(np.min(cumulative_equity - running_max)) if len(drawdown) > 0 else 0.0
        )
        metrics["max_drawdown"] = max_drawdown_pct  # Store as percentage
        metrics["max_drawdown_abs"] = max_drawdown_abs  # Store absolute value
        metrics["max_drawdown_pct"] = max_drawdown_pct

        # 5. Win rate
        winning_trades = (strategy_returns > 0).sum()
        total_trades = len(strategy_returns)
        metrics["win_rate"] = (
            float(winning_trades / total_trades) if total_trades > 0 else 0.0
        )

        # 6. Average win/loss ratio
        wins = strategy_returns[strategy_returns > 0]
        losses = strategy_returns[strategy_returns < 0]
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.abs(np.mean(losses))) if len(losses) > 0 else 0.0
        metrics["avg_win"] = avg_win
        metrics["avg_loss"] = avg_loss
        metrics["win_loss_ratio"] = avg_win / avg_loss if avg_loss > 1e-8 else 0.0

        # 7. Volatility (annualized)
        volatility = np.std(strategy_returns) * np.sqrt(252.0)
        metrics["volatility"] = float(volatility)

        # 8. Calmar ratio (return / max_drawdown)
        if abs(max_drawdown_pct) > 1e-8:
            metrics["calmar_ratio"] = float(annualized_return / abs(max_drawdown_pct))
        else:
            metrics["calmar_ratio"] = 0.0

    except Exception as e:
        print(f"⚠️ Error calculating financial metrics: {e}")
        # Return defaults
        metrics = {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate": 0.0,
        }

    return metrics


def evaluate_model_performance(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    include_financial_metrics: bool = True,
    price_data: Optional[
        pd.DataFrame
    ] = None,  # Optional: price data for calculating real returns
):
    """Evaluate model performance with comprehensive metrics."""
    X_eval = X_test
    y_eval = y_test

    predictions = model.predict(X_eval)

    # Handle multiclass predictions: LightGBM returns class indices for multiclass
    # Shape: (n_samples, num_classes) for probability predictions, (n_samples,) for class predictions
    # For multiclass (3-class), convert probability array to class predictions if needed
    if predictions.ndim == 2 and predictions.shape[1] > 1:
        # Probability array: convert to class predictions
        predictions_class = np.argmax(predictions, axis=1)
        probabilities = predictions
    else:
        # Already class predictions (1D array)
        predictions_class = predictions
        probabilities = None

    # Determine if binary or multiclass based on unique values in predictions
    unique_preds = np.unique(predictions_class)
    is_binary = len(unique_preds) == 2 and all(p in [0, 1] for p in unique_preds)
    is_multiclass = len(unique_preds) > 2

    if is_binary or is_multiclass:
        # Classification: use class predictions
        predictions_for_metrics = predictions_class
        predictions_to_store = predictions_class
    else:
        # Regression: use predictions as-is
        predictions_for_metrics = predictions_class
        predictions_to_store = predictions_class

    # Binary classification labels
    # Convert to numpy array first to handle pandas Int64 type
    import pandas as pd

    try:
        if hasattr(y_eval, "to_numpy"):
            y_eval_np = y_eval.to_numpy(dtype=float, na_value=np.nan)
        else:
            y_eval_np = np.asarray(y_eval, dtype=float)
    except (ValueError, TypeError):
        # Fallback: convert to Series first, then to numpy
        y_eval_series = (
            pd.Series(y_eval) if not isinstance(y_eval, pd.Series) else y_eval
        )
        y_eval_np = y_eval_series.to_numpy(dtype=float, na_value=np.nan)
    # Filter NaN before checking
    y_eval_valid = y_eval_np[~np.isnan(y_eval_np)]

    is_binary_classification = False
    if y_eval_np.ndim == 1:
        # Check if values are integers (after filtering NaN)
        if len(y_eval_valid) > 0:
            unique_eval = np.unique(y_eval_valid)
            if len(unique_eval) <= 2 and np.all(np.isin(unique_eval, [0, 1])):
                y_eval_binary = y_eval_np.astype(int)
                is_binary_classification = True
            else:
                y_eval_binary = None
        else:
            y_eval_binary = None
    else:
        y_eval_binary = None

    if is_binary_classification:
        if predictions.ndim == 1:
            probabilities = predictions
        elif predictions.ndim == 2 and predictions.shape[1] == 1:
            probabilities = predictions[:, 0]
        elif predictions.ndim == 2 and predictions.shape[1] == 2:
            probabilities = predictions[:, 1]
        else:
            probabilities = predictions
        predictions_class = (probabilities >= 0.5).astype(int)
        predictions_for_metrics = predictions_class
        predictions_to_store = predictions_class

    # Basic numeric metrics
    mse = mean_squared_error(
        y_eval if not is_binary_classification else y_eval_binary,
        predictions_for_metrics,
    )
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(
        y_eval if not is_binary_classification else y_eval_binary,
        predictions_for_metrics,
    )
    # Calculate R² for regression metrics
    target_for_r2 = y_eval if not is_binary_classification else y_eval_binary
    r2 = (
        r2_score(target_for_r2, predictions_for_metrics)
        if len(np.unique(target_for_r2)) > 1
        else 0.0
    )

    print(f"📊 {model_name} Performance:")
    print(f"  R²: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")

    results = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions": predictions_to_store,
    }

    # Add financial or directional metrics
    if include_financial_metrics:
        if is_binary_classification:
            # Binary classification: compute win rate (0=Short, 1=Long)
            y_pred_cls = predictions_class.astype(int)
            y_true_cls = y_eval_binary.astype(int)

            # Calculate win rate (accuracy for binary classification)
            win_rate = float(np.mean(y_pred_cls == y_true_cls))

            # Long predictions win rate
            long_mask = y_pred_cls == 1
            long_total = int(np.sum(long_mask))
            if long_total > 0:
                long_correct = np.sum((y_pred_cls == 1) & (y_true_cls == 1))
                long_win_rate = float(long_correct / long_total)
            else:
                long_win_rate = 0.0

            # Short predictions win rate
            short_mask = y_pred_cls == 0
            short_total = int(np.sum(short_mask))
            if short_total > 0:
                short_correct = np.sum((y_pred_cls == 0) & (y_true_cls == 0))
                short_win_rate = float(short_correct / short_total)
            else:
                short_win_rate = 0.0

            fm = results.setdefault("financial_metrics", {})
            fm["win_rate"] = win_rate
            fm["long_win_rate"] = long_win_rate
            fm["short_win_rate"] = short_win_rate

            # For classification tasks, calculate Sharpe Ratio and Max Drawdown
            # Try to use real backtest if price data is available, otherwise use win_rate as proxy
            if price_data is not None and "close" in price_data.columns:
                try:
                    # Use real backtest with actual price data
                    if len(price_data) == len(y_pred_cls):
                        # Calculate strategy returns from predictions and actual prices
                        strategy_returns = calculate_strategy_returns_from_predictions(
                            y_pred_cls, price_data, horizon=1
                        )
                        # Calculate financial metrics from real returns
                        backtest_metrics = calculate_financial_metrics_from_returns(
                            strategy_returns, risk_free_rate=0.0
                        )
                        fm["sharpe_ratio"] = backtest_metrics.get("sharpe_ratio", 0.0)
                        fm["max_drawdown"] = backtest_metrics.get("max_drawdown", 0.0)
                        fm["total_return"] = backtest_metrics.get("total_return", 0.0)
                        fm["annualized_return"] = backtest_metrics.get(
                            "annualized_return", 0.0
                        )
                        fm["volatility"] = backtest_metrics.get("volatility", 0.0)
                    else:
                        # Fallback: use win rate as proxy
                        sharpe_approx = (win_rate - 0.5) * 4.0
                        fm["sharpe_ratio"] = float(sharpe_approx)
                        max_dd_approx = -(1.0 - win_rate) * 0.1
                        fm["max_drawdown"] = float(max_dd_approx)
                        total_return_approx = (win_rate - 0.5) * 2.0
                        fm["total_return"] = float(total_return_approx)
                except Exception as e:
                    # If backtest fails, fall back to win rate approximation
                    print(
                        f"  ⚠️  Backtest calculation failed: {e}, using win_rate approximation"
                    )
                    sharpe_approx = (win_rate - 0.5) * 4.0
                    fm["sharpe_ratio"] = float(sharpe_approx)
                    max_dd_approx = -(1.0 - win_rate) * 0.1
                    fm["max_drawdown"] = float(max_dd_approx)
                    total_return_approx = (win_rate - 0.5) * 2.0
                    fm["total_return"] = float(total_return_approx)
            else:
                # No price data available: use win rate as proxy for Sharpe Ratio
                sharpe_approx = (win_rate - 0.5) * 4.0
                fm["sharpe_ratio"] = float(sharpe_approx)
                max_dd_approx = -(1.0 - win_rate) * 0.1
                fm["max_drawdown"] = float(max_dd_approx)
                total_return_approx = (win_rate - 0.5) * 2.0
                fm["total_return"] = float(total_return_approx)

            print(f"  Win Rate: {win_rate:.4f}")
            print(f"  Long Win Rate: {long_win_rate:.4f}")
            print(f"  Short Win Rate: {short_win_rate:.4f}")
            print(f"  Sharpe Ratio: {fm.get('sharpe_ratio', 0):.4f}")
            print(f"  Max Drawdown: {fm.get('max_drawdown', 0):.4f}")

            # Classification diagnostics
            metrics = results.setdefault("classification_metrics", {})
            accuracy = float(np.mean(y_pred_cls == y_true_cls))
            metrics["accuracy"] = accuracy
            try:
                metrics["f1_macro"] = float(
                    f1_score(y_true_cls, y_pred_cls, average="macro")
                )
            except Exception:
                metrics["f1_macro"] = None
            try:
                metrics["f1_weighted"] = float(
                    f1_score(y_true_cls, y_pred_cls, average="weighted")
                )
            except Exception:
                metrics["f1_weighted"] = None

            # Binary classification: probabilities shape is (n_samples, 2)
            if (
                probabilities is not None
                and probabilities.ndim == 2
                and probabilities.shape[1] == 2
            ):
                try:
                    # Binary classification: use positive class probabilities
                    metrics["roc_auc_macro"] = float(
                        roc_auc_score(
                            y_true_cls,
                            probabilities[:, 1],  # Use positive class probabilities
                        )
                    )
                except Exception:
                    metrics["roc_auc_macro"] = None
                try:
                    metrics["pr_auc_macro"] = float(
                        average_precision_score(
                            y_true_cls,
                            probabilities[:, 1],  # Use positive class probabilities
                        )
                    )
                except Exception:
                    metrics["pr_auc_macro"] = None
            else:
                metrics["roc_auc_macro"] = None
                metrics["pr_auc_macro"] = None

            try:
                cm = confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1])
                metrics["confusion_matrix"] = cm.tolist()
                metrics["labels"] = [0, 1]
            except Exception:
                metrics["confusion_matrix"] = None
                metrics["labels"] = None

            try:
                cls_report = classification_report(
                    y_true_cls,
                    y_pred_cls,
                    output_dict=True,
                    zero_division=0,
                )
                metrics["classification_report"] = cls_report
            except Exception:
                metrics["classification_report"] = None

            metrics["support"] = int(len(y_true_cls))

            print(f"  Accuracy: {accuracy:.4f}")
            if metrics["f1_macro"] is not None:
                print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
            if metrics["roc_auc_macro"] is not None:
                print(f"  ROC AUC (macro): {metrics['roc_auc_macro']:.4f}")
            if metrics["pr_auc_macro"] is not None:
                print(f"  PR AUC: {metrics['pr_auc_macro']:.4f}")
        else:
            # Regression/binary: compute financial metrics using returns-like predictions
            financial_metrics = calculate_financial_metrics(
                y_eval, predictions_for_metrics
            )
            results["financial_metrics"] = financial_metrics
            print(f"  Sharpe Ratio: {financial_metrics.get('sharpe_ratio', 0):.4f}")
            print(f"  Total Return: {financial_metrics.get('total_return', 0):.4f}")
            print(f"  Max Drawdown: {financial_metrics.get('max_drawdown', 0):.4f}")

    return results
