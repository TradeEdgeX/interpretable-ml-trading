"""Config-driven feature engineer that loads features defined in YAML configs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Set, Tuple

import pandas as pd

from src.features.loader.strategy_feature_loader import StrategyFeatureLoader


BASE_DATA_COLUMNS: Set[str] = {
    "timestamp",
    "datetime",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
    "_symbol",
    "trade_count",
    "buy_qty",
    "sell_qty",
    "delta",
    "taker_buy_ratio",
    "cvd",
    "cvd_roll20",
    "cvd_roll60",
    "cvd_roll288",
}

EXCLUDE_PREFIXES: Tuple[str, ...] = (
    "signal_",
    "binary_signal_",
    "future_return_",
)


@dataclass
class ConfigFeatureEngineer:
    """
    Config-driven feature engineer that loads features by strategy name.

    Args:
        strategy_name: Strategy defined in strategy_features.yaml
        feature_loader: Optional shared StrategyFeatureLoader instance
    """

    strategy_name: str
    feature_loader: StrategyFeatureLoader = field(
        default_factory=StrategyFeatureLoader
    )
    _feature_cols: List[str] = field(default_factory=list, init=False)

    def engineer_all_features(
        self,
        df: pd.DataFrame,
        fit: bool = True,
        required_features: Optional[Set[str]] = None,
    ) -> pd.DataFrame:
        """
        Load features for the configured strategy.

        Args:
            df: Input dataframe
            fit: Whether this is fit (train) or transform (test)
            required_features: Optional subset of features to compute

        Returns:
            DataFrame with requested features appended
        """
        requested_override = (
            list(required_features) if required_features is not None else None
        )

        result_df = self.feature_loader.load_strategy_features(
            df,
            strategy_name=self.strategy_name,
            fit=fit,
            requested_features_override=requested_override,
        )

        self._feature_cols = self._extract_feature_columns(result_df)
        return result_df

    def engineer_features(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """Backward-compatible alias."""
        return self.engineer_all_features(df, fit=fit)

    def get_feature_columns(self, df: Optional[pd.DataFrame] = None) -> List[str]:
        """
        Return the latest engineered feature columns.

        If no features have been computed yet and df is provided, an inference
        pass will be executed (without fitting) to determine the columns.
        """
        if not self._feature_cols and df is not None:
            _ = self.engineer_all_features(df, fit=False)
        return self._feature_cols

    def save_scalers(self, path: str) -> None:
        """No-op for compatibility; config-based features do not use scalers."""
        print(
            f"⚠️  ConfigFeatureEngineer.save_scalers is a no-op. "
            f"Config-driven features use deterministic calculations."
        )

    def load_scalers(self, path: str) -> None:
        """No-op for compatibility; config-based features do not use scalers."""
        print(
            f"⚠️  ConfigFeatureEngineer.load_scalers is a no-op. "
            f"No external scaler state is required."
        )

    def clear_cache(self) -> None:
        """Clear in-memory cache of computed feature columns."""
        self._feature_cols = []

    def _extract_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Extract engineered feature columns by removing base data columns."""
        base_cols = set(BASE_DATA_COLUMNS)
        feature_cols: List[str] = []
        for col in df.columns:
            if col in base_cols:
                continue
            if any(col.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
                continue
            feature_cols.append(col)
        return feature_cols


def engineer_features_by_strategy(
    df: pd.DataFrame,
    strategy_name: str,
    feature_engineer: Optional[ConfigFeatureEngineer] = None,
    fit: bool = True,
    required_features: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, ConfigFeatureEngineer]:
    """
    Convenience helper to engineer features for a given strategy.

    Returns the engineered dataframe and the feature engineer instance so it can
    be reused for subsequent transforms (e.g., train/test split handling).
    """
    if feature_engineer is None:
        feature_engineer = ConfigFeatureEngineer(strategy_name=strategy_name)

    engineered_df = feature_engineer.engineer_all_features(
        df, fit=fit, required_features=required_features
    )
    return engineered_df, feature_engineer


def get_feature_columns_by_strategy(
    df: pd.DataFrame,
    strategy_name: str,
) -> List[str]:
    """Utility to extract feature columns for a specific strategy."""
    engineer = ConfigFeatureEngineer(strategy_name=strategy_name)
    engineer.engineer_all_features(df, fit=False)
    return engineer.get_feature_columns()

