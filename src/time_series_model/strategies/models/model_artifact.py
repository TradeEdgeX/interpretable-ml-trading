"""
ModelArtifact: Unified model deployment artifact management.

This module provides a unified class to save and load all components required
for model deployment: model, preprocessor, used_features, and feature_config.
"""

from __future__ import annotations

import json
import joblib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.time_series_model.strategies.models.strategy_trainer import FeaturePreprocessor


@dataclass
class ModelArtifact:
    """
    Unified model deployment artifact container.

    This class encapsulates all components required for consistent model inference:
    - model: Trained model(s) (LightGBM/XGBoost/CatBoost)
    - preprocessor: FeaturePreprocessor for feature transformation
    - used_features: List of feature names actually used by the model
    - feature_config: Feature configuration (YAML path or dict) for feature computation
    - metadata: Additional metadata (strategy name, model type, task type, etc.)
    """

    model: Any  # Trained model(s) - can be a single model or list of models
    preprocessor: FeaturePreprocessor
    used_features: List[str]
    feature_config: Optional[Dict[str, Any]] = None  # Feature config dict or path
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, output_dir: Path, model_filename: str = "model.pkl") -> None:
        """
        Save all components to disk.

        Args:
            output_dir: Directory to save artifacts
            model_filename: Filename for model (default: "model.pkl")
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = output_dir / model_filename
        joblib.dump(self.model, model_path)
        print(f"   💾 Model saved to {model_path}")

        # Save preprocessor
        preprocessor_path = output_dir / "preprocessor.pkl"
        joblib.dump(self.preprocessor, preprocessor_path)
        print(f"   💾 Preprocessor saved to {preprocessor_path}")

        # Save used_features
        features_path = output_dir / "used_features.json"
        with open(features_path, "w", encoding="utf-8") as f:
            json.dump(self.used_features, f, indent=2)
        print(f"   💾 Used features saved to {features_path}")

        # Save feature_config if provided
        if self.feature_config:
            config_path = output_dir / "feature_config.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.feature_config, f, indent=2, default=str)
            print(f"   💾 Feature config saved to {config_path}")

        # Save metadata
        metadata_path = output_dir / "model_artifact_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, default=str)
        print(f"   💾 Metadata saved to {metadata_path}")

    @classmethod
    def load(
        cls, artifact_dir: Path, model_filename: str = "model.pkl"
    ) -> ModelArtifact:
        """
        Load all components from disk.

        Args:
            artifact_dir: Directory containing saved artifacts
            model_filename: Filename for model (default: "model.pkl")

        Returns:
            ModelArtifact instance with all components loaded
        """
        artifact_dir = Path(artifact_dir)

        # Load model
        model_path = artifact_dir / model_filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = joblib.load(model_path)
        print(f"   ✅ Model loaded from {model_path}")

        # Load preprocessor
        preprocessor_path = artifact_dir / "preprocessor.pkl"
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
        preprocessor = joblib.load(preprocessor_path)
        print(f"   ✅ Preprocessor loaded from {preprocessor_path}")

        # Load used_features
        features_path = artifact_dir / "used_features.json"
        if not features_path.exists():
            raise FileNotFoundError(f"Used features file not found: {features_path}")
        with open(features_path, "r", encoding="utf-8") as f:
            used_features = json.load(f)
        print(f"   ✅ Used features loaded from {features_path}")

        # Load feature_config if exists
        feature_config = None
        config_path = artifact_dir / "feature_config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                feature_config = json.load(f)
            print(f"   ✅ Feature config loaded from {config_path}")

        # Load metadata if exists
        metadata = {}
        metadata_path = artifact_dir / "model_artifact_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            print(f"   ✅ Metadata loaded from {metadata_path}")

        return cls(
            model=model,
            preprocessor=preprocessor,
            used_features=used_features,
            feature_config=feature_config,
            metadata=metadata,
        )

    def predict(
        self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Make predictions using the artifact.

        Args:
            df: DataFrame with features
            feature_cols: Optional list of feature columns to use (default: used_features)

        Returns:
            Model predictions as numpy array
        """
        # Use used_features if feature_cols not specified
        if feature_cols is None:
            feature_cols = self.used_features

        # Transform features using preprocessor
        X = self.preprocessor.transform(df, feature_cols=feature_cols)

        # Make predictions
        # Handle both single model and list of models (for ensemble/CV)
        if isinstance(self.model, list):
            # Average predictions from multiple models
            predictions = [m.predict(X) for m in self.model]
            return np.mean(predictions, axis=0)
        else:
            return self.model.predict(X)

    def get_artifact_info(self) -> Dict[str, Any]:
        """
        Get summary information about the artifact.

        Returns:
            Dictionary with artifact information
        """
        return {
            "n_features": len(self.used_features),
            "feature_cols": (
                self.used_features[:10]
                if len(self.used_features) > 10
                else self.used_features
            ),
            "has_feature_config": self.feature_config is not None,
            "metadata": self.metadata,
        }
