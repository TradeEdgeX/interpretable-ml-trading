"""
Unit tests for ModelArtifact class.
"""

import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.time_series_model.strategies.models.model_artifact import ModelArtifact
from src.time_series_model.strategies.models.strategy_trainer import FeaturePreprocessor
from pathlib import Path


class MockModel:
    """Mock model for testing."""

    def __init__(self, predictions=None):
        self.predictions = (
            predictions if predictions is not None else np.array([0.5, 0.6, 0.7])
        )

    def predict(self, X):
        return self.predictions


@pytest.fixture
def sample_preprocessor():
    """Create a sample FeaturePreprocessor for testing."""
    return FeaturePreprocessor(
        feature_cols=["feature1", "feature2", "feature3"],
        numeric_cols=["feature1", "feature2", "feature3"],
        categorical_cols=[],
        categorical_mappings={},
        feature_multipliers=None,
    )


@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    return MockModel()


@pytest.fixture
def sample_artifact(sample_model, sample_preprocessor):
    """Create a sample ModelArtifact for testing."""
    return ModelArtifact(
        model=sample_model,
        preprocessor=sample_preprocessor,
        used_features=["feature1", "feature2", "feature3"],
        feature_config={"requested_features": ["feature1", "feature2", "feature3"]},
        metadata={
            "strategy": "test_strategy",
            "model_type": "lightgbm",
            "task_type": "regression",
        },
    )


def test_model_artifact_creation(sample_model, sample_preprocessor):
    """Test ModelArtifact creation."""
    artifact = ModelArtifact(
        model=sample_model,
        preprocessor=sample_preprocessor,
        used_features=["feature1", "feature2"],
        metadata={"strategy": "test"},
    )

    assert artifact.model == sample_model
    assert artifact.preprocessor == sample_preprocessor
    assert artifact.used_features == ["feature1", "feature2"]
    assert artifact.feature_config is None
    assert artifact.metadata == {"strategy": "test"}


def test_model_artifact_save(tmp_path: Path, sample_artifact):
    """Test ModelArtifact save functionality."""
    output_dir = tmp_path / "artifacts"
    sample_artifact.save(output_dir)

    # Check all files are created
    assert (output_dir / "model.pkl").exists()
    assert (output_dir / "preprocessor.pkl").exists()
    assert (output_dir / "used_features.json").exists()
    assert (output_dir / "feature_config.json").exists()
    assert (output_dir / "model_artifact_metadata.json").exists()

    # Verify used_features content
    with open(output_dir / "used_features.json") as f:
        loaded_features = json.load(f)
    assert loaded_features == ["feature1", "feature2", "feature3"]

    # Verify metadata content
    with open(output_dir / "model_artifact_metadata.json") as f:
        loaded_metadata = json.load(f)
    assert loaded_metadata["strategy"] == "test_strategy"
    assert loaded_metadata["model_type"] == "lightgbm"


def test_model_artifact_save_custom_filename(tmp_path: Path, sample_artifact):
    """Test ModelArtifact save with custom model filename."""
    output_dir = tmp_path / "artifacts"
    sample_artifact.save(output_dir, model_filename="custom_model.pkl")

    assert (output_dir / "custom_model.pkl").exists()
    assert not (output_dir / "model.pkl").exists()


def test_model_artifact_save_without_feature_config(
    tmp_path: Path, sample_model, sample_preprocessor
):
    """Test ModelArtifact save without feature_config."""
    artifact = ModelArtifact(
        model=sample_model,
        preprocessor=sample_preprocessor,
        used_features=["feature1"],
        feature_config=None,
        metadata={},
    )

    output_dir = tmp_path / "artifacts"
    artifact.save(output_dir)

    # feature_config.json should not be created
    assert not (output_dir / "feature_config.json").exists()
    assert (output_dir / "model.pkl").exists()
    assert (output_dir / "preprocessor.pkl").exists()


def test_model_artifact_load(tmp_path: Path, sample_artifact):
    """Test ModelArtifact load functionality."""
    # Save first
    output_dir = tmp_path / "artifacts"
    sample_artifact.save(output_dir)

    # Load
    loaded_artifact = ModelArtifact.load(output_dir)

    # Verify all components are loaded
    assert loaded_artifact.model is not None
    assert loaded_artifact.preprocessor is not None
    assert loaded_artifact.used_features == ["feature1", "feature2", "feature3"]
    assert loaded_artifact.feature_config == {
        "requested_features": ["feature1", "feature2", "feature3"]
    }
    assert loaded_artifact.metadata["strategy"] == "test_strategy"


def test_model_artifact_load_missing_files(tmp_path: Path):
    """Test ModelArtifact load with missing files."""
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()

    # Missing model file
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        ModelArtifact.load(output_dir)

    # Create model file but missing preprocessor
    import joblib

    joblib.dump(MockModel(), output_dir / "model.pkl")
    with pytest.raises(FileNotFoundError, match="Preprocessor file not found"):
        ModelArtifact.load(output_dir)

    # Create preprocessor but missing used_features
    joblib.dump(
        FeaturePreprocessor(
            feature_cols=["f1"],
            numeric_cols=["f1"],
            categorical_cols=[],
            categorical_mappings={},
        ),
        output_dir / "preprocessor.pkl",
    )
    with pytest.raises(FileNotFoundError, match="Used features file not found"):
        ModelArtifact.load(output_dir)


def test_model_artifact_load_without_optional_files(tmp_path: Path, sample_artifact):
    """Test ModelArtifact load when optional files (feature_config, metadata) are missing."""
    output_dir = tmp_path / "artifacts"
    sample_artifact.save(output_dir)

    # Remove optional files
    (output_dir / "feature_config.json").unlink()
    (output_dir / "model_artifact_metadata.json").unlink()

    # Should still load successfully with None/empty defaults
    loaded_artifact = ModelArtifact.load(output_dir)
    assert loaded_artifact.feature_config is None
    assert loaded_artifact.metadata == {}


def test_model_artifact_predict(sample_artifact):
    """Test ModelArtifact predict functionality."""
    # Create test DataFrame
    df = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [4.0, 5.0, 6.0],
            "feature3": [7.0, 8.0, 9.0],
        }
    )

    predictions = sample_artifact.predict(df)

    # Verify predictions shape and values
    assert predictions.shape == (3,)
    assert np.allclose(predictions, [0.5, 0.6, 0.7])


def test_model_artifact_predict_with_custom_feature_cols(sample_artifact):
    """Test ModelArtifact predict with custom feature columns."""
    # Create a model that returns predictions matching input rows
    model = MockModel(predictions=np.array([0.1, 0.2]))
    artifact = ModelArtifact(
        model=model,
        preprocessor=sample_artifact.preprocessor,
        used_features=sample_artifact.used_features,
    )

    df = pd.DataFrame(
        {
            "feature1": [1.0, 2.0],
            "feature2": [4.0, 5.0],
            "feature3": [7.0, 8.0],
            "other_feature": [10.0, 11.0],
        }
    )

    # Use only subset of features
    predictions = artifact.predict(df, feature_cols=["feature1", "feature2"])

    assert predictions.shape == (2,)
    assert np.allclose(predictions, [0.1, 0.2])


def test_model_artifact_predict_with_ensemble_model(sample_preprocessor):
    """Test ModelArtifact predict with ensemble model (list of models)."""
    model1 = MockModel(predictions=np.array([0.5, 0.6]))
    model2 = MockModel(predictions=np.array([0.7, 0.8]))
    model3 = MockModel(predictions=np.array([0.9, 1.0]))

    artifact = ModelArtifact(
        model=[model1, model2, model3],
        preprocessor=sample_preprocessor,
        used_features=["feature1", "feature2", "feature3"],
    )

    df = pd.DataFrame(
        {
            "feature1": [1.0, 2.0],
            "feature2": [4.0, 5.0],
            "feature3": [7.0, 8.0],
        }
    )

    predictions = artifact.predict(df)

    # Should average predictions from all models
    expected = np.mean([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]], axis=0)
    assert predictions.shape == (2,)
    assert np.allclose(predictions, expected)


def test_model_artifact_get_artifact_info(sample_artifact):
    """Test ModelArtifact get_artifact_info."""
    info = sample_artifact.get_artifact_info()

    assert info["n_features"] == 3
    assert len(info["feature_cols"]) == 3
    assert info["has_feature_config"] is True
    assert "metadata" in info
    assert info["metadata"]["strategy"] == "test_strategy"


def test_model_artifact_get_artifact_info_many_features(
    sample_model, sample_preprocessor
):
    """Test ModelArtifact get_artifact_info with many features (truncation)."""
    many_features = [f"feature_{i}" for i in range(20)]
    artifact = ModelArtifact(
        model=sample_model,
        preprocessor=sample_preprocessor,
        used_features=many_features,
    )

    info = artifact.get_artifact_info()

    assert info["n_features"] == 20
    assert len(info["feature_cols"]) == 10  # Should be truncated to 10


def test_model_artifact_roundtrip(tmp_path: Path, sample_artifact):
    """Test complete save/load roundtrip."""
    output_dir = tmp_path / "artifacts"
    sample_artifact.save(output_dir)

    loaded_artifact = ModelArtifact.load(output_dir)

    # Verify all components match
    assert loaded_artifact.used_features == sample_artifact.used_features
    assert loaded_artifact.feature_config == sample_artifact.feature_config
    assert loaded_artifact.metadata == sample_artifact.metadata

    # Test predict with loaded artifact
    df = pd.DataFrame(
        {
            "feature1": [1.0, 2.0],
            "feature2": [4.0, 5.0],
            "feature3": [7.0, 8.0],
        }
    )

    original_predictions = sample_artifact.predict(df)
    loaded_predictions = loaded_artifact.predict(df)

    assert np.allclose(original_predictions, loaded_predictions)
