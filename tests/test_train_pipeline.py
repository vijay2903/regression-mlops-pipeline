"""
Integration tests for the training pipeline.

This test verifies that the complete training workflow can:

1. Load configuration
2. Load and validate data
3. Split data into train and test sets
4. Train configured models
5. Evaluate trained models
6. Save model artifacts
7. Save comparison results
8. Select and save the best model metadata

The real data-loading function is patched so the test remains
small, deterministic, and independent of the actual dataset.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import train


class MockRunInfo:
    """Minimal MLflow run information object."""

    def __init__(self, run_id):
        self.run_id = run_id


class MockRun:
    """Minimal context manager that behaves like an MLflow run."""

    def __init__(self, run_id="dummy-run-id"):
        self.info = MockRunInfo(run_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


@pytest.fixture
def sample_training_data():
    """
    Create a small regression dataset.

    The target column matches the project's actual configuration:
    MedHouseVal.
    """

    return pd.DataFrame(
        {
            "MedInc": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
            ],
            "HouseAge": [
                10.0,
                15.0,
                20.0,
                25.0,
                30.0,
                35.0,
                40.0,
                45.0,
                50.0,
                55.0,
            ],
            "AveRooms": [
                2.0,
                2.5,
                3.0,
                3.5,
                4.0,
                4.5,
                5.0,
                5.5,
                6.0,
                6.5,
            ],
            "AveBedrms": [
                1.0,
                1.1,
                1.2,
                1.3,
                1.4,
                1.5,
                1.6,
                1.7,
                1.8,
                1.9,
            ],
            "Population": [
                100.0,
                150.0,
                200.0,
                250.0,
                300.0,
                350.0,
                400.0,
                450.0,
                500.0,
                550.0,
            ],
            "AveOccup": [
                2.0,
                2.1,
                2.2,
                2.3,
                2.4,
                2.5,
                2.6,
                2.7,
                2.8,
                2.9,
            ],
            "Latitude": [
                34.0,
                34.1,
                34.2,
                34.3,
                34.4,
                34.5,
                34.6,
                34.7,
                34.8,
                34.9,
            ],
            "Longitude": [
                -118.0,
                -118.1,
                -118.2,
                -118.3,
                -118.4,
                -118.5,
                -118.6,
                -118.7,
                -118.8,
                -118.9,
            ],
            "MedHouseVal": [
                1.0,
                1.5,
                2.0,
                2.5,
                3.0,
                3.5,
                4.0,
                4.5,
                5.0,
                5.5,
            ],
        }
    )


def test_train_pipeline(
    monkeypatch,
    tmp_path,
    sample_training_data,
):
    """
    Test the complete training pipeline using temporary directories.

    Both models from the real configuration are included:
    - LinearRegression
    - RandomForest
    """

    # ------------------------------------------------------------------
    # Create a temporary configuration file
    # ------------------------------------------------------------------

    config = {
        "data": {
            "target_column": "MedHouseVal",
            "test_size": 0.2,
            "random_state": 42,
        },
        "models": [
            {
                "name": "LinearRegression",
                "type": "linear_regression",
                "filename": "baseline_linear_regression.joblib",
            },
            {
                "name": "RandomForest",
                "type": "random_forest",
                "filename": "random_forest.joblib",
            },
        ],
        "paths": {
            "results_filename": "model_comparison.json",
            "prediction_input_filename": "prediction_input.csv",
            "prediction_output_filename": "predictions.csv",
        },
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(config, indent=4),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Create temporary output directories
    # ------------------------------------------------------------------

    results_dir = tmp_path / "results"
    models_dir = tmp_path / "models"

    # ------------------------------------------------------------------
    # Patch paths used by scripts.train
    # ------------------------------------------------------------------

    monkeypatch.setattr(train, "CONFIG_PATH", config_path)
    monkeypatch.setattr(train, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(train, "MODELS_DIR", models_dir)

    # ------------------------------------------------------------------
    # Patch data loading
    # ------------------------------------------------------------------

    def mock_load_data():
        """Return the small in-memory dataset instead of reading a file."""
        return sample_training_data.copy()

    monkeypatch.setattr(train, "load_data", mock_load_data)

    # ------------------------------------------------------------------
    # Patch MLflow so the test does not require a running MLflow server
    # ------------------------------------------------------------------

    monkeypatch.setattr(
        train.mlflow,
        "start_run",
        lambda *args, **kwargs: MockRun(f"run-{kwargs.get('run_name', 'default')}"),
    )

    monkeypatch.setattr(
        train.mlflow,
        "active_run",
        lambda: MockRun("active-run-id"),
    )

    monkeypatch.setattr(
        train.mlflow,
        "log_param",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        train.mlflow,
        "log_params",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        train.mlflow,
        "log_metric",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        train.mlflow,
        "log_metrics",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        train.mlflow,
        "set_tag",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        train.mlflow,
        "log_artifact",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        train.mlflow.sklearn,
        "log_model",
        lambda *args, **kwargs: None,
    )

    # ------------------------------------------------------------------
    # Execute the complete training pipeline
    # ------------------------------------------------------------------

    train.train_pipeline()

    # ------------------------------------------------------------------
    # Verify output directories were created
    # ------------------------------------------------------------------

    assert results_dir.exists()
    assert models_dir.exists()

    # ------------------------------------------------------------------
    # Verify both trained model files were saved
    # ------------------------------------------------------------------

    linear_regression_model = (
        models_dir / "baseline_linear_regression.joblib"
    )

    random_forest_model = models_dir / "random_forest.joblib"

    assert linear_regression_model.exists()
    assert random_forest_model.exists()

    # ------------------------------------------------------------------
    # Verify model comparison results were saved
    # ------------------------------------------------------------------

    results_file = results_dir / "model_comparison.json"

    assert results_file.exists()

    results = json.loads(
        results_file.read_text(encoding="utf-8")
    )

    assert isinstance(results, list)
    assert len(results) == 2

    # Verify structural schema of model_comparison.json
    for result in results:
        assert "model" in result
        assert "model_filename" in result
        assert "train" in result
        assert "test" in result

        assert "rmse" in result["train"]
        assert "mae" in result["train"]
        assert "r2" in result["train"]

        assert "rmse" in result["test"]
        assert "mae" in result["test"]
        assert "r2" in result["test"]

        assert result["test"]["rmse"] >= 0
        assert result["test"]["mae"] >= 0

    # ------------------------------------------------------------------
    # Verify best-model metadata was saved
    # ------------------------------------------------------------------

    best_model_file = results_dir / "best_model.json"

    assert best_model_file.exists()

    best_model = json.loads(
        best_model_file.read_text(encoding="utf-8")
    )

    # Verify structural schema of best_model.json
    assert "model_name" in best_model
    assert "model_filename" in best_model
    assert "metrics" in best_model

    assert best_model["model_name"] in {
        "LinearRegression",
        "RandomForest",
    }

    assert (models_dir / best_model["model_filename"]).exists()

    assert "test" in best_model["metrics"]
    assert "rmse" in best_model["metrics"]["test"]
    assert best_model["metrics"]["test"]["rmse"] >= 0