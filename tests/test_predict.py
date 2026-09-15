import json

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import scripts.predict as predict


def test_load_config(tmp_path, monkeypatch):
    """
    Verify that load_config() reads and returns
    the JSON configuration correctly.
    """

    # Arrange
    config = {
        "paths": {
            "prediction_input_filename": "input.csv",
            "prediction_output_filename": "predictions.csv",
        }
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    monkeypatch.setattr(predict, "CONFIG_PATH", config_path)

    # Act
    result = predict.load_config()

    # Assert
    assert result == config


def test_generate_predictions_raises_error_when_input_file_is_missing(
    tmp_path,
    monkeypatch,
):
    """
    Verify that generate_predictions() raises FileNotFoundError
    when the configured prediction input file does not exist.
    """

    # Arrange
    config = {
        "paths": {
            "prediction_input_filename": "missing.csv",
            "prediction_output_filename": "predictions.csv",
        }
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    results_dir = tmp_path / "results"
    results_dir.mkdir()

    best_model_path = results_dir / "best_model.json"
    best_model_path.write_text(
        json.dumps({"model_filename": "model.joblib"})
    )

    input_dir = tmp_path / "input"
    models_dir = tmp_path / "models"

    input_dir.mkdir()
    models_dir.mkdir()

    monkeypatch.setattr(predict, "CONFIG_PATH", config_path)
    monkeypatch.setattr(predict, "BEST_MODEL_PATH", best_model_path)
    monkeypatch.setattr(predict, "INPUT_DIR", input_dir)
    monkeypatch.setattr(predict, "OUTPUT_DIR", results_dir)
    monkeypatch.setattr(predict, "MODELS_DIR", models_dir)

    # Act and Assert
    with pytest.raises(
        FileNotFoundError,
        match="Prediction input file not found",
    ):
        predict.generate_predictions()


def test_generate_predictions_creates_prediction_file(
    tmp_path,
    monkeypatch,
):
    """
    Verify that generate_predictions() loads the model,
    generates predictions, and writes the output CSV.
    """

    # Arrange
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "results"
    models_dir = tmp_path / "models"

    input_dir.mkdir()
    output_dir.mkdir()
    models_dir.mkdir()

    config = {
        "paths": {
            "prediction_input_filename": "input.csv",
            "prediction_output_filename": "predictions.csv",
        }
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    best_model_path = output_dir / "best_model.json"
    best_model_path.write_text(
        json.dumps({"model_filename": "model.joblib"})
    )

    # Create a small trained model for testing.
    training_data = pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4],
            "feature_2": [10, 20, 30, 40],
        }
    )

    target = [11, 22, 33, 44]

    model = LinearRegression()
    model.fit(training_data, target)

    model_artifact = {
        "model": model,
        "feature_columns": ["feature_1", "feature_2"],
        "target_column": "target",
    }

    model_path = models_dir / "model.joblib"
    joblib.dump(model_artifact, model_path)

    # Create valid prediction input data.
    input_data = pd.DataFrame(
        {
            "feature_1": [5, 6],
            "feature_2": [50, 60],
        }
    )

    input_data.to_csv(input_dir / "input.csv", index=False)

    # Redirect predict.py paths to temporary directories.
    monkeypatch.setattr(predict, "CONFIG_PATH", config_path)
    monkeypatch.setattr(predict, "BEST_MODEL_PATH", best_model_path)
    monkeypatch.setattr(predict, "INPUT_DIR", input_dir)
    monkeypatch.setattr(predict, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(predict, "MODELS_DIR", models_dir)

    # Act
    prediction_data = predict.generate_predictions()

    # Assert
    output_path = output_dir / "predictions.csv"

    assert output_path.exists()
    assert isinstance(prediction_data, pd.DataFrame)

    assert len(prediction_data) == 2
    assert "feature_1" in prediction_data.columns
    assert "feature_2" in prediction_data.columns
    assert "prediction" in prediction_data.columns

    assert pd.api.types.is_numeric_dtype(
        prediction_data["prediction"]
    )

    # Also verify the saved CSV contains the same columns.
    saved_predictions = pd.read_csv(output_path)

    assert list(saved_predictions.columns) == [
        "feature_1",
        "feature_2",
        "prediction",
    ]

    assert len(saved_predictions) == 2


def test_generate_predictions_rejects_input_with_missing_feature(
    tmp_path,
    monkeypatch,
):
    """
    Verify that generate_predictions() rejects input data
    missing one of the required feature columns.
    """

    # Arrange
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "results"
    models_dir = tmp_path / "models"

    input_dir.mkdir()
    output_dir.mkdir()
    models_dir.mkdir()

    config = {
        "paths": {
            "prediction_input_filename": "input.csv",
            "prediction_output_filename": "predictions.csv",
        }
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    best_model_path = output_dir / "best_model.json"
    best_model_path.write_text(
        json.dumps({"model_filename": "model.joblib"})
    )

    model = LinearRegression()

    training_data = pd.DataFrame(
        {
            "feature_1": [1, 2, 3],
            "feature_2": [10, 20, 30],
        }
    )

    model.fit(training_data, [11, 22, 33])

    model_artifact = {
        "model": model,
        "feature_columns": ["feature_1", "feature_2"],
        "target_column": "target",
    }

    joblib.dump(model_artifact, models_dir / "model.joblib")

    # feature_2 is intentionally missing.
    invalid_input = pd.DataFrame(
        {
            "feature_1": [5, 6],
        }
    )

    invalid_input.to_csv(input_dir / "input.csv", index=False)

    monkeypatch.setattr(predict, "CONFIG_PATH", config_path)
    monkeypatch.setattr(predict, "BEST_MODEL_PATH", best_model_path)
    monkeypatch.setattr(predict, "INPUT_DIR", input_dir)
    monkeypatch.setattr(predict, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(predict, "MODELS_DIR", models_dir)

    # Act and Assert
    with pytest.raises(ValueError):
        predict.generate_predictions()