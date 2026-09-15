import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.predict import generate_predictions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.json"
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "results"
BEST_MODEL_PATH = OUTPUT_DIR / "best_model.json"


@pytest.fixture
def prediction_output_path():
    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)

    return OUTPUT_DIR / config["paths"]["prediction_output_filename"]


def test_prediction_pipeline_returns_dataframe():
    result = generate_predictions()

    assert isinstance(result, pd.DataFrame)


def test_prediction_pipeline_creates_prediction_column():
    result = generate_predictions()

    assert "prediction" in result.columns


def test_prediction_pipeline_generates_predictions(prediction_output_path):
    result = generate_predictions()

    assert len(result) > 0
    assert result["prediction"].notna().all()
    assert prediction_output_path.exists()


def test_prediction_output_has_expected_columns():
    result = generate_predictions()

    with open(BEST_MODEL_PATH, "r") as file:
        best_model_metadata = json.load(file)

    model_filename = best_model_metadata["model_filename"]

    # The exact feature columns are stored in the model artifact.
    from src.model_io import load_model

    model_artifact = load_model(PROJECT_ROOT / "models" / model_filename)
    feature_columns = model_artifact["feature_columns"]

    expected_columns = feature_columns + ["prediction"]

    assert list(result.columns) == expected_columns


def test_saved_predictions_match_returned_predictions(prediction_output_path):
    result = generate_predictions()
    saved_result = pd.read_csv(prediction_output_path)

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        saved_result.reset_index(drop=True),
    )