import pandas as pd
import pytest

from src.model import train_model
from src.model_io import save_model, load_model


def test_save_and_load_model(tmp_path):
    """
    Test that a model artifact can be saved and loaded successfully.

    tmp_path is a built-in pytest fixture.
    It provides a temporary directory for this test.

    The test does not modify the actual models/ directory
    inside our project.
    """

    # Create a small training dataset.
    X_train = pd.DataFrame({
        "feature_1": [1, 2, 3, 4, 5],
        "feature_2": [5, 4, 3, 2, 1],
    })

    y_train = pd.Series([2, 4, 6, 8, 10])

    # Train a model using the actual train_model() function.
    model = train_model(
        X_train,
        y_train
    )

    # Create the same artifact structure used by predict.py.
    #
    # predict.py expects the loaded file to contain:
    # - "model"
    # - "feature_columns"
    # - "target_column"
    model_artifact = {
        "model": model,
        "feature_columns": [
            "feature_1",
            "feature_2",
        ],
        "target_column": "target",
    }

    # Create a temporary path for the model file.
    #
    # tmp_path is automatically created by pytest.
    # The test file will not be saved inside the real models/ folder.
    model_path = tmp_path / "test_model.joblib"

    # Save the model artifact using our actual save_model() function.
    save_model(
        model_artifact,
        model_path
    )

    # Confirm that the model file was created.
    assert model_path.exists()

    # Load the model artifact using our actual load_model() function.
    loaded_artifact = load_model(model_path)

    # Confirm that the expected keys are present.
    assert "model" in loaded_artifact
    assert "feature_columns" in loaded_artifact
    assert "target_column" in loaded_artifact

    # Confirm that the metadata was preserved correctly.
    assert loaded_artifact["feature_columns"] == [
        "feature_1",
        "feature_2",
    ]

    assert loaded_artifact["target_column"] == "target"

    # Confirm that the loaded model can make predictions.
    X_test = pd.DataFrame({
        "feature_1": [6],
        "feature_2": [0],
    })

    predictions = loaded_artifact["model"].predict(X_test)

    # We supplied one test row, so we should receive one prediction.
    assert len(predictions) == 1


def test_load_model_missing_file(tmp_path):
    """
    Test that loading a model file that does not exist
    raises FileNotFoundError.
    """

    # Create a path that does not exist.
    missing_model_path = tmp_path / "does_not_exist.joblib"

    # Our load_model() function is expected to raise
    # FileNotFoundError when the file is missing.
    with pytest.raises(FileNotFoundError):
        load_model(missing_model_path)