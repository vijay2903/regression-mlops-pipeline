# pandas is used to create small sample DataFrames for testing.
import pandas as pd

# pytest provides testing utilities such as pytest.raises().
import pytest

# Import the functions that we want to test from our actual project code.
from src.validate import (
    validate_training_data,
    validate_prediction_data,
)


def test_validate_training_data_success():
    """
    Test that valid training data passes validation.

    Expected behavior:
    - The target column exists.
    - Feature columns exist.
    - All values are numeric.
    - There are no missing values.
    - The function returns the feature column names.
    """

    # Create a small, valid training dataset.
    # We do not need the full California Housing dataset for this test.
    df = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0],
        "feature_2": [4.0, 5.0, 6.0],
        "target": [7.0, 8.0, 9.0],
    })

    # Call the actual function from src/validate.py.
    feature_columns = validate_training_data(
        df,
        target_column="target"
    )

    # assert checks whether a condition is True.
    #
    # If this condition is True, the test passes.
    # If this condition is False, the test fails.
    assert feature_columns == ["feature_1", "feature_2"]


def test_validate_training_data_missing_target():
    """
    Test that training data without the target column is rejected.

    Expected behavior:
    validate_training_data() should raise a ValueError.
    """

    # This DataFrame contains only feature columns.
    # The required target column is missing.
    df = pd.DataFrame({
        "feature_1": [1.0, 2.0],
        "feature_2": [3.0, 4.0],
    })

    # pytest.raises() checks that a specific error is raised.
    #
    # We expect validate_training_data() to raise ValueError.
    # If ValueError is raised, the test passes.
    # If no error is raised, the test fails.
    with pytest.raises(ValueError):
        validate_training_data(
            df,
            target_column="target"
        )


def test_validate_training_data_missing_values():
    """
    Test that training data containing missing values is rejected.

    Expected behavior:
    validate_training_data() should raise a ValueError.
    """

    # feature_1 contains a missing value: None.
    df = pd.DataFrame({
        "feature_1": [1.0, None],
        "target": [3.0, 4.0],
    })

    # The validation function should reject missing values.
    with pytest.raises(ValueError):
        validate_training_data(
            df,
            target_column="target"
        )


def test_validate_prediction_data_success():
    """
    Test that valid prediction data passes validation.

    Prediction data should:
    - contain all expected feature columns;
    - not contain the target column;
    - contain numeric values;
    - not contain missing values.
    """

    # This is valid prediction input.
    # Notice that the target column is not included.
    df = pd.DataFrame({
        "feature_1": [1.0, 2.0],
        "feature_2": [3.0, 4.0],
    })

    # This function should complete without raising an error.
    validate_prediction_data(
        df,
        feature_columns=["feature_1", "feature_2"],
        target_column="target"
    )


def test_validate_prediction_data_missing_column():
    """
    Test that prediction data missing an expected feature is rejected.

    Expected behavior:
    validate_prediction_data() should raise a ValueError.
    """

    # feature_2 is missing from this prediction input.
    df = pd.DataFrame({
        "feature_1": [1.0, 2.0],
    })

    # The validation function should detect the missing column
    # and raise ValueError.
    with pytest.raises(ValueError):
        validate_prediction_data(
            df,
            feature_columns=["feature_1", "feature_2"],
            target_column="target"
        )