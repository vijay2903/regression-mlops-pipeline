# pandas is used to create a small sample dataset.
import pandas as pd

# We are testing functions from our actual source code.
from src.data_loader import split_data


def test_split_data():
    """
    Test that split_data() correctly separates features and target,
    and creates training and testing sets.
    """

    # Create a small sample dataset.
    # We use a small dataset because unit tests should be fast.
    df = pd.DataFrame({
        "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature_2": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "target": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    })

    # Call the actual split_data() function.
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column="target",
        test_size=0.2,
        random_state=42
    )

    # The target column should not be present in X.
    assert "target" not in X_train.columns
    assert "target" not in X_test.columns

    # Both feature columns should be present.
    assert list(X_train.columns) == ["feature_1", "feature_2"]
    assert list(X_test.columns) == ["feature_1", "feature_2"]

    # The original dataset has 10 rows.
    # With test_size=0.2, we expect:
    # - 8 training rows
    # - 2 testing rows
    assert len(X_train) == 8
    assert len(X_test) == 2

    # The target arrays should have matching row counts.
    assert len(y_train) == 8
    assert len(y_test) == 2


def test_split_data_target_is_separated():
    """
    Test that the target column is correctly separated from features.
    """

    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [5, 6, 7, 8],
        "target": [10, 20, 30, 40],
    })

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column="target",
        test_size=0.25,
        random_state=42
    )

    # The target should not accidentally remain in the feature data.
    assert "target" not in X_train
    assert "target" not in X_test

    # The target values should come from the original target column.
    combined_targets = pd.concat([y_train, y_test])

    assert set(combined_targets) == {10, 20, 30, 40}