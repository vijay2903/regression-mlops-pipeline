import pandas as pd
import pytest

from src.model import train_model
from src.evaluate import evaluate_model


def test_evaluate_model_returns_expected_metrics():
    """
    Test that evaluate_model() returns MAE, RMSE, and R².
    """

    # Create a simple perfectly linear dataset.
    X_train = pd.DataFrame({
        "feature": [1, 2, 3, 4, 5],
    })

    y_train = pd.Series([3, 5, 7, 9, 11])

    model = train_model(X_train, y_train)

    # Use the same relationship for evaluation.
    X_test = pd.DataFrame({
        "feature": [6, 7, 8],
    })

    y_test = pd.Series([13, 15, 17])

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    # Check that all expected metric names exist.
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics

    # Because the relationship is perfectly linear,
    # predictions should be exactly correct.
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


def test_evaluate_model_with_prediction_error():
    """
    Test that evaluate_model() calculates non-zero errors
    when predictions are not perfect.
    """

    X_train = pd.DataFrame({
        "feature": [1, 2, 3, 4, 5],
    })

    y_train = pd.Series([3, 5, 7, 9, 11])

    model = train_model(X_train, y_train)

    X_test = pd.DataFrame({
        "feature": [6, 7, 8],
    })

    # These values intentionally contain errors.
    y_test = pd.Series([14, 14, 20])

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    # Errors should be greater than zero.
    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0

    # R² should be less than 1 for imperfect predictions.
    assert metrics["r2"] < 1