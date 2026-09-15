import pandas as pd

# Import the function we want to test.
from src.model import train_model


def test_train_model_returns_fitted_model():
    """
    Test that train_model() returns a fitted model.

    A fitted sklearn model should have learned coefficients.
    """

    # Create a simple relationship:
    # target = 2 * feature + 1
    X_train = pd.DataFrame({
        "feature": [1, 2, 3, 4, 5],
    })

    y_train = pd.Series([3, 5, 7, 9, 11])

    # Train the model.
    model = train_model("linear_regression", X_train, y_train)

    # LinearRegression should have a coef_ attribute after fitting.
    assert hasattr(model, "coef_")

    # It should also have an intercept_ attribute.
    assert hasattr(model, "intercept_")

def test_train_random_forest_returns_fitted_model():
    """
    Test that train_model() can train a Random Forest model.
    """

    X_train = pd.DataFrame({
        "feature": [1, 2, 3, 4, 5],
    })

    y_train = pd.Series([3, 5, 7, 9, 11])

    model = train_model(
        "random_forest",
        X_train,
        y_train
    )

    # A fitted RandomForestRegressor has estimators_.
    assert hasattr(model, "estimators_")

    # It should be able to generate predictions.
    predictions = model.predict(X_train)

    assert len(predictions) == len(y_train)


def test_train_model_can_predict():
    """
    Test that the trained model can make predictions.
    """

    X_train = pd.DataFrame({
        "feature": [1, 2, 3, 4, 5],
    })

    y_train = pd.Series([3, 5, 7, 9, 11])

    model = train_model("linear_regression", X_train, y_train)

    # Predict for two new observations.
    X_test = pd.DataFrame({
        "feature": [6, 7],
    })

    predictions = model.predict(X_test)

    # The expected predictions are:
    # feature=6 → 13
    # feature=7 → 15
    assert len(predictions) == 2
    assert predictions[0] == 13
    assert predictions[1] == 15