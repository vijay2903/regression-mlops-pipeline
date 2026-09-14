import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, X, y):
    """
    Evaluate a trained model and return regression metrics.
    """

    predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }

