from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def train_model(model_type, X_train, y_train):
    """
    Train the baseline Linear Regression model.
    """ 
    if model_type == "linear_regression":
        model = LinearRegression()

    elif model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

    else:
        raise ValueError(f"Unsupported Model Type: {model_type}")

    model.fit(X_train, y_train)

    return model


