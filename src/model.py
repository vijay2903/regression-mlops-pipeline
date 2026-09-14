from sklearn.linear_model import LinearRegression

def train_model(X_train, y_train):
    """
    Train the baseline Linear Regression model.
    """ 
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


