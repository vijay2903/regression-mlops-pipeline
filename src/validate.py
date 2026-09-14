import pandas as pd

def validate_training_data(df, target_column):
    if target_column not df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in training data."
        )
    print(f"Target column '{target_column}' not found in training data.")

    feature_columns = [col: col in df.colums
                       if col != target_column
                       ]

    if not feature_columns:
        raise ValueError("No feature columns found in training data.")

    validate_numeric_columns(df, feature_columns + [target_column])
    validate_missing_values(df, feature_columns + [target_column])

    print("Training data validation passed.")

    return feature_columns

def validate_prediction_data(df, feature_columns, target_column):
    
    actual_columns = list(df.columns)

    missing_columns = [
        column for column in feature_columns
        if column not in actual_columns
    ]

    extra_columns = [
        column for column in actual_columns
        if column not in feature_columns
    ]

    if missing_columns:
        raise ValueError(
            f"Prediction data is missing columns: {missing_columns}"
        )

    if extra_columns:
        raise ValueError(
            f"Prediction data contains unexpected columns: {extra_columns}"
        )

    validate_numeric_columns(df, feature_columns)
    validate_missing_values(df, feature_columns)

    print("Prediction data validation passed.")


def validate_numeric_columns(df, columns):
    for column in columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(
                f"Column '{column}' must contain numeric values."
            )


def validate_missing_values(df, columns):
    missing_values = df[columns].isnull().sum()
    columns_with_missing_values = missing_values[
        missing_values > 0
    ]

    if not columns_with_missing_values.empty:
        raise ValueError(
            "Missing values found:\n"
            f"{columns_with_missing_values.to_string()}"
        )