from pathlib import Path
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "california_housing.csv"

def load_data(raw_data_path = RAW_DATA_PATH, raw_data_dir = RAW_DATA_DIR):
    """
    Load the californina Housing Dataset
    Returns 
    df: pandas.Dataframe
        Complete dataset containing features and target
    """
    if raw_data_path.exists():
        print(f"Loading Dataset Form: {raw_data_path}")
        return pd.read_csv(raw_data_path)

    print("Raw dataset not found. Downloading dataset...")

    data = fetch_california_housing(as_frame=True)
    df = data.frame

    raw_data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_data_path, index=False)

    print(f"Dataset saved to: {raw_data_path}")

    return df

def split_data(df, target_column, test_size = 0.2, random_state=42):
    """
    Separate features and target, then split into train and test sets.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing the target column 'MedHouseVal'.

    test_size : float, default=0.2
        Proportion of data reserved for testing.

    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    print("Splitting Done")

    return X_train, X_test, y_train, y_test

