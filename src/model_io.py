from pathlib import Path
import joblib

def save_model(model, model_path):
    """
    Save a trained model to disk.
    """

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)

    print(f"Model Saved to: {model_path}")

def load_model(model_path):
    """
    Load a saved model from disk.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model = joblib.load(model_path)

    print(f"Model Loaded From: {model_path}")

    return model

