from pathlib import Path
import pandas as pd
import json
from src.model_io import load_model
from src.validate import validate_prediction_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "config.json"
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
BEST_MODEL_PATH = RESULTS_DIR / "best_model.json"

def load_config():
    with open(CONFIG_PATH, "r") as file:
        return json.load(file)

def main():

    #step 1: config load
    config = load_config()

    with open(BEST_MODEL_PATH, "r") as f:
        best_model_metadata = json.load(f)

    model_filename = best_model_metadata["model_filename"]
    model_path = MODELS_DIR / model_filename
    input_filename = config["paths"]["prediction_input_filename"]
    output_filename = config["paths"]["prediction_output_filename"]

    model_path = MODELS_DIR / model_filename
    input_path = INPUT_DIR / input_filename
    output_path = OUTPUT_DIR / output_filename

    if not input_path.exists():
        raise FileNotFoundError(
            f"Prediction input file not found: {input_path}"
        )

    #Step 2: load model
    model_artifact = load_model(model_path)

    model = model_artifact["model"]
    
    #Step 3 prediction:
    feature_columns = model_artifact["feature_columns"]
    target_column = model_artifact["target_column"]
    
    input_data = pd.read_csv(input_path)
    
    validate_prediction_data(
        input_data,
        feature_columns,
        target_column
    )

    predictions = model.predict(input_data)

    prediction_data = input_data.copy()
    prediction_data["prediction"] = predictions

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_data.to_csv(output_path, index=False)

    print("\nPredictions:")
    print(prediction_data)

    print(f"\nPredictions saved to: {output_path}")

if __name__ == "__main__":
    main()