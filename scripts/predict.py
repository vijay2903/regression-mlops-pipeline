from pathlib import Path
import pandas as pd
import json
from src.model_io import load_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "config.json"
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

def load_config():
    with open(CONFIG_PATH, "r") as file:
        return json.load(file)

def main():

    #step 1: config load
    config = load_config()

    model_filename = config["model"]["filename"]
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
    model = load_model(model_path)

    #Step 3 prediction:

    input_data = pd.read_csv(input_path)

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