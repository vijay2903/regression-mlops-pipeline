import json
from pathlib import Path

from src.data_loader import load_data, split_data
from src.evaluate import evaluate_model
from src.model import train_model
from src.model_io import save_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_PATH = PROJECT_ROOT / "config.json"

def load_config():
    """
    load training configs from config.json
    """

    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)

    return config


def main():
    #Step 0 load config
    config = load_config()

    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]

    model_name = config["model"]["name"]
    model_filename = config["model"]["filename"]
    results_filename = config["paths"]["results_filename"]

    #1. Load data

    df = load_data()

    #2 Split data
    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=test_size,
        random_state=random_state
        )

    #3 Train Model
    model = train_model(X_train, y_train)

    #4 Evaluate Model
    train_results = evaluate_model(model, X_train, y_train)
    test_results = evaluate_model(model, X_test, y_test)

    #5 Create Output Directories
    model_path = MODELS_DIR / model_filename
    save_model(model, model_path)

    #Save results

    results = {
        "model": model_name,
        "train": train_results,
        "test": test_results,
    }

    results_path = RESULTS_DIR / results_filename

    with open(results_path, "w") as file:
        json.dump(results, file, indent=4)

    #8. Print Summary
    print("\nTraining complete.")
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {results_path}")

    print("\nTest metrics:")
    for metric, value in test_results.items():
        print(f"{metric.upper()}: {value:.4f}")

if __name__ == "__main__":
    main()
