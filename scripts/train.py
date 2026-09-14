import json
from pathlib import Path

import mlflow
import mlflow.sklearn

from src.validate import validate_training_data
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
    target_column = config["data"]["target_column"]


    # Configure MLflow Experiment
    mlflow.set_experiment("california-housing-regression")

    #Start an MLflow run:

    with mlflow.start_run(run_name=model_name):

        #1. Load data & validate
        df = load_data()
    
        feature_columns = validate_training_data(
            df,
            target_column
        )
    
        #2 Split data
        X_train, X_test, y_train, y_test = split_data(
            df,
            target_column=target_column,
            test_size=test_size,
            random_state=random_state
        )
    
        #3 Train Model
        model = train_model(X_train, y_train)
    
        #4 Evaluate Model
        train_results = evaluate_model(model, X_train, y_train)
        test_results = evaluate_model(model, X_test, y_test)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)  
    
        #5 Create model artifact
        model_path = MODELS_DIR / model_filename
        model_artifact = {
            "model": model,
            "feature_columns": feature_columns,
            "target_column": target_column
        }

        # Step 6: Save model locally
        save_model(model_artifact, model_path)
    
        #Step 7: Save results
        results = {
            "model": model_name,
            "train": train_results,
            "test": test_results,
        }
    
        results_path = RESULTS_DIR / results_filename
    
        with open(results_path, "w") as file:
            json.dump(results, file, indent=4)

        #step 8: log parameters to MLflow
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("target_column", target_column)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("feature_count", len(feature_columns))

        #Step 9: Log metrics ot MLflow
        mlflow.log_metric("train_mae", train_results["mae"])
        mlflow.log_metric("train_rmse", train_results["rmse"])
        mlflow.log_metric("train_r2", train_results["r2"])

        mlflow.log_metric("test_mae", test_results["mae"])
        mlflow.log_metric("test_rmse", test_results["rmse"])
        mlflow.log_metric("test_r2", test_results["r2"])

        #Step 10: log model to MLFlow

        mlflow.sklearn.log_model(
            sk_model=model,
            name=model_name
        )

        #Step 11: Log local files as MLflow artifacts
        mlflow.log_artifact(
            str(model_path),
            artifact_path="local_model"
        )

        mlflow.log_artifact(
            str(results_path),
            artifact_path="results"
        )
    
        #Step 12: Print Summary

        run_id = mlflow.active_run().info.run_id


        print("\nTraining complete.")
        print(f"MLflow Run ID: {run_id}")
        print(f"Model saved to: {model_path}")
        print(f"Results saved to: {results_path}")
    
        print("\nTest metrics:")
        for metric, value in test_results.items():
            print(f"{metric.upper()}: {value:.4f}")
    

if __name__ == "__main__":
    main()


