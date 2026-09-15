import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.validate import validate_training_data
from src.data_loader import load_data, split_data
from src.evaluate import evaluate_model
from src.model import train_model
from src.model_io import save_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_PATH = PROJECT_ROOT / "config.json"

def configure_mlflow():
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "sqlite:////app/mlflow.db",
    )

    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "california-housing-regression-minio",
    )

    artifact_location = os.getenv(
        "MLFLOW_ARTIFACT_LOCATION",
        "s3://mlops-artifacts/mlflow",
    )

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = client.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
        )
        print(f"Created MLflow experiment: {experiment_name}")
        print(f"Experiment ID: {experiment_id}")
    else:
        print(f"Using existing MLflow experiment: {experiment_name}")

    mlflow.set_experiment(experiment_name)

    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLflow artifact URI: {artifact_location}")


def load_config():
    """
    load training configs from config.json
    """

    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)

    return config


def train_pipeline():
    #Step 0 load config
    config = load_config()

    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]
    target_column = config["data"]["target_column"]
    models_config = config["models"]

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

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison_results = []

    # Configure MLflow Experiment
    configure_mlflow()

    #Start an MLflow run:

    for model_config in models_config:
        model_name = model_config["name"]
        model_type = model_config["type"]
        model_filename = model_config["filename"]

        with mlflow.start_run(run_name=model_name):

            #3 Train Model
            model = train_model(model_type, X_train, y_train)
    
            #4 Evaluate Model
            train_results = evaluate_model(model, X_train, y_train)
            test_results = evaluate_model(model, X_test, y_test)
    
            #5 Create model artifact
            model_path = MODELS_DIR / model_filename
            model_artifact = {
                "model": model,
                "model_name": model_name,
                "feature_columns": feature_columns,
                "target_column": target_column
            }

            # Step 6: Save model locally
            save_model(model_artifact, model_path)
    
             #Step 7: Save results
            results = {
                "model": model_name,
                "model_filename": model_filename,
                "train": train_results,
                "test": test_results,
            }
    
            comparison_results.append(results)

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

    
            #Step 12: Print Summary

            run_id = mlflow.active_run().info.run_id


            print("\nTraining complete.")
            print(f"MLflow Run ID: {run_id}")
            print(f"\nCompleted experiment: {model_name}")
            print(f"Model saved to: {model_path}")     
    
            print("\nTest metrics:")
            for metric, value in test_results.items():
                print(f"{metric.upper()}: {value:.4f}")

    comparison_path = RESULTS_DIR / "model_comparison.json"

    with open(comparison_path, "w") as file:
        json.dump(comparison_results, file, indent=4)

    print("\nAll experiments completed.")
    print(f"Comparison results saved to: {comparison_path}")

    #select the best model based on RMSE
    best_model = min(
        comparison_results,
        key=lambda result: result["test"]["rmse"]
    )

    print("Best model:", best_model["model"])
    print("Test RMSE:", best_model["test"]["rmse"])

    best_model_metadata = {
        "model_name": best_model["model"],
        "model_filename": best_model["model_filename"],
        "selection_metric": "rmse",
        "selection_value": best_model["test"]["rmse"],
        "metrics": {
            "train": best_model["train"],
            "test": best_model["test"]
            }
        }

    BEST_MODEL_PATH = RESULTS_DIR / "best_model.json"

    with open(BEST_MODEL_PATH, "w") as f:
        json.dump(best_model_metadata, f, indent=4)

    # Create a separate MLflow run for final comparison and model selection
    with mlflow.start_run(run_name="model-selection-summary"):

        mlflow.log_param(
            "selection_metric",
            "test_rmse"
        )

        mlflow.log_param(
            "selected_model",
            best_model["model"]
        )

        mlflow.log_metric(
            "best_test_rmse",
            best_model["test"]["rmse"]
        )

        # Log summary artifacts using absolute paths
        mlflow.log_artifact(
            str(comparison_path),
            artifact_path="results"
        )

        mlflow.log_artifact(
            str(BEST_MODEL_PATH),
            artifact_path="results"
        )

        print(
            "Logged comparison and best-model metadata "
            "to MLflow."
        )

    print(f"Best model metadata saved to: {BEST_MODEL_PATH}")

def main():
    train_pipeline()
    

if __name__ == "__main__":
    main()


