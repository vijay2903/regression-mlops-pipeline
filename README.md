# California Housing Regression Project

This repository contains an end-to-end machine learning workflow for predicting California housing prices. It includes:

- Data loading and validation
- Training and evaluation of multiple regression models
- Model comparison and selection of the best-performing model
- MLflow experiment tracking
- Batch prediction workflow
- FastAPI-based HTTP inference service
- Docker support for training and prediction services
- CI configuration with GitHub Actions

## Project Overview

The project uses the California Housing dataset and trains configurable regression models defined in `config.json`. By default, it evaluates:

- Linear Regression
- Random Forest Regressor

The training pipeline saves:

- trained model files under `models/`
- comparison results under `results/model_comparison.json`
- selected best-model metadata under `results/best_model.json`

## Repository Structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── input/
│   │   └── prediction_input.csv
│   └── raw/
│       └── california_housing.csv
├── models/
│   ├── baseline_linear_regression.joblib
│   └── random_forest.joblib
├── notebooks/
│   ├── 01_data_loading_and_eda.ipynb
│   └── 02_baseline_model.ipynb
├── results/
│   ├── baseline_results.json
│   ├── best_model.json
│   ├── model_comparison.json
│   └── predictions.csv
├── scripts/
│   ├── predict.py
│   └── train.py
├── src/
│   ├── api.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── model.py
│   ├── model_io.py
│   ├── validate.py
│   └── __init__.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_evaluate.py
│   ├── test_model.py
│   ├── test_model_io.py
│   ├── test_predict.py
│   ├── test_predict_pipeline.py
│   ├── test_train_pipeline.py
│   └── test_validate.py
├── .dockerignore
├── .gitignore
├── config.json
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── mlflow.db
```

## Configuration

The main training and inference settings are stored in `config.json`.

Example configuration:

```json
{
  "data": {
    "target_column": "MedHouseVal",
    "test_size": 0.2,
    "random_state": 42
  },
  "models": [
    {
      "name": "LinearRegression",
      "type": "linear_regression",
      "filename": "baseline_linear_regression.joblib"
    },
    {
      "name": "RandomForest",
      "type": "random_forest",
      "filename": "random_forest.joblib"
    }
  ],
  "paths": {
    "results_filename": "baseline_results.json",
    "prediction_input_filename": "prediction_input.csv",
    "prediction_output_filename": "predictions.csv"
  }
}
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv myenv
```

On Windows PowerShell:

```powershell
.\myenv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Training Pipeline

Run the training workflow:

```bash
python -m scripts.train
```

This workflow:

1. Loads the California Housing dataset
2. Validates the data
3. Splits the data into train/test sets
4. Trains all configured models
5. Evaluates each model
6. Saves artifacts to `models/`
7. Saves comparison results to `results/model_comparison.json`
8. Selects the best model based on RMSE
9. Saves best-model metadata to `results/best_model.json`
10. Logs the run to MLflow

## Prediction Pipeline

Run batch predictions:

```bash
python -m scripts.predict
```

This loads the model selected in `results/best_model.json`, validates the input CSV, generates predictions, and saves them to `results/predictions.csv`.

## FastAPI Service

The API is defined in `src/api.py`.

### Start the API locally

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### API endpoints

- `GET /` — service status message
- `GET /health` — checks whether the best model can be loaded
- `POST /predict` — predicts housing values from JSON input

### Example prediction request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.0,
    "HouseAge": 25,
    "AveRooms": 5.0,
    "AveBedrms": 1.5,
    "Population": 500,
    "AveOccup": 3.0,
    "Latitude": 37.77,
    "Longitude": -122.42
  }'
```

## Docker

The repository includes a `Dockerfile` and `docker-compose.yml` for containerized runs.

### Train in Docker

```bash
docker compose run --rm ml-training
```

### Run the API in Docker

```bash
docker compose up ml-prediction
```

The API will be available at:

```text
http://localhost:8000
```

## MLflow

The training pipeline uses MLflow for experiment tracking and model logging. Artifacts and experiment metadata are stored under `mlruns/`.

## Testing

Run the test suite with:

```bash
pytest -v
```

The project includes tests for:

- data loading
- validation
- evaluation metrics
- model training
- model serialization/deserialization
- prediction pipeline
- training pipeline

## CI

GitHub Actions is configured in `.github/workflows/ci.yml` to:

- set up Python 3.11
- install dependencies
- run `pytest`

## Dependencies

The project dependencies are listed in `requirements.txt` and currently include:

- `joblib`
- `matplotlib`
- `mlflow`
- `numpy`
- `pandas`
- `pytest`
- `scikit-learn`
- `seaborn`
- `fastapi`
- `pydantic`
- `uvicorn`

## Notes

- The dataset is stored in `data/raw/california_housing.csv` and can be downloaded automatically if missing.
- `results/best_model.json` is used by the prediction pipeline and API to select the best-trained model.
- Generated artifacts such as models, results, and MLflow run data are intentionally excluded from version control by `.gitignore`.
