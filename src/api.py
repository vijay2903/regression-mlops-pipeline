"""
FastAPI application for serving the best trained regression model.

This file exposes the trained ML model through HTTP endpoints.

Example workflow:

Client sends housing features
        ↓
FastAPI validates the input
        ↓
The best trained model is loaded
        ↓
The model generates a prediction
        ↓
FastAPI returns the prediction as JSON
"""
from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib

# ---------------------------------------------------------
# 1. Define project directories
# ---------------------------------------------------------

# __file__ points to this file:
#
# /app/src/api.py
#
# Path(__file__).resolve().parent gives:
#
# /app/src
#
# .parent again gives:
#
# /app
#
# Therefore, PROJECT_ROOT represents the root of our project.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
BEST_MODEL_METADATA_PATH = RESULTS_DIR / "best_model.json"


#log paths
PREDICTION_LOG_PATH = PROJECT_ROOT / "logs" / "predictions.jsonl"
PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

#Helper

def log_prediction(features: dict, prediction: float) -> None:
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "prediction": float(prediction),
    }

    with PREDICTION_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(log_entry) + "\n")


# ---------------------------------------------------------
# 2. Create the FastAPI application
# ---------------------------------------------------------

# FastAPI() creates the application object.
#
# This object will contain our API endpoints/routes.

app = FastAPI(
    title="California Housing Prediction API",
    description="API for predicting California housing values",
    version="1.0.0"
)

# ---------------------------------------------------------
# 3. Define the expected input structure
# ---------------------------------------------------------

class HousingInput(BaseModel):
    """
    Defines the input data expected by the /predict endpoint
    
    Pydantic uses this class to validate incoming json
    
    If a required field is missing or has an invalid type,
    FastAPI automatically returns a validation error
    """

    MedInc: float = Field(..., ge=0)
    HouseAge: float = Field(..., ge=0, le=100)
    AveRooms: float = Field(..., gt=0)
    AveBedrms: float = Field(..., gt=0)
    Population: float = Field(..., ge=0)
    AveOccup: float = Field(..., gt=0)
    Latitude: float = Field(..., ge=32, le=42)
    Longitude: float = Field(..., ge=-125, le=-114)


# ---------------------------------------------------------
# 4. Load the best trained model
# ---------------------------------------------------------

def load_best_model():
    """
    Load the model selected by the training pipeline
    """

    #check whether the medadata file exists
    if not BEST_MODEL_METADATA_PATH.exists():
        print(
            "Best model metadata not found. "
            "Running the training pipeline..."
        )

        from scripts.train import train_pipeline
        train_pipeline()

    #Read the best-model metadata
    if not BEST_MODEL_METADATA_PATH.exists():
        raise FileNotFoundError(
            "Training completed, but best_model.json "
            "was still not created."
        )

    #Load best model metadata
    with open(BEST_MODEL_METADATA_PATH, "r", encoding="utf-8",) as file:
        metadata = json.load(file)

    #Get the selected model filename
    model_filename = metadata["model_filename"]

    model_path = MODELS_DIR/model_filename

    if not model_path.exists():
        raise FileNotFoundError(
            f"Selected model file not found at: {model_path}"
        )

    #load and return the trained model
    model_artifact = joblib.load(model_path)
    model = model_artifact["model"]

    return model


# ---------------------------------------------------------
# 5. Basic root endpoint
# ---------------------------------------------------------

@app.get("/")
def root():
    """
    A simple endpoint to confirm that the API is running.

    GET means the client is requesting information.
    """
    return {
        "message": "California Housing Prediction API is running"
    }

# ---------------------------------------------------------
# 6. Health-check endpoint
# ---------------------------------------------------------

@app.get("/health")
def health_check():
    """
    Health-check endpoint.

    This is useful for checking whether:
    - The API process is running
    - The model can be loaded

    Monitoring systems and Docker health checks can use
    endpoints like this.
    """
    try:
        #try loading the best model
        model = load_best_model()

        return {
            "status": "healthy",
            "model_loaded": True,
            "model_type": type(model).__name__,
        }

    except Exception as error:
        # HTTPException allows us to return a meaningful
        # HTTP error response to the client.
        raise HTTPException(
            status_code=503,
            detail=f"Model is not available: {str(error)}",
        )

# ---------------------------------------------------------
# 7. Prediction endpoint
# ---------------------------------------------------------

@app.post("/predict")
def predict(input_data: HousingInput):
    """
    Generate a prediction from housing features.

    POST is used because the client sends data to the API.

    FastAPI automatically:
    1. Reads the incoming JSON.
    2. Validates it using HousingInput.
    3. Converts it into a Python object.
    4. Passes it to this function.
    """

    try:
        # Load the best model selected by the training pipeline
        model = load_best_model()

        # Convert the validated Pydantic object into a dictionary.
        #
        # model_dump() is used by Pydantic v2.
        # dict() is used by Pydantic v1.
        #
        # This compatibility check allows the code to work
        # with either version.

        if hasattr(input_data, "model_dump"):
            input_dict = input_data.model_dump()
        else:
            raise ValueError(f"PLEASE FIX ME")

        input_df = pd.DataFrame([input_dict])

        # Generate prediction
        prediction = model.predict(input_df)
        prediction_value = float(prediction[0])
        # Return a JSON-compatible response.
        #
        # float() converts the NumPy/scikit-learn numeric result
        # into a normal Python float.
        log_prediction(
            features=input_data.model_dump(),
            prediction= prediction_value,
            )
        return {
            "prediction" : prediction_value
            }

    except Exception as error:
        # If something goes wrong, return an HTTP 500 error.
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        )
    
