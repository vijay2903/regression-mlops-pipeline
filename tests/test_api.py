from fastapi.testclient import TestClient
from pathlib import Path

from src.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200


def test_prediction_endpoint():
    payload = {
        "MedInc": 5.0,
        "HouseAge": 20.0,
        "AveRooms": 5.0,
        "AveBedrms": 1.0,
        "Population": 500.0,
        "AveOccup": 3.0,
        "Latitude": 34.0,
        "Longitude": -118.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert isinstance(result["prediction"], (int, float))

def test_best_model_metadata_exists():
    model_metadata_path = Path("results/best_model.json")

    assert model_metadata_path.exists()

    metadata = model_metadata_path.read_text(encoding="utf-8")

    assert metadata.strip()