"""
API Tests
"""
import pytest
from fastapi.testclient import TestClient
from src.app import app
from src.train import train_model
import os

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup():
    if not os.path.exists('models/wine_model.pkl'):
        train_model()

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction():
    features = [13.2, 2.77, 2.51, 18.5, 96.0, 1.9, 0.58, 0.28, 0.45, 6.5, 1.05, 3.33, 820.0]
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_invalid_features():
    response = client.post("/predict", json={"features": [1.0, 2.0]})
    assert response.status_code == 400