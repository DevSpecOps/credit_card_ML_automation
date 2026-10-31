from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)

VALID_PAYLOAD = {
    "LIMIT_BAL": 50000,
    "AGE": 30,
    "SEX": 1,
    "EDUCATION": 1,
    "MARRIAGE": 1,
    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 1000,
    "BILL_AMT2": 1000,
    "BILL_AMT3": 1000,
    "BILL_AMT4": 1000,
    "BILL_AMT5": 1000,
    "BILL_AMT6": 1000,
    "PAY_AMT1": 500,
    "PAY_AMT2": 500,
    "PAY_AMT3": 500,
    "PAY_AMT4": 500,
    "PAY_AMT5": 500,
    "PAY_AMT6": 500,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("src.api.app.get_model")
def test_predict_valid(mock_get_model):
    # Создаем фейковую модель, которая всегда выдает вероятность 0.2
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
    mock_get_model.return_value = mock_model

    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    data = response.json()
    assert "default_prediction" in data
    assert "default_probability" in data
    assert data["default_prediction"] == 0
    assert data["default_probability"] == 0.2


def test_predict_invalid_age():
    bad_payload = VALID_PAYLOAD.copy()
    bad_payload["AGE"] = 150  # Больше 100 (Pydantic должен отклонить)
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422  # Unprocessable Entity
