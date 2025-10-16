from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_schema():
    response = client.post("/predict", json={"features": [0, 1, 2]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] == 1.0
