from fastapi.testclient import TestClient
from app.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


def test_rag_api():
    with TestClient(app) as client:
        response = client.post("/api/v1/rag/search", json={"query": "bathroom water leak", "top_k": 2})
        assert response.status_code == 200
        assert response.json()[0]["document_id"] == "engineering_water_leak"
