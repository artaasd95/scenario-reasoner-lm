"""FastAPI endpoint tests (SR-10)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.serving.scenario_service import create_app


@pytest.fixture
def client():
    return TestClient(create_app(provider_name="mock"))


class TestServiceEndpoints:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_generate(self, client):
        resp = client.post(
            "/scenarios/generate",
            json={"theta": {"chain_length": 3}, "path_type": "bounded", "n_paths": 2},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_paths"] == 2
        assert len(data["paths"]) == 2

    def test_generate_validation_error(self, client):
        resp = client.post("/scenarios/generate", json={"theta": {}, "n_paths": 0})
        assert resp.status_code == 422

    def test_get_not_found(self, client):
        resp = client.get("/scenarios/nonexistent-id")
        assert resp.status_code == 404

    def test_get_after_generate(self, client):
        gen = client.post(
            "/scenarios/generate",
            json={"theta": {"domain": "physical"}, "path_type": "bounded", "n_paths": 1},
        )
        artifact = gen.json()
        resp = client.get(f"/scenarios/{artifact['id']}")
        assert resp.status_code == 200
        assert resp.json()["id"] == artifact["id"]

    def test_evaluate_roundtrip(self, client):
        gen = client.post(
            "/scenarios/generate",
            json={"theta": {"domain": "physical"}, "path_type": "bounded", "n_paths": 1},
        )
        artifact = gen.json()
        resp = client.post("/scenarios/evaluate", json=artifact)
        assert resp.status_code == 200
        assert resp.json()["coherence"] >= artifact["coherence"]
