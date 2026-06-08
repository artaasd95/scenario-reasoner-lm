"""Service smoke tests for CI (SR-13)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.serving.scenario_service import create_app
from src.serving.schemas import ScenarioArtifact

pytestmark = pytest.mark.smoke


def test_service_generate_evaluate_smoke():
    client = TestClient(create_app(provider_name="mock"))
    assert client.get("/health").json()["status"] == "ok"

    gen = client.post(
        "/scenarios/generate",
        json={"theta": {"chain_length": 3}, "path_type": "bounded", "n_paths": 3},
    )
    assert gen.status_code == 200
    artifact = ScenarioArtifact.model_validate(gen.json())
    assert artifact.verification_summary is not None

    ev = client.post("/scenarios/evaluate", json=gen.json())
    assert ev.status_code == 200

    got = client.get(f"/scenarios/{artifact.id}")
    assert got.status_code == 200
