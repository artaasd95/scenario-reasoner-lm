"""FastAPI scenario service (SR-10)."""

from __future__ import annotations

from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.serving.providers import ScenarioProvider, get_provider
from src.serving.schemas import ScenarioArtifact, ScenarioRequest


def create_app(provider_name: str = "mock") -> FastAPI:
    app = FastAPI(title="Scenario Reasoner API", version="0.2.0")
    provider: ScenarioProvider = get_provider(provider_name)

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/scenarios/generate", response_model=ScenarioArtifact)
    def generate_scenarios(request: ScenarioRequest) -> ScenarioArtifact:
        try:
            return provider.generate(request)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.post("/scenarios/evaluate", response_model=ScenarioArtifact)
    def evaluate_scenario(artifact: ScenarioArtifact) -> ScenarioArtifact:
        return provider.evaluate(artifact)

    @app.get("/scenarios/{scenario_id}", response_model=ScenarioArtifact)
    def get_scenario(scenario_id: str) -> ScenarioArtifact:
        found = provider.get(scenario_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"Scenario {scenario_id} not found")
        return found

    return app


class ServiceConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    provider: str = "mock"
    log_level: str = "info"
