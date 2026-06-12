"""Scenario generation providers — mock (default) and gate-checked mock (live alias)."""

from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, Optional
from uuid import uuid4

from src.serving.schemas import (
    ClaimRecord,
    ClaimStatus,
    PathRecord,
    Provenance,
    ScenarioArtifact,
    ScenarioRequest,
    TailTag,
    VerificationSummary,
)


class ScenarioProvider(ABC):
    @abstractmethod
    def generate(self, request: ScenarioRequest) -> ScenarioArtifact:
        ...

    @abstractmethod
    def evaluate(self, artifact: ScenarioArtifact) -> ScenarioArtifact:
        ...

    def get(self, scenario_id: str) -> ScenarioArtifact | None:
        """Return a previously generated artifact, if the provider caches them."""
        return None


class MockScenarioProvider(ScenarioProvider):
    """Deterministic mock provider — no API keys required."""

    def __init__(self, store: Dict[str, ScenarioArtifact] | None = None) -> None:
        self._store = store if store is not None else {}

    def generate(self, request: ScenarioRequest) -> ScenarioArtifact:
        paths = []
        tail_tags: list[str] = []
        for i in range(request.n_paths):
            feas = 0.75 + (i * 0.05)
            tag = TailTag.none if i == 0 else TailTag.tail
            tail_tags.append(tag.value)
            paths.append(
                PathRecord(
                    path_id=f"path_{i}",
                    feasibility=min(feas, 1.0),
                    tail_tag=tag,
                    primary_quality_score=0.7 + i * 0.03,
                    theta_stratum=str(request.theta.get("domain", "causal")),
                    text=f"CVaR {10.0 + i:.1f}% drawdown {5.0 + i:.1f}% on path {i}",
                )
            )
        artifact = ScenarioArtifact(
            id=str(uuid4()),
            theta=request.theta,
            path_type=request.path_type,
            n_paths=request.n_paths,
            paths=paths,
            coherence=0.82,
            feasibility=sum(p.feasibility for p in paths) / len(paths),
            tail_tags=tail_tags,
            provenance=Provenance(
                provider="mock",
                datasource_id="mock_generator",
                generated_at=datetime.now(timezone.utc).isoformat(),
            ),
            verification_summary=self._verify(paths),
        )
        self._store[artifact.id] = artifact
        return artifact

    def evaluate(self, artifact: ScenarioArtifact) -> ScenarioArtifact:
        artifact.coherence = min(1.0, artifact.coherence + 0.02)
        artifact.verification_summary = self._verify(artifact.paths)
        self._store[artifact.id] = artifact
        return artifact

    def get(self, scenario_id: str) -> ScenarioArtifact | None:
        return self._store.get(scenario_id)

    def _verify(self, paths: list[PathRecord]) -> VerificationSummary:
        try:
            from src.verification.report import build_verification_report

            texts = [p.text for p in paths if p.text]
            return build_verification_report(texts)
        except ImportError:
            claims = [
                ClaimRecord(text=p.text, status=ClaimStatus.UNVERIFIED, detail="verifier unavailable")
                for p in paths
                if p.text
            ]
            return VerificationSummary(overall_score=0.5, claims=claims)


class PermissionCheckedMockProvider(ScenarioProvider):
    """
    Mock provider that requires ``ALLOW_LIVE_PROVIDER=1``.

    Does not call a live LLM — provenance is ``gate_checked_mock`` so consumers
  are not misled into believing output came from a paid API.
    """

    def __init__(self) -> None:
        if os.environ.get("ALLOW_LIVE_PROVIDER", "0") != "1":
            raise PermissionError(
                "Gate-checked mock provider requires ALLOW_LIVE_PROVIDER=1"
            )
        self._mock = MockScenarioProvider()

    def generate(self, request: ScenarioRequest) -> ScenarioArtifact:
        artifact = self._mock.generate(request)
        artifact.provenance.provider = "gate_checked_mock"
        return artifact

    def evaluate(self, artifact: ScenarioArtifact) -> ScenarioArtifact:
        return self._mock.evaluate(artifact)

    def get(self, scenario_id: str) -> ScenarioArtifact | None:
        return self._mock.get(scenario_id)


# Backward-compatible alias; prefer PermissionCheckedMockProvider.
class LiveScenarioProvider(PermissionCheckedMockProvider):
    """Deprecated alias for :class:`PermissionCheckedMockProvider`."""

    def __init__(self) -> None:
        warnings.warn(
            "LiveScenarioProvider is deprecated; use PermissionCheckedMockProvider. "
            "The 'live' name does not invoke a real LLM.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()


def get_provider(name: str = "mock") -> ScenarioProvider:
    if name == "live":
        return PermissionCheckedMockProvider()
    return MockScenarioProvider()
