"""Schema validation tests (SR-09)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.serving.schemas import PathType, ScenarioArtifact, ScenarioRequest, TailTag


class TestServingSchemas:
    def test_scenario_request_valid(self):
        req = ScenarioRequest(theta={"chain_length": 3}, path_type="bounded", n_paths=2)
        assert req.n_paths == 2

    def test_scenario_request_n_paths_bounds(self):
        with pytest.raises(ValidationError):
            ScenarioRequest(theta={}, n_paths=0)

    def test_path_type_enum(self):
        assert PathType.wide.value == "wide"

    def test_tail_tag_enum(self):
        assert TailTag.tail.value == "tail"

    def test_artifact_roundtrip(self):
        from src.serving.providers import MockScenarioProvider

        artifact = MockScenarioProvider().generate(
            ScenarioRequest(theta={"domain": "physical"}, path_type="bounded", n_paths=2)
        )
        restored = ScenarioArtifact.model_validate_json(artifact.model_dump_json())
        assert restored.id == artifact.id
        assert len(restored.paths) == 2
