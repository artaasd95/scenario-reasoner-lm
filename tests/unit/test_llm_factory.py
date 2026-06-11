from __future__ import annotations

from pathlib import Path

import pytest

from src.llm_integration.factory import create_llm_provider


@pytest.mark.asyncio
async def test_factory_mock_yaml(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text("provider: mock\nmodel_id: test\n", encoding="utf-8")
    provider = create_llm_provider(cfg)
    out = await provider.complete("ping", "test")
    assert out.model_id == "test"
