"""Build DataSource from pipeline YAML source block."""

from __future__ import annotations

from src.data.sources.file_source import FileDataSource
from src.data.sources.generator_source import GeneratorDataSource


def build_source(source_cfg: dict):
    stype = source_cfg.get("type", "generator")
    if stype == "generator":
        return GeneratorDataSource(
            fixture_ids=source_cfg.get("fixture_ids"),
            include_game=source_cfg.get("include_game", True),
        )
    if stype == "file":
        return FileDataSource(source_cfg["path"])
    raise ValueError(f"Unsupported source type: {stype}")
