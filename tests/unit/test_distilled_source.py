"""Tests for distilled DPO pair loader."""

from __future__ import annotations

from pathlib import Path

from src.data.sources.distilled_source import DistilledDataSource


def test_load_dpo_pairs_from_fixture_manifest():
    manifest = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "distilled"
        / "scenario_traces_v1"
        / "manifest.json"
    )
    source = DistilledDataSource(manifest)
    pairs = source.load_dpo_pairs()
    assert len(pairs) == 2
    assert "prompt" in pairs[0]
    assert "chosen" in pairs[0]
    assert "rejected" in pairs[0]
