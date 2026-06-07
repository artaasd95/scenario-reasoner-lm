"""Integration test for pre vs post train comparison harness."""

from __future__ import annotations

from pathlib import Path

from src.evaluation.pre_post_compare import PrePostCompareConfig, run_comparison


def test_pre_post_compare_produces_delta_table(tmp_path):
    config = PrePostCompareConfig(
        model_id="qwen3-0.6b",
        adapter_path=str(tmp_path / "fake_adapter"),
        fixtures_path="data/eval/simulation_fixtures.json",
        seed=42,
        training_method="unsloth",
        offline=True,
    )
    out = tmp_path / "comparison"
    report = run_comparison(config, out)

    md_path = out / "comparison_report.md"
    assert md_path.is_file()
    md = md_path.read_text(encoding="utf-8")
    assert "Pre vs Post Train Comparison" in md
    assert "Delta" in md
    assert report.model_id == "qwen3-0.6b"
    assert (out / "eval_results.json").is_file()
    assert (out / "pre_report.json").is_file()
    assert (out / "post_report.json").is_file()

    if report.aggregate_deltas:
        assert any(v != 0 for v in report.aggregate_deltas.values())
