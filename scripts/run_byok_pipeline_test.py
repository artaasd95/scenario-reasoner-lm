#!/usr/bin/env python
"""
End-to-end BYOK reasoning pipeline test (inference only, no training).

Reads API credentials from freellmapi.txt (gitignored), wires the custom OpenAI-
compatible adapter, and exercises the main runtime paths:

  1. Provider connectivity (chat completion)
  2. Context budget assembly
  3. LLM-enhanced scenario generation
  4. Reasoning monitors (CoT / ToT / Aha)
  5. Goal-alignment scoring on LLM output
  6. Smoke reasoning eval (fixture baseline, no extra API calls)

Usage:
    python scripts/run_byok_pipeline_test.py
    python scripts/run_byok_pipeline_test.py --output artifacts/byok_pipeline_test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

API_FILE = _REPO_ROOT / "freellmapi.txt"
CONFIG_PATH = _REPO_ROOT / "configs" / "llm_freellm.yaml"
DEFAULT_MODEL = "gpt-oss-20b"


def load_api_file(path: Path = API_FILE) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"API file not found: {path}\n"
            "Create freellmapi.txt with base_url=, api_key=, and model= lines."
        )
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    required = ("base_url", "api_key")
    missing = [k for k in required if k not in out]
    if missing:
        raise ValueError(f"freellmapi.txt missing required keys: {missing}")
    return out


def apply_api_env(api: dict[str, str]) -> str:
    os.environ["FREELLMAPI_KEY"] = api["api_key"]
    os.environ["ALLOW_LIVE_PROVIDER"] = "1"
    os.environ["LLM_PROVIDER"] = "custom"
    return api.get("model", DEFAULT_MODEL)


def _step(name: str, fn):
    started = time.perf_counter()
    try:
        result = fn()
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {"step": name, "status": "pass", "elapsed_ms": round(elapsed_ms, 1), "result": result}
    except Exception as exc:  # noqa: BLE001 — collect all step failures
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "step": name,
            "status": "fail",
            "elapsed_ms": round(elapsed_ms, 1),
            "error": f"{type(exc).__name__}: {exc}",
        }


async def _test_provider_complete(model_id: str) -> dict[str, Any]:
    from src.llm_integration.factory import create_llm_provider

    provider = create_llm_provider(CONFIG_PATH)
    prompt = (
        "You are testing a causal reasoning pipeline. In 2-3 sentences, explain "
        "how a supply-chain shock can cascade to enterprise risk. Use Step 1 / Step 2 / Therefore."
    )
    completion = await provider.complete(prompt, model_id=model_id, max_tokens=256)
    assert completion.text.strip(), "empty completion"
    assert completion.backend_id == "custom"
    return {
        "backend_id": completion.backend_id,
        "model_id": completion.model_id,
        "latency_ms": round(completion.latency_ms, 1),
        "token_usage": completion.token_usage,
        "text_preview": completion.text[:300],
    }


def _test_context_assembly(model_id: str) -> dict[str, Any]:
    from src.llm_integration.config import LLMConfig
    from src.llm_integration.context import ContextSegment, assemble_context
    import yaml

    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    config = LLMConfig.model_validate(raw)
    segments = [
        ContextSegment(name="system", content="You are a scenario reasoner.", priority=0, protected=True),
        ContextSegment(name="goal", content="Assess Taiwan foundry supply risk.", priority=1),
        ContextSegment(name="evidence", content="10-K cites sole-source ASIC partner.", priority=2),
        ContextSegment(name="history", content="Prior turn: baseline risk noted.", priority=3),
    ]
    assembled = assemble_context(segments, model_id, config=config)
    assert assembled.text, "assembled context empty"
    return assembled.to_log_dict()


def _test_scenario_generation(model_id: str) -> dict[str, Any]:
    from src.llm_integration.factory import create_llm_provider
    from src.scenarios.causal.generator import CausalScenarioGenerator
    from src.scenarios.llm_generator import LLMEnhancedScenarioGenerator

    provider = create_llm_provider(CONFIG_PATH)
    base = CausalScenarioGenerator(seed=42)
    generator = LLMEnhancedScenarioGenerator(base, provider, model_id=model_id)
    instances = generator.generate_batch(n=2)
    assert len(instances) == 2
    rows = []
    for inst in instances:
        assert inst.reasoning_trace, "missing reasoning_trace from LLM"
        assert inst.metadata.get("llm_provider") == "custom"
        rows.append(
            {
                "prompt_preview": inst.prompt[:120],
                "answer": inst.answer,
                "reasoning_preview": inst.reasoning_trace[:200],
                "llm_model_id": inst.metadata.get("llm_model_id"),
            }
        )
    return {"count": len(rows), "instances": rows}


def _test_monitors(reasoning_text: str) -> dict[str, Any]:
    from src.monitoring.aha_monitor import AhaMonitor
    from src.monitoring.cot_monitor import CoTMonitor
    from src.monitoring.tot_monitor import ToTMonitor

    cot = CoTMonitor(log_every=0)
    tot = ToTMonitor(log_every=0)
    aha = AhaMonitor(log_every=0)
    cot_traces = cot.update([reasoning_text])
    tot_traces = tot.update([reasoning_text])
    aha_traces = aha.update([reasoning_text])
    return {
        "has_cot": cot_traces[0].has_cot,
        "has_tot": tot_traces[0].has_tot,
        "has_aha": aha_traces[0].has_aha,
    }


def _test_goal_alignment(reasoning_text: str) -> dict[str, float]:
    from src.metrics.goal_preservation_metrics import score_goal_alignment

    goal = "Explain supply-chain cascade to enterprise financial risk"
    theta = {"domain": "enterprise", "chain_length": 3}
    scores = score_goal_alignment(reasoning_text, goal, theta)
    assert "on_target_composite" in scores
    return scores


def _test_smoke_reasoning_eval() -> dict[str, Any]:
    from src.eval.scenario_reasoning_eval import run_smoke_reasoning_eval

    report = run_smoke_reasoning_eval(seed=42)
    return {
        "smoke_mode": report.metadata.smoke_mode,
        "provider_mode": report.metadata.provider_mode,
        "theta_slice_count": len(report.per_theta_slice),
        "aggregate": report.aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="BYOK reasoning pipeline integration test")
    parser.add_argument("--api-file", default=str(API_FILE), help="Path to freellmapi.txt")
    parser.add_argument("--output", default="artifacts/byok_pipeline_test", help="Report output directory")
    args = parser.parse_args()

    api = load_api_file(Path(args.api_file))
    model_id = apply_api_env(api)

    report: dict[str, Any] = {
        "schema": "byok_pipeline_test_v1",
        "model_id": model_id,
        "config": str(CONFIG_PATH.relative_to(_REPO_ROOT)),
        "base_url": api["base_url"],
        "steps": [],
    }

    async def run_provider():
        return await _test_provider_complete(model_id)

    report["steps"].append(_step("provider_complete", lambda: asyncio.run(run_provider())))
    provider_step = report["steps"][-1]
    reasoning_sample = ""
    if provider_step["status"] == "pass":
        reasoning_sample = provider_step["result"]["text_preview"]

    report["steps"].append(_step("context_assembly", lambda: _test_context_assembly(model_id)))
    report["steps"].append(_step("llm_scenario_generation", lambda: _test_scenario_generation(model_id)))

    if reasoning_sample:
        report["steps"].append(
            _step("reasoning_monitors", lambda: _test_monitors(reasoning_sample))
        )
        report["steps"].append(
            _step("goal_alignment", lambda: _test_goal_alignment(reasoning_sample))
        )

    report["steps"].append(_step("smoke_reasoning_eval", _test_smoke_reasoning_eval))

    passed = sum(1 for s in report["steps"] if s["status"] == "pass")
    failed = sum(1 for s in report["steps"] if s["status"] == "fail")
    report["summary"] = {"passed": passed, "failed": failed, "total": len(report["steps"])}

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "byok_pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["summary"], indent=2))
    for step in report["steps"]:
        status = step["status"].upper()
        print(f"  [{status}] {step['step']} ({step['elapsed_ms']} ms)")
        if step["status"] == "fail":
            print(f"         {step['error']}")

    print(f"\nReport: {report_path}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
