"""Demo: generate scenarios using selected LLM provider adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_integration.factory import create_llm_provider_for_name
from src.scenarios.causal.generator import CausalScenarioGenerator
from src.scenarios.llm_generator import LLMEnhancedScenarioGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Scenario LLM generation demo")
    parser.add_argument("--adapter", default="mock", help="Provider alias or config path")
    parser.add_argument("--context", default="market stress", help="Demo context label")
    parser.add_argument("--model-id", default="demo-model", help="Generation model id")
    parser.add_argument("--count", type=int, default=3, help="Number of scenarios")
    args = parser.parse_args()

    provider = create_llm_provider_for_name(args.adapter)
    base = CausalScenarioGenerator(seed=42)
    generator = LLMEnhancedScenarioGenerator(base, provider, model_id=args.model_id)
    instances = generator.generate_batch(n=args.count)

    out = Path("demo_test_output") / "scenario_llm_demo.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for inst in instances:
            row = inst.to_dict()
            row["context"] = args.context
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Generated {len(instances)} scenarios with adapter {args.adapter}; output={out}")


if __name__ == "__main__":
    main()
