"""
Scenario generation CLI for Scenario Reasoner LM.

Generates a JSONL file of causal/counterfactual scenario instances by sweeping
over a θ-grid of chain lengths, domains, difficulties, and intervention types.

Usage::

    # Quick test — 10 physical direct-chain scenarios
    python scripts/generate_scenarios.py \\
        --chain-lengths 3 5 \\
        --domains physical \\
        --difficulties easy medium \\
        --intervention-types direct \\
        --n-per-combo 10 \\
        --output data/raw/causal/test_scenarios.jsonl

    # Full training grid
    python scripts/generate_scenarios.py \\
        --chain-lengths 3 5 \\
        --domains physical social \\
        --difficulties easy medium \\
        --intervention-types direct counterfactual \\
        --n-per-combo 200 \\
        --output data/raw/causal/train_scenarios.jsonl \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate causal/counterfactual scenario instances"
    )
    parser.add_argument(
        "--chain-lengths", nargs="+", type=int, default=[3, 5],
        metavar="N", help="Chain lengths to include in the θ-grid.",
    )
    parser.add_argument(
        "--domains", nargs="+",
        default=["physical", "social"],
        choices=["physical", "medical", "social", "mechanical"],
        help="Domains to include.",
    )
    parser.add_argument(
        "--difficulties", nargs="+",
        default=["easy", "medium"],
        choices=["easy", "medium", "hard"],
        help="Difficulty levels to include.",
    )
    parser.add_argument(
        "--intervention-types", nargs="+",
        default=["direct", "counterfactual"],
        choices=["direct", "confounded", "counterfactual"],
        help="Causal intervention types.",
    )
    parser.add_argument(
        "--n-per-combo", type=int, default=200,
        help="Number of scenario instances per θ combination.",
    )
    parser.add_argument(
        "--output", type=str,
        default="data/raw/causal/scenarios.jsonl",
        help="Output JSONL file path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.scenarios.causal.generator import CausalScenarioGenerator
    from src.scenarios.causal.taxonomy import CausalThetaSampler

    sampler = CausalThetaSampler(
        chain_length_range=(min(args.chain_lengths), max(args.chain_lengths)),
        intervention_types=args.intervention_types,
        domains=args.domains,
        difficulties=args.difficulties,
        seed=args.seed,
    )
    generator = CausalScenarioGenerator(seed=args.seed, sampler=sampler)

    theta_grid = sampler.grid(
        chain_lengths=args.chain_lengths,
        intervention_types=args.intervention_types,
        domains=args.domains,
        difficulties=args.difficulties,
    )

    logger.info(
        "θ-grid: %d combinations × %d instances = %d total scenarios",
        len(theta_grid),
        args.n_per_combo,
        len(theta_grid) * args.n_per_combo,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(output_path, "w", encoding="utf-8") as fh:
        for theta in theta_grid:
            instances = generator.generate_batch(
                n=args.n_per_combo,
                theta_sampler=lambda t=theta: t,
            )
            for inst in instances:
                fh.write(json.dumps(inst.to_dict(), ensure_ascii=False) + "\n")
            total += len(instances)
            logger.info(
                "  %s | chain=%d | difficulty=%s → %d instances written",
                theta.domain,
                theta.chain_length,
                theta.difficulty,
                len(instances),
            )

    logger.info("Done. %d scenarios written to %s", total, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    main()
