"""Shared training helpers — config resolution, policy, data loading."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.data.sources.distilled_source import DistilledDataSource
from src.models.model_registry import ModelRegistry
from src.training.policies.registry import PolicyRegistry


def resolve_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve model paths and merge policy reward weights into config."""
    cfg = dict(config)
    registry = ModelRegistry.load()

    if cfg.get("model_id") or (
        cfg.get("model_name_or_path")
        and "/" in str(cfg.get("model_name_or_path", ""))
        and not str(cfg.get("model_name_or_path", "")).startswith("/")
    ):
        try:
            resolved = registry.resolve_model_name_or_path(cfg)
            cfg["model_name_or_path"] = resolved
            if cfg.get("model_id"):
                entry = registry.get(cfg["model_id"])
                cfg["hub_id"] = entry.hub_id
                cfg.setdefault("model_tier", entry.tier)
        except (EnvironmentError, KeyError, FileNotFoundError):
            if cfg.get("model_id"):
                raise
            # Legacy hub path without local cache — keep as-is

    training_cfg = cfg.get("training", {})
    policy_name = training_cfg.get("policy")
    if policy_name:
        policy = PolicyRegistry.with_defaults().get(policy_name)
        cfg.setdefault("reward_weights", policy.reward_weights)

    return cfg


def load_preference_data(
    config: Dict[str, Any],
    *,
    model=None,
    tokenizer=None,
    train_dataset=None,
) -> List[Dict[str, str]]:
    """Build or load DPO preference pairs based on training.data_source."""
    training_cfg = config.get("training", {})
    data_source = training_cfg.get("data_source", "inline")

    if data_source == "distilled":
        manifest = training_cfg.get("distilled_manifest")
        if not manifest:
            raise ValueError(
                "training.distilled_manifest required when data_source=distilled"
            )
        return DistilledDataSource(manifest).load_dpo_pairs()

    if train_dataset is None or model is None or tokenizer is None:
        raise ValueError("inline data_source requires model, tokenizer, and train_dataset")

    from src.training.causal_reward import CausalRewardFunction
    from src.training.preference_builder import PreferenceBuilder
    from src.training.reward_composer import RewardComposer

    reward_weights = config.get("reward_weights", {})
    reward_fn = CausalRewardFunction()
    composer = RewardComposer(
        task_reward_fn=reward_fn,
        alpha=reward_weights.get("alpha_cot", 0.15),
        beta=reward_weights.get("beta_tot", 0.10),
        gamma=reward_weights.get("gamma_aha", 0.05),
    )
    pb_cfg = config.get("preference_builder", {})
    builder = PreferenceBuilder(
        model=model,
        tokenizer=tokenizer,
        reward_composer=composer,
        num_samples=pb_cfg.get("num_samples", 4),
        temperature=pb_cfg.get("temperature", 0.8),
        max_new_tokens=pb_cfg.get("max_new_tokens", 512),
    )
    prompts = [train_dataset[i]["input"] for i in range(len(train_dataset))]
    expected_answers = [train_dataset[i]["output"] for i in range(len(train_dataset))]
    thetas = [train_dataset[i].get("theta") for i in range(len(train_dataset))]
    return builder.build_dataset(
        prompts=prompts,
        expected_answers=expected_answers,
        thetas=thetas,
    )


def build_inline_train_dataset(config: Dict[str, Any]):
    """Generate causal scenarios for inline training data_source."""
    from src.data.causal_dataset import CausalReasoningDataset
    from src.scenarios.causal.generator import CausalScenarioGenerator
    from src.scenarios.causal.taxonomy import CausalThetaSampler

    scenario_cfg = config.get("scenario", {})
    sampler = CausalThetaSampler(
        chain_length_range=(
            min(scenario_cfg.get("chain_lengths", [3])),
            max(scenario_cfg.get("chain_lengths", [5])),
        ),
        intervention_types=scenario_cfg.get("intervention_types"),
        domains=scenario_cfg.get("domains"),
        difficulties=scenario_cfg.get("difficulties"),
        seed=config.get("data", {}).get("seed", 42),
    )
    generator = CausalScenarioGenerator(
        seed=config.get("data", {}).get("seed", 42),
        sampler=sampler,
    )
    theta_grid = sampler.grid(
        chain_lengths=scenario_cfg.get("chain_lengths", [3, 5]),
        intervention_types=scenario_cfg.get("intervention_types"),
        domains=scenario_cfg.get("domains"),
        difficulties=scenario_cfg.get("difficulties"),
    )
    n_per_combo = scenario_cfg.get("n_per_combo", 200)
    all_instances = []
    for theta in theta_grid:
        all_instances.extend(
            generator.generate_batch(n=n_per_combo, theta_sampler=lambda t=theta: t)
        )
    full_dataset = CausalReasoningDataset.from_scenario_instances(all_instances)
    train_dataset, _ = full_dataset.stratified_split(
        train_frac=config.get("data", {}).get("train_frac", 0.9)
    )
    return train_dataset, len(theta_grid), len(all_instances)
