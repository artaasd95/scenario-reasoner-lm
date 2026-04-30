"""
Training components for Scenario Reasoner LM.

Provides:
    * :class:`~src.training.causal_reward.CausalRewardFunction`
      — rule-based task reward for causal reasoning
    * :class:`~src.training.reward_composer.RewardComposer`
      — composite reward combining task reward + monitoring signals
    * :class:`~src.training.preference_builder.PreferenceBuilder`
      — DPO preference pair construction
    * :class:`~src.training.rlhf_trainer.RLHFTrainer`
      — end-to-end DPO training loop
"""

from src.training.causal_reward import CausalRewardFunction
from src.training.preference_builder import PreferenceBuilder
from src.training.reward_composer import RewardComposer
from src.training.rlhf_trainer import RLHFTrainer

__all__ = [
    "CausalRewardFunction",
    "RewardComposer",
    "PreferenceBuilder",
    "RLHFTrainer",
]
