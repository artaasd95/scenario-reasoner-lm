"""
Unit tests for causal scenario taxonomy, generator, and reward functions.

These tests cover pure-Python modules only — no torch required.
"""

from __future__ import annotations

import pytest

from src.scenarios.causal.taxonomy import CausalTheta, CausalThetaSampler
from src.scenarios.causal.generator import CausalScenarioGenerator
from src.scenarios.base_scenario import ScenarioInstance
from src.training.causal_reward import CausalRewardFunction
from src.training.reward_composer import RewardComposer
from src.metrics.causal_metrics import (
    CausalChainAccuracy,
    CounterfactualValidityScore,
    TrajectoryConsistency,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def direct_theta():
    return CausalTheta(
        chain_length=3,
        intervention_type="direct",
        domain="physical",
        difficulty="easy",
        entity_count=2,
    )


@pytest.fixture
def cf_theta():
    return CausalTheta(
        chain_length=4,
        intervention_type="counterfactual",
        domain="social",
        difficulty="medium",
        entity_count=3,
    )


@pytest.fixture
def conf_theta():
    return CausalTheta(
        chain_length=3,
        intervention_type="confounded",
        num_confounders=1,
        domain="medical",
        difficulty="easy",
        entity_count=2,
    )


@pytest.fixture
def generator():
    return CausalScenarioGenerator(seed=42)


@pytest.fixture
def reward_fn():
    return CausalRewardFunction()


# ── CausalTheta tests ─────────────────────────────────────────────────────────

class TestCausalTheta:
    def test_valid_construction(self, direct_theta):
        assert direct_theta.chain_length == 3
        assert direct_theta.domain == "physical"

    def test_invalid_chain_length(self):
        with pytest.raises(ValueError, match="chain_length"):
            CausalTheta(chain_length=1)

    def test_invalid_chain_length_too_large(self):
        with pytest.raises(ValueError, match="chain_length"):
            CausalTheta(chain_length=9)

    def test_invalid_intervention_type(self):
        with pytest.raises(ValueError, match="intervention_type"):
            CausalTheta(intervention_type="unknown")

    def test_invalid_domain(self):
        with pytest.raises(ValueError, match="domain"):
            CausalTheta(domain="quantum")

    def test_invalid_difficulty(self):
        with pytest.raises(ValueError, match="difficulty"):
            CausalTheta(difficulty="extreme")

    def test_invalid_confounders(self):
        with pytest.raises(ValueError, match="num_confounders"):
            CausalTheta(num_confounders=5)

    def test_to_dict_roundtrip(self, direct_theta):
        d = direct_theta.to_dict()
        assert d["chain_length"] == 3
        assert d["domain"] == "physical"
        assert d["intervention_type"] == "direct"


class TestCausalThetaSampler:
    def test_sample_returns_valid_theta(self):
        sampler = CausalThetaSampler(seed=0)
        theta = sampler.sample()
        assert isinstance(theta, CausalTheta)

    def test_sample_reproducible_with_seed(self):
        s1 = CausalThetaSampler(seed=99)
        s2 = CausalThetaSampler(seed=99)
        assert s1.sample().to_dict() == s2.sample().to_dict()

    def test_confounders_zero_for_direct(self):
        sampler = CausalThetaSampler(seed=7, intervention_types=["direct"])
        for _ in range(10):
            theta = sampler.sample()
            assert theta.num_confounders == 0

    def test_grid_correct_size(self):
        sampler = CausalThetaSampler()
        grid = sampler.grid(
            chain_lengths=[3, 5],
            intervention_types=["direct"],
            domains=["physical", "social"],
            difficulties=["easy", "medium"],
        )
        # 2 chains × 1 type × 2 domains × 2 difficulties = 8
        assert len(grid) == 8

    def test_grid_all_unique(self):
        sampler = CausalThetaSampler()
        grid = sampler.grid(
            chain_lengths=[3, 5],
            domains=["physical", "social"],
            difficulties=["easy"],
        )
        dicts = [t.to_dict() for t in grid]
        assert len(dicts) == len(set(frozenset(d.items()) for d in dicts))


# ── CausalScenarioGenerator tests ────────────────────────────────────────────

class TestCausalScenarioGenerator:
    def test_instantiate_returns_scenario_instance(self, generator, direct_theta):
        inst = generator.instantiate(direct_theta)
        assert isinstance(inst, ScenarioInstance)

    def test_instance_has_required_fields(self, generator, direct_theta):
        inst = generator.instantiate(direct_theta)
        assert inst.prompt
        assert inst.reasoning_trace
        assert inst.answer
        assert inst.theta is direct_theta
        assert inst.scenario_id

    def test_prompt_contains_system_prompt(self, generator, direct_theta):
        inst = generator.instantiate(direct_theta)
        assert "causal reasoning" in inst.prompt.lower()

    def test_trace_has_steps(self, generator, direct_theta):
        inst = generator.instantiate(direct_theta)
        assert "Step 1:" in inst.reasoning_trace

    def test_trace_has_conclusion(self, generator, direct_theta):
        inst = generator.instantiate(direct_theta)
        trace_lower = inst.reasoning_trace.lower()
        assert any(kw in trace_lower for kw in ("therefore", "thus", "final outcome"))

    def test_direct_chain_answer_format(self, generator, direct_theta):
        inst = generator.instantiate(direct_theta)
        assert "final outcome" in inst.answer.lower()

    def test_counterfactual_answer_contains_no(self, generator, cf_theta):
        inst = generator.instantiate(cf_theta)
        assert "no" in inst.answer.lower() or "not" in inst.answer.lower()

    def test_confounded_answer_contains_yes(self, generator, conf_theta):
        inst = generator.instantiate(conf_theta)
        # confounded with confounder → outcome still occurs (yes)
        assert "yes" in inst.answer.lower() or "still" in inst.answer.lower()

    def test_chain_length_respected(self, generator):
        for length in [2, 4, 6]:
            theta = CausalTheta(chain_length=length, domain="physical",
                                difficulty="easy", intervention_type="direct")
            inst = generator.instantiate(theta)
            # Trace should mention at least (length) step markers
            step_count = inst.reasoning_trace.lower().count("step ")
            assert step_count >= length, (
                f"Expected >= {length} steps, got {step_count} for chain_length={length}"
            )

    def test_generate_batch(self, generator):
        batch = generator.generate_batch(n=10)
        assert len(batch) == 10
        assert all(isinstance(i, ScenarioInstance) for i in batch)

    def test_generate_batch_with_sampler(self, generator, direct_theta):
        batch = generator.generate_batch(n=5, theta_sampler=lambda: direct_theta)
        assert len(batch) == 5
        assert all(b.theta is direct_theta for b in batch)

    def test_to_dict_is_json_serializable(self, generator, direct_theta):
        import json
        inst = generator.instantiate(direct_theta)
        d = inst.to_dict()
        json.dumps(d)  # should not raise

    def test_evaluate_trajectory(self, generator, direct_theta):
        trajectory = [
            "Step 1: heavy rainfall causes flooding.",
            "Step 2: flooding causes soil erosion.",
            "Therefore, the final outcome is: soil erosion.",
        ]
        score = generator.evaluate_trajectory(trajectory, direct_theta)
        assert 0.0 <= score <= 1.0

    def test_is_valid_transition_domain_entity(self, generator, direct_theta):
        assert generator.is_valid_transition(
            "heavy rainfall", "causes", "flooding", direct_theta
        )

    def test_is_valid_transition_rejects_cycle(self, generator, direct_theta):
        assert not generator.is_valid_transition(
            "flooding", "causes", "flooding", direct_theta
        )

    def test_is_valid_transition_rejects_unknown_entity(self, generator, direct_theta):
        assert not generator.is_valid_transition(
            "heavy rainfall", "causes", "unicorn", direct_theta
        )


# ── CausalRewardFunction tests ────────────────────────────────────────────────

class TestCausalRewardFunction:
    GOOD_TRACE = (
        "Step 1: heavy rainfall causes flooding.\n"
        "Step 2: flooding causes soil erosion.\n"
        "Step 3: soil erosion causes landslide.\n"
        "Therefore, the final outcome is: landslide."
    )
    GOOD_ANSWER = "The final outcome is: landslide."
    BAD_TRACE = "I'm not sure. Something might happen."
    BAD_ANSWER = "unknown"

    def test_good_trace_scores_higher_than_bad(self, reward_fn, direct_theta):
        good = reward_fn.score("p", self.GOOD_TRACE, self.GOOD_ANSWER,
                                expected_answer=self.GOOD_ANSWER, theta=direct_theta)
        bad = reward_fn.score("p", self.BAD_TRACE, self.BAD_ANSWER,
                               expected_answer=self.GOOD_ANSWER, theta=direct_theta)
        assert good > bad

    def test_score_in_range(self, reward_fn, direct_theta):
        score = reward_fn.score("p", self.GOOD_TRACE, self.GOOD_ANSWER,
                                 expected_answer=self.GOOD_ANSWER, theta=direct_theta)
        assert 0.0 <= score <= 1.0

    def test_exact_match_answer_bonus(self, reward_fn):
        score_exact = reward_fn.score("p", self.GOOD_TRACE, self.GOOD_ANSWER,
                                       expected_answer=self.GOOD_ANSWER)
        score_wrong = reward_fn.score("p", self.GOOD_TRACE, "wrong answer",
                                       expected_answer=self.GOOD_ANSWER)
        assert score_exact > score_wrong

    def test_no_expected_answer_neutral(self, reward_fn):
        score = reward_fn.score("p", self.GOOD_TRACE, self.GOOD_ANSWER)
        assert 0.0 <= score <= 1.0

    def test_circular_causation_penalized(self, reward_fn, direct_theta):
        circular_trace = (
            "Step 1: flooding causes flooding.\n"
            "Step 2: flooding leads to flooding.\n"
            "Therefore, flooding."
        )
        normal_trace = self.GOOD_TRACE
        score_circular = reward_fn.score("p", circular_trace, self.GOOD_ANSWER,
                                          expected_answer=self.GOOD_ANSWER, theta=direct_theta)
        score_normal = reward_fn.score("p", normal_trace, self.GOOD_ANSWER,
                                        expected_answer=self.GOOD_ANSWER, theta=direct_theta)
        assert score_circular < score_normal

    def test_cf_scenario_requires_intervention_language(self, reward_fn, cf_theta):
        cf_trace_good = (
            "Step 1: Without unemployment, the chain is broken.\n"
            "Therefore, housing insecurity would NOT occur. No."
        )
        cf_trace_bad = (
            "Step 1: unemployment leads to housing insecurity.\n"
            "Therefore, housing insecurity."
        )
        score_good = reward_fn.score("p", cf_trace_good, "No.",
                                      theta=cf_theta)
        score_bad = reward_fn.score("p", cf_trace_bad, "Yes.",
                                     theta=cf_theta)
        assert score_good > score_bad


# ── RewardComposer tests ──────────────────────────────────────────────────────

class TestRewardComposer:
    def test_score_returns_all_keys(self, reward_fn, direct_theta):
        composer = RewardComposer(reward_fn)
        result = composer.score(
            "prompt",
            "Step 1: X causes Y.\nTherefore, Y.",
            "The final outcome is: Y.",
            expected_answer="The final outcome is: Y.",
            theta=direct_theta,
        )
        assert set(result.keys()) == {"R_task", "R_cot", "R_tot", "R_aha", "R_total"}

    def test_r_total_in_range(self, reward_fn, direct_theta):
        composer = RewardComposer(reward_fn)
        result = composer.score("p", "Step 1: A causes B.\nTherefore B.", "B.", theta=direct_theta)
        assert 0.0 <= result["R_total"] <= 1.0

    def test_cot_reward_positive_for_step_trace(self, reward_fn, direct_theta):
        composer = RewardComposer(reward_fn, alpha=0.5)
        result = composer.score(
            "p",
            "Step 1: A causes B.\nStep 2: B causes C.\nTherefore C.",
            "C.",
            theta=direct_theta,
        )
        assert result["R_cot"] > 0.0

    def test_score_batch_length_matches(self, reward_fn):
        composer = RewardComposer(reward_fn)
        results = composer.score_batch(
            prompts=["p1", "p2", "p3"],
            traces=["Step 1: A.", "B.", "Step 1: X.\nTherefore Y."],
            answers=["A.", "B.", "Y."],
        )
        assert len(results) == 3

    def test_custom_weights_are_normalized(self, reward_fn):
        composer = RewardComposer(reward_fn, alpha=1.0, beta=0.0, gamma=0.0)
        assert 0.0 < composer.alpha <= 1.0


# ── Causal metrics tests ──────────────────────────────────────────────────────

class TestCausalChainAccuracy:
    def test_good_trace_scores_higher(self):
        m = CausalChainAccuracy()
        good = "Step 1: A causes B.\nStep 2: B causes C.\nTherefore, C."
        bad = "Something happened."
        m.update([good], ["C."])
        score_good = m.compute()
        m.reset()
        m.update([bad], ["C."])
        score_bad = m.compute()
        assert score_good > score_bad

    def test_score_in_range(self):
        m = CausalChainAccuracy()
        m.update(["Step 1: A causes B.\nTherefore, B."], ["B."])
        assert 0.0 <= m.compute() <= 1.0

    def test_empty_returns_zero(self):
        m = CausalChainAccuracy()
        assert m.compute() == 0.0

    def test_reset_clears_state(self):
        m = CausalChainAccuracy()
        m.update(["Step 1: A causes B.\nTherefore, B."], ["B."])
        m.reset()
        assert m.compute() == 0.0

    def test_circular_penalty(self):
        m = CausalChainAccuracy()
        circular = "Step 1: flooding causes flooding.\nTherefore, flooding."
        normal = "Step 1: rain causes flooding.\nTherefore, flooding."
        m.update([circular], ["flooding."])
        c_score = m.compute()
        m.reset()
        m.update([normal], ["flooding."])
        n_score = m.compute()
        assert c_score < n_score


class TestCounterfactualValidityScore:
    def test_direct_always_one(self):
        m = CounterfactualValidityScore()
        m.update(["anything"], ["ref"], intervention_types=["direct"])
        assert m.compute() == pytest.approx(1.0)

    def test_good_cf_trace_scores_high(self):
        m = CounterfactualValidityScore()
        good = "Without X occurring, Y would not have happened. No."
        m.update([good], ["No."], intervention_types=["counterfactual"])
        assert m.compute() > 0.7

    def test_bad_cf_trace_scores_low(self):
        m = CounterfactualValidityScore()
        bad = "X leads to Y and Z follows."
        m.update([bad], ["No."], intervention_types=["counterfactual"])
        assert m.compute() < 0.7

    def test_reset_works(self):
        m = CounterfactualValidityScore()
        m.update(["Without X, no Y. No."], ["No."], intervention_types=["counterfactual"])
        m.reset()
        assert m.compute() == 0.0


class TestTrajectoryConsistency:
    def test_consistent_trace_scores_one(self):
        m = TrajectoryConsistency()
        m.update(["Step 1: A causes B.\nTherefore, B."], ["B."])
        assert m.compute() == pytest.approx(1.0)

    def test_polarity_flip_penalized(self):
        m = TrajectoryConsistency()
        flip = "Step 1: Without X, Y would not occur. Yes."
        m.update([flip], ["No."])
        assert m.compute() < 1.0

    def test_empty_returns_zero(self):
        m = TrajectoryConsistency()
        assert m.compute() == 0.0

    def test_batch_averaging(self):
        m = TrajectoryConsistency()
        traces = [
            "Step 1: A causes B.\nTherefore, B.",
            "Step 1: X causes X.\nTherefore, X.",
        ]
        refs = ["B.", "X."]
        m.update(traces, refs)
        score = m.compute()
        assert 0.0 <= score <= 1.0
