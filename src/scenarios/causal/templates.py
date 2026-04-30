"""
Domain-specific sentence templates for causal chain scenario generation.

Each domain provides entity pools, causal verb pools, and sentence templates.
All templates use ``{slot}`` placeholders filled at generation time by
:class:`~src.scenarios.causal.generator.CausalScenarioGenerator`.

Templates are intentionally kept simple so the generated text is grammatically
correct without an LLM — the reasoning difficulty comes from the *structure*
(chain length, confounders, intervention type), not from vocabulary complexity.
"""

from __future__ import annotations

from typing import Dict, List

# ── Entity pools ─────────────────────────────────────────────────────────────────
# Each domain maintains 15+ distinct entities to support chain_length up to 8
# with enough leftover entities for confounders.

DOMAIN_ENTITIES: Dict[str, List[str]] = {
    "physical": [
        "heavy rainfall",
        "flooding",
        "soil erosion",
        "landslide",
        "river overflow",
        "drought",
        "heatwave",
        "wildfire",
        "air pollution",
        "glacial melt",
        "temperature rise",
        "ice formation",
        "electrical storm",
        "strong wind",
        "atmospheric pressure drop",
        "dust storm",
        "ocean current shift",
        "ground saturation",
    ],
    "medical": [
        "bacterial infection",
        "fever",
        "systemic inflammation",
        "organ stress",
        "immune response activation",
        "dehydration",
        "elevated cortisol levels",
        "cell membrane damage",
        "tissue necrosis",
        "chronic pain",
        "antibiotic resistance",
        "nutritional deficiency",
        "cardiovascular strain",
        "respiratory distress",
        "metabolic imbalance",
        "blood pressure spike",
        "platelet aggregation",
        "neural signal disruption",
    ],
    "social": [
        "unemployment",
        "income inequality",
        "housing insecurity",
        "poor nutrition",
        "limited healthcare access",
        "educational underinvestment",
        "social isolation",
        "community mistrust",
        "political instability",
        "rise in crime rates",
        "reduced civic engagement",
        "weakened social safety net",
        "emigration",
        "increased mental health burden",
        "reduced workforce productivity",
        "family breakdown",
        "school dropout rates",
        "voter apathy",
    ],
    "mechanical": [
        "lubrication failure",
        "bearing overheating",
        "metal fatigue",
        "hydraulic pressure drop",
        "seal degradation",
        "vibration imbalance",
        "corrosion buildup",
        "electrical short circuit",
        "coolant leak",
        "thermal expansion",
        "gear misalignment",
        "motor overload",
        "structural stress fracture",
        "fuel contamination",
        "pump cavitation",
        "valve blockage",
        "drive belt slippage",
        "sensor malfunction",
    ],
}

# ── Causal verb pools by difficulty ──────────────────────────────────────────────
# Easy:   simple, unambiguous causal verbs
# Medium: more nuanced causal language
# Hard:   technical/academic causal expressions

CAUSAL_VERBS: Dict[str, List[str]] = {
    "easy": [
        "causes",
        "leads to",
        "results in",
        "produces",
    ],
    "medium": [
        "triggers",
        "contributes to",
        "drives",
        "brings about",
        "gives rise to",
    ],
    "hard": [
        "mediates the onset of",
        "precipitates",
        "catalyzes",
        "exacerbates",
        "is a proximate cause of",
        "is a necessary precondition for",
    ],
}

# ── Chain link sentence templates ────────────────────────────────────────────────
# {cause} → causing entity
# {verb}  → causal verb from CAUSAL_VERBS
# {effect}→ effect entity

CHAIN_LINK_TEMPLATES: List[str] = [
    "{cause} {verb} {effect}.",
    "When {cause} occurs, it {verb} {effect}.",
    "The presence of {cause} {verb} {effect}.",
    "As a direct consequence, {cause} {verb} {effect}.",
    "Evidence shows that {cause} typically {verb} {effect}.",
]

# ── Confounder introduction templates ────────────────────────────────────────────
# {confounder} → the confounding variable
# {cause}      → chain's root cause
# {effect}     → chain's final effect

CONFOUNDER_TEMPLATES: List[str] = [
    "Note that {confounder} independently affects both {cause} and {effect}.",
    "A shared upstream factor, {confounder}, influences both {cause} and {effect}.",
    "However, {confounder} is a common driver of both {cause} and {effect}.",
]

# ── Counterfactual question templates ────────────────────────────────────────────
# {removed_cause} → the root cause being hypothetically removed
# {final_effect}  → the terminal effect node
# {confounder}    → shared cause (confounded variant only)

CF_QUESTION_TEMPLATES: Dict[str, List[str]] = {
    "direct": [
        "If {removed_cause} had not occurred, would {final_effect} have happened?",
        "Suppose {removed_cause} was absent. What would be the effect on {final_effect}?",
        "In a world where {removed_cause} did not take place, would {final_effect} still occur?",
    ],
    "confounded": [
        (
            "If {removed_cause} had not occurred but {confounder} remained, "
            "would {final_effect} still arise?"
        ),
        (
            "Assuming {removed_cause} was eliminated while {confounder} persisted, "
            "what would happen to {final_effect}?"
        ),
    ],
    "counterfactual": [
        (
            "Suppose we intervened to prevent {removed_cause} from occurring "
            "(do({removed_cause} = absent)). Would {final_effect} still happen?"
        ),
        (
            "Under a do-calculus intervention that removes {removed_cause}, "
            "what is the expected outcome for {final_effect}?"
        ),
    ],
}

# ── Direct causal question templates ─────────────────────────────────────────────
# {chain_description} → numbered list of causal chain sentences

DIRECT_QUESTION_TEMPLATES: List[str] = [
    (
        "Given the following chain of events, what is the final outcome?\n\n"
        "{chain_description}"
    ),
    (
        "Analyze the causal sequence below and identify what ultimately results:\n\n"
        "{chain_description}"
    ),
    (
        "Based on the causal chain described, trace the reasoning to the end effect:\n\n"
        "{chain_description}"
    ),
]

# ── CoT trace templates ───────────────────────────────────────────────────────────

COT_STEP_TEMPLATE = "Step {step_num}: {cause} {verb} {effect}."

COT_CONCLUSION_TEMPLATE = "Therefore, the final outcome is: {final_effect}."

COT_CF_STEP_TEMPLATE = (
    "Step {step_num}: Without {removed_cause}, "
    "the causal chain is broken at this first link."
)

COT_CF_CONCLUSION_DIRECT = (
    "Therefore, since the causal chain is broken at Step {break_step}, "
    "{final_effect} would NOT occur in the counterfactual world."
)

COT_CF_CONCLUSION_CONFOUNDED = (
    "Therefore, even without {removed_cause}, {confounder} would still independently "
    "drive {final_effect}. The outcome would still occur despite the intervention."
)

# ── System prompt ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a careful causal reasoning assistant. "
    "When answering questions about causal chains, reason step-by-step through "
    "each link in the chain before stating your final conclusion."
)
