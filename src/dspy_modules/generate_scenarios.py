"""
Hypothesis generation and scenario building (DSPy or offline stub).
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from src.dspy_modules.signatures import BuildScenario, GenerateHypotheses, _require_dspy
from src.risk.enterprise_theta import EnterpriseRiskTheta
from src.risk.schema import EnterpriseRiskScenarioCard, EvidenceChunk


# Bundled offline scenarios for demo without LLM credentials
_OFFLINE_SCENARIOS: List[Dict[str, Any]] = [
    {
        "title": "Single-foundry ASIC cutoff after Taiwan disruption",
        "section": "Risk Factors",
        "chain": [
            "Export controls tighten on advanced packaging in Taiwan",
            "ACME cannot receive custom ASIC wafers for 6+ months",
            "Controller production halts for automotive safety lines",
            "Major OEMs invoke penalty clauses and switch to competitors",
        ],
        "rationale": "10-K discloses sole foundry dependency but understates cascade to customer penalties.",
        "severity": "catastrophic",
        "likelihood": "medium",
        "horizon": "6-18 months",
        "confidence": 0.78,
        "warnings": ["Foundry lead-time notices", "OEM dual-source RFQs"],
        "mitigations": ["Second-source ASIC qualification program"],
    },
    {
        "title": "SEC revenue recognition investigation triggers restatement spiral",
        "section": "Legal Proceedings",
        "chain": [
            "SEC enforcement concludes misallocated multi-year bundle revenue",
            "Financial restatement reduces reported EBITDA below covenant",
            "Revolver covenant breach accelerates debt repayment demands",
            "Customers renegotiate contracts citing material adverse change",
        ],
        "rationale": "Legal section notes investigation but not covenant linkage in MD&A.",
        "severity": "catastrophic",
        "likelihood": "medium",
        "horizon": "0-6 months",
        "confidence": 0.72,
        "warnings": ["Increased audit fees", "Delayed 10-K filing"],
        "mitigations": ["Covenant waiver negotiations", "Independent forensic review"],
    },
    {
        "title": "Staging breach escalates to production telemetry exposure",
        "section": "Cybersecurity",
        "chain": [
            "Attacker pivots from staging to shared IAM misconfiguration",
            "Production customer telemetry exfiltrated for 3 weeks undetected",
            "EU OEMs suspend connectivity features pending investigation",
            "Regulatory fines and contractual indemnity claims accumulate",
        ],
        "rationale": "Cyber disclosure is preliminary; production impact not ruled out.",
        "severity": "catastrophic",
        "likelihood": "high",
        "horizon": "0-6 months",
        "confidence": 0.81,
        "warnings": ["Anomalous outbound traffic", "MFA bypass attempts"],
        "mitigations": ["Segment staging from production identity plane"],
    },
    {
        "title": "EU machinery regulation blocks 19% revenue base",
        "section": "Regulatory",
        "chain": [
            "Functional safety documentation misses March 2027 deadline",
            "EU shipments halted for non-compliant controller families",
            "European OEM programs defer volume ramps",
            "Fixed cost base absorbs revenue gap; liquidity tightens",
        ],
        "rationale": "Regulatory section cites deadline but not revenue-at-risk quantification.",
        "severity": "high",
        "likelihood": "high",
        "horizon": "18-36 months",
        "confidence": 0.85,
        "warnings": ["Notified body audit findings", "OEM compliance questionnaires"],
        "mitigations": ["Accelerate safety file submissions per product line"],
    },
    {
        "title": "Japanese sole-source passive shortage ends legacy platform support",
        "section": "Supply Chain",
        "chain": [
            "38-week lead times extend to allocation rationing",
            "Legacy platforms lack safety stock coverage",
            "Installed base customers face unplanned downtime",
            "Service revenue collapses; replacement parts litigation rises",
        ],
        "rationale": "Supply chain notes legacy EOL but not service-revenue exposure.",
        "severity": "high",
        "likelihood": "high",
        "horizon": "6-18 months",
        "confidence": 0.77,
        "warnings": ["Vendor force-majeure letters", "Allocation notices to planners"],
        "mitigations": ["Last-time-buy program for legacy SKUs"],
    },
]


class GenerateScenariosModule:
    """Generate hypotheses and build scenario cards."""

    def __init__(self, use_dspy: bool = True) -> None:
        self.use_dspy = use_dspy
        self._hypothesis_predictor = None
        self._scenario_predictor = None
        if use_dspy:
            _require_dspy()
            import dspy

            self._hypothesis_predictor = dspy.Predict(GenerateHypotheses)
            self._scenario_predictor = dspy.Predict(BuildScenario)

    def generate_hypotheses(
        self,
        evidence_payload: dict,
        company_name: str = "ACME Corporation",
        trace_callback: Optional[callable] = None,
    ) -> List[dict]:
        evidence_json = json.dumps(evidence_payload.get("evidence", []))
        if self._hypothesis_predictor is not None:
            result = self._hypothesis_predictor(
                evidence_json=evidence_json,
                company_context=company_name,
            )
            hypotheses = json.loads(result.hypotheses_json)
        else:
            hypotheses = [
                {
                    "hypothesis": s["title"],
                    "supporting_chunk_ids": [],
                    "rationale": s["rationale"],
                }
                for s in _OFFLINE_SCENARIOS
            ]

        if trace_callback:
            trace_callback(
                stage="hypotheses",
                inputs={"company": company_name},
                outputs={"count": len(hypotheses)},
            )
        return hypotheses

    def build_scenarios(
        self,
        hypotheses: List[dict],
        chunks: List[EvidenceChunk],
        theta: EnterpriseRiskTheta,
        trace_callback: Optional[callable] = None,
    ) -> List[EnterpriseRiskScenarioCard]:
        cards: List[EnterpriseRiskScenarioCard] = []
        chunk_by_section = {c.section_name: c for c in chunks}

        if self._scenario_predictor is not None:
            evidence_json = json.dumps([c.to_dict() for c in chunks])
            for hyp in hypotheses[: theta.num_scenarios]:
                result = self._scenario_predictor(
                    hypothesis_json=json.dumps(hyp),
                    evidence_json=evidence_json,
                )
                cards.append(EnterpriseRiskScenarioCard.from_dict(json.loads(result.scenario_json)))
        else:
            for spec in _OFFLINE_SCENARIOS[: theta.num_scenarios]:
                section = spec["section"]
                ev = chunk_by_section.get(section) or chunks[0]
                cards.append(
                    EnterpriseRiskScenarioCard(
                        title=spec["title"],
                        source_evidence=[ev],
                        causal_chain=spec["chain"],
                        missed_risk_rationale=spec["rationale"],
                        severity=spec["severity"],
                        likelihood=spec["likelihood"],
                        horizon=spec["horizon"],
                        confidence=spec["confidence"],
                        warning_signals=spec["warnings"],
                        mitigations=spec["mitigations"],
                        trace_id=str(uuid.uuid4()),
                    )
                )

        if trace_callback:
            for i, card in enumerate(cards):
                trace_callback(
                    stage=f"scenario_build_{i + 1}",
                    inputs={"title": card.title},
                    outputs={"trace_id": card.trace_id},
                )
        return cards
