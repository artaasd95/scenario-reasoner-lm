#!/usr/bin/env python
"""
Export enterprise demo artifacts as JSON and Markdown.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.demo.pipeline import run_enterprise_demo


def scenario_to_markdown(card: dict, index: int) -> str:
    lines = [
        f"## {index}. {card['title']}",
        "",
        f"**Severity:** {card['severity']} | **Likelihood:** {card['likelihood']} | "
        f"**Horizon:** {card['horizon']} | **Confidence:** {card['confidence']:.0%}",
        "",
        "### Causal chain",
    ]
    for step in card["causal_chain"]:
        lines.append(f"- {step}")
    lines.extend(["", "### Missed-risk rationale", card["missed_risk_rationale"], ""])
    if card.get("warning_signals"):
        lines.append("### Warning signals")
        for w in card["warning_signals"]:
            lines.append(f"- {w}")
        lines.append("")
    if card.get("mitigations"):
        lines.append("### Mitigations")
        for m in card["mitigations"]:
            lines.append(f"- {m}")
        lines.append("")
    if card.get("source_evidence"):
        lines.append("### Source evidence")
        for ev in card["source_evidence"]:
            lines.append(f"- **{ev['section_name']}** (`{ev['chunk_id']}`): {ev['quote_text'][:200]}…")
        lines.append("")
    lines.append(f"*Trace id: `{card.get('trace_id', '')}`*")
    lines.append("")
    return "\n".join(lines)


def export_artifacts(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "scenarios.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    md_lines = [
        "# Enterprise Risk Scenarios",
        "",
        f"**Filing:** {result['filing_id']}",
        f"**Trace:** {result.get('trace_id', 'n/a')}",
        "",
    ]
    for i, card in enumerate(result["scenarios"], start=1):
        md_lines.append(scenario_to_markdown(card, i))

    md_path = output_dir / "scenarios.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {json_path} and {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filing", default="acme_corp_10k")
    parser.add_argument("--output", default="artifacts/enterprise_demo")
    parser.add_argument("--offline", action="store_true", default=True)
    args = parser.parse_args()

    result = run_enterprise_demo(filing_id=args.filing, offline=args.offline, output_dir=None)
    export_artifacts(result, Path(args.output))


if __name__ == "__main__":
    main()
