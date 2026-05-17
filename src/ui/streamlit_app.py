"""
Streamlit UI for the enterprise risk 10-K demo.

Supports bundled sample filing, optional upload, progress stages,
five scenario cards, evidence excerpts, score badges, trace links,
and JSON/Markdown export.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.demo.pipeline import run_enterprise_demo
from src.ingestion.tenk_loader import list_bundled_filings, load_tenk_filing

st.set_page_config(page_title="Enterprise Risk Scenario Generator", layout="wide")
st.title("Enterprise Risk Scenario Generator")
st.caption(
    "Five catastrophic-but-plausible scenarios from one 10-K. "
    "Not financial advice. Demo showcase only."
)

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Input")
    bundled = list_bundled_filings()
    filing_id = st.selectbox("Sample filing", bundled, index=0)
    uploaded = st.file_uploader("Optional 10-K upload (.txt)", type=["txt"])
    offline = st.toggle("Offline demo (no API keys)", value=True)
    run_btn = st.button("Run demo", type="primary")

with col_right:
    st.subheader("Non-goals")
    st.info(
        "This demo does not provide financial advice, replace a GRC platform, "
        "or require causal RLHF training to run."
    )

if run_btn:
    progress = st.progress(0, text="Starting…")
    stages = [
        "Loading filing",
        "Extracting sections",
        "Chunking evidence",
        "Extracting risks",
        "Generating hypotheses",
        "Building scenarios",
        "Critique & ranking",
        "Rendering",
    ]

    try:
        if uploaded is not None:
            upload_path = _REPO_ROOT / "artifacts" / "_uploaded_tenk.txt"
            upload_path.parent.mkdir(parents=True, exist_ok=True)
            upload_path.write_bytes(uploaded.read())
            progress.progress(1 / len(stages), text=stages[0])
            result = run_enterprise_demo(filing_id=str(upload_path), offline=offline)
        else:
            for i, label in enumerate(stages):
                progress.progress((i + 1) / len(stages), text=label)
            result = run_enterprise_demo(filing_id=filing_id, offline=offline)

        progress.progress(1.0, text="Complete")
        st.session_state["demo_result"] = result
    except Exception as exc:
        st.error(f"Demo failed: {exc}")

if "demo_result" in st.session_state:
    result = st.session_state["demo_result"]
    st.divider()
    meta_cols = st.columns(4)
    meta_cols[0].metric("Scenarios", len(result["scenarios"]))
    meta_cols[1].metric("Chunks", result.get("num_chunks", 0))
    meta_cols[2].write(f"**Trace:** `{result.get('trace_id', 'n/a')}`")
    if result.get("langfuse_url"):
        meta_cols[3].markdown(f"[Langfuse trace]({result['langfuse_url']})")

    for i, card in enumerate(result["scenarios"], start=1):
        with st.expander(f"{i}. {card['title']}", expanded=i <= 2):
            badge_cols = st.columns(4)
            badge_cols[0].markdown(f"**Severity:** `{card['severity']}`")
            badge_cols[1].markdown(f"**Likelihood:** `{card['likelihood']}`")
            badge_cols[2].markdown(f"**Horizon:** `{card['horizon']}`")
            badge_cols[3].markdown(f"**Confidence:** {card['confidence']:.0%}")

            st.markdown("**Causal chain**")
            for step in card["causal_chain"]:
                st.markdown(f"- {step}")

            st.markdown("**Missed-risk rationale**")
            st.write(card["missed_risk_rationale"])

            if card.get("source_evidence"):
                st.markdown("**Evidence**")
                for ev in card["source_evidence"]:
                    st.code(ev["quote_text"][:500], language=None)

    st.divider()
    st.subheader("Export")
    st.download_button(
        "Download JSON",
        data=json.dumps(result, indent=2),
        file_name="enterprise_risk_scenarios.json",
        mime="application/json",
    )
    from scripts.export_demo_artifacts import scenario_to_markdown

    md = "\n".join(
        scenario_to_markdown(c, i) for i, c in enumerate(result["scenarios"], start=1)
    )
    st.download_button(
        "Download Markdown",
        data=md,
        file_name="enterprise_risk_scenarios.md",
        mime="text/markdown",
    )
