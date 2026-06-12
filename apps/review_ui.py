"""Streamlit expert review UI (SR-23)."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.feedback.expert_schema import ExpertAction
from src.feedback.store import FeedbackStore
from src.serving.providers import MockScenarioProvider
from src.serving.schemas import ScenarioRequest

st.set_page_config(page_title="Scenario Expert Review", layout="wide")
st.title("Scenario Expert Review")

store_path = Path("data/feedback/expert_feedback.jsonl")
store = FeedbackStore(store_path)
provider = MockScenarioProvider()

artifact_path = st.sidebar.text_input("Artifact JSON path (optional)", "")
if artifact_path and Path(artifact_path).is_file():
    from src.serving.schemas import ScenarioArtifact
    artifact = ScenarioArtifact.model_validate_json(Path(artifact_path).read_text(encoding="utf-8"))
else:
    artifact = provider.generate(ScenarioRequest(theta={"chain_length": 3, "domain": "physical"}))

st.subheader(f"Scenario {artifact.id}")
st.json(artifact.theta)
st.metric("Coherence", f"{artifact.coherence:.2f}")
st.metric("Feasibility", f"{artifact.feasibility:.2f}")
st.write("Verification score:", artifact.verification_summary.overall_score)

for path in artifact.paths:
    with st.expander(path.path_id):
        st.write(path.text)
        st.write(f"Feasibility: {path.feasibility:.2f} | Tail: {path.tail_tag.value}")

col1, col2, col3, col4, col5 = st.columns(5)
note = st.text_area("Note", "")

if col1.button("Approve"):
    store.create(artifact.id, ExpertAction.APPROVE, note=note, theta=artifact.theta, status="approved")
    st.success("Approved")
if col2.button("Reject"):
    store.create(artifact.id, ExpertAction.REJECT, note=note, theta=artifact.theta, status="rejected")
    st.warning("Rejected")
if col3.button("Flag Numerical"):
    store.create(artifact.id, ExpertAction.FLAG_NUMERICAL, note=note, status="flagged")
    st.info("Flagged numerical")
if col4.button("Flag Logic"):
    store.create(artifact.id, ExpertAction.FLAG_LOGIC, note=note, status="flagged")
    st.info("Flagged logic")
if col5.button("Add Note"):
    store.create(artifact.id, ExpertAction.ADD_NOTE, note=note, status="noted")
    st.info("Note saved")

st.subheader("Recent feedback")
pending = store.list_by_status("pending")
approved = store.list_by_status("approved")
st.write(f"Pending: {len(pending)} | Approved: {len(approved)}")
for fb in store.list_recent(5):
    st.code(json.dumps(fb.to_dict(), indent=2))
