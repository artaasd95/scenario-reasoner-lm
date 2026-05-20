# Enterprise Risk Demo Contract

**Vault seed:** S2-01 · **Milestone:** S2 — Enterprise risk showcase contract  
**Status:** ongoing · **Target due:** 2026-05-15

## Demo goal

From **one bundled 10-K filing**, produce **five catastrophic but plausible enterprise risk scenarios**. Each scenario is source-grounded, auditable, and exportable as JSON or Markdown.

The demo path is intentionally narrow:

1. Load bundled sample filing (or optional user upload).
2. Extract SEC sections and evidence chunks.
3. Run DSPy pipeline: evidence → hypotheses → scenarios → critique → ranking.
4. Render five scenario cards with traces and export artifacts.

Bundled entrypoint: `data/samples/tenk/acme_corp_10k.txt`  
CLI: `python scripts/run_enterprise_demo.py --offline`  
UI: `streamlit run src/ui/streamlit_app.py`

## Non-goals

| Non-goal | Rationale |
| --- | --- |
| **Financial advice** | Scenarios illustrate risk reasoning for audit and research demos, not investment or trading recommendations. |
| **Broad GRC platform** | No policy workflow engine, control libraries, or enterprise GRC integrations in S2. |
| **Training-first detour** | Causal RLHF training remains a separate substrate; S2 does not require DPO runs to demo the 10-K flow. |

## Acceptance criteria (S2-01)

- [x] Demo goal locked to five scenarios from one 10-K.
- [x] README and this spec state non-goals above.
- [x] Decision log records calibrated enterprise-risk positioning.

## Decision log

| Date | Decision | Rationale |
| --- | --- | --- |
| 2026-05-17 | **Showcase = reasoning traceability, not GRC product** | Buyers care whether catastrophic risks are grounded in filing text with inspectable chains—not whether we replace ServiceNow or Archer. |
| 2026-05-17 | **One bundled 10-K before live SEC** | Offline demo must work without EDGAR credentials or rate limits; live ingestion is a follow-on. |
| 2026-05-17 | **Five scenarios, not open-ended generation** | Fixed cardinality makes ranking, critique, and UI comparison tractable for executive demos. |
| 2026-05-17 | **Extend Θ, do not replace causal abstractions** | `EnterpriseRiskTheta` parallels `CausalTheta`; causal RLHF code paths stay intact. |
| 2026-05-17 | **BootstrapFewShot before MIPRO** | Optimize only after a tiny eval set measures grounding and plausibility; MIPRO deferred. |
| 2026-05-17 | **Langfuse parent trace `tenk_demo_run`** | White-box spans for each pipeline stage; no-op client when keys absent. |
| 2026-05-20 | **S5 simulation vs measurement paths** | See `docs/scenario-simulation-paths.md`; live runs gated; headline 10-K benchmark unchanged. |

## Related artifacts

| Area | Path |
| --- | --- |
| Scenario schema | `src/risk/schema.py` |
| Enterprise Θ | `src/risk/enterprise_theta.py` |
| 10-K ingestion | `src/ingestion/` |
| DSPy modules | `src/dspy_modules/` |
| Tracing | `src/tracing/`, `docs/langfuse-tracing.md` |
| UI | `src/ui/streamlit_app.py` |
| Export | `scripts/export_demo_artifacts.py` |

## Schema reference

Scenario cards use `EnterpriseRiskScenarioCard` (see `src/risk/schema.py`):

- `title`, `source_evidence`, `causal_chain`, `missed_risk_rationale`
- `severity`, `likelihood`, `horizon`, `confidence`
- `warning_signals`, `mitigations`, `trace_id`

Evidence chunks use `EvidenceChunk`: `section_name`, `chunk_id`, `source_span`, `quote_text`.
