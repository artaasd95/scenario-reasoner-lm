# Scenario Measurability Contract (S10-01)

Each evaluated scenario path must expose:

| Field | Type | Description |
| --- | --- | --- |
| `feasibility` | float ∈ [0,1] | Path feasibility score |
| `tail_tag` | enum | `none`, `tail`, `non_possible`, `novel` |
| `theta_stratum` | string | θ family / stratum label |
| `primary_quality_score` | float | Main quality metric for the path |

## Artifacts

- **Eval/train:** `scenario_measurement.json` — list under `paths`
- **Robustness:** `robustness_report.json` — aggregate + per-type slices

Implementation: [src/evaluation/scenario_measurement.py](../src/evaluation/scenario_measurement.py)

Pipeline `measure` stage populates these via [src/data/pipeline/runner.py](../src/data/pipeline/runner.py).

## Non-goals

- Live provider scoring
- Financial advice rubrics beyond research disclaimers
