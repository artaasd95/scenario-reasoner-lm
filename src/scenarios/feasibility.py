"""Path feasibility assessment for pipeline measure stage (S10-02)."""

from __future__ import annotations

from typing import Any, Dict


def assess_path_feasibility(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lightweight feasibility hook for pipeline measure stage.

    Returns feasible_score in [0, 1] and boolean feasible flag.
    """
    theta = row.get("theta", {})
    if row.get("infeasible") is True:
        return {"feasible": False, "feasible_score": 0.0, "reason": "marked_infeasible"}

    score = 0.8
    if not theta:
        score = 0.4
    if row.get("scenario_type") == "game" and not row.get("search_graph"):
        score = min(score, 0.5)
    measurement = row.get("measurement", {})
    if measurement.get("feasibility") is not None:
        score = float(measurement["feasibility"])

    return {
        "feasible": score >= 0.2,
        "feasible_score": round(score, 4),
        "reason": "",
    }
