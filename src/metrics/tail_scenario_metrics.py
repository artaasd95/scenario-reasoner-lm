"""Tail / non-possible scenario metrics and generator tags (S10-02)."""

from __future__ import annotations

from typing import Any, Dict


TAIL_TAGS = ("none", "tail", "non_possible", "novel")


def tag_tail_scenario(row: Dict[str, Any]) -> Dict[str, Any]:
    """Assign tail_tag from row metadata or heuristic flags."""
    explicit = row.get("tail_tag")
    if explicit in TAIL_TAGS:
        return {"tail_tag": explicit, "is_tail": explicit == "tail", "is_non_possible": explicit == "non_possible"}

    if row.get("non_possible") is True:
        return {"tail_tag": "non_possible", "is_tail": False, "is_non_possible": True}

    stress = str(row.get("theta", {}).get("stress_regime", ""))
    if stress == "severe":
        return {"tail_tag": "tail", "is_tail": True, "is_non_possible": False}

    difficulty = str(row.get("theta", {}).get("difficulty", ""))
    if difficulty == "hard":
        return {"tail_tag": "tail", "is_tail": True, "is_non_possible": False}

    return {"tail_tag": "none", "is_tail": False, "is_non_possible": False}


def tail_scenario_metrics(rows: list[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate tail / novelty rates over path rows."""
    if not rows:
        return {"tail_rate": 0.0, "non_possible_rate": 0.0, "novelty_rate": 0.0}
    tags = [tag_tail_scenario(r)["tail_tag"] for r in rows]
    n = len(tags)
    return {
        "tail_rate": round(sum(1 for t in tags if t == "tail") / n, 4),
        "non_possible_rate": round(sum(1 for t in tags if t == "non_possible") / n, 4),
        "novelty_rate": round(sum(1 for t in tags if t not in ("none",)) / n, 4),
    }
