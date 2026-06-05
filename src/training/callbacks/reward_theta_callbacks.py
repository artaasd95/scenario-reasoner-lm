"""Training callbacks — reward decomposition, θ-stratified early stop (S10-08)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional


class RewardDecompositionCallback:
    """Accumulate R_task / R_cot / R_tot / R_aha across steps."""

    def __init__(self) -> None:
        self.history: List[Dict[str, float]] = []

    def on_step_end(self, breakdown: Dict[str, float]) -> None:
        self.history.append(dict(breakdown))

    def aggregate(self) -> Dict[str, float]:
        if not self.history:
            return {}
        keys = self.history[0].keys()
        return {k: round(sum(h[k] for h in self.history) / len(self.history), 4) for k in keys}


class ThetaStratifiedEarlyStopCallback:
    """Early stop when θ-stratum metric plateaus."""

    def __init__(self, patience: int = 3, min_delta: float = 0.01) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best: Dict[str, float] = {}
        self._stall: Dict[str, int] = defaultdict(int)

    def on_epoch_end(self, stratum_metrics: Dict[str, float]) -> bool:
        """Return True if training should stop."""
        stop = True
        for stratum, value in stratum_metrics.items():
            best = self._best.get(stratum, float("-inf"))
            if value > best + self.min_delta:
                self._best[stratum] = value
                self._stall[stratum] = 0
                stop = False
            else:
                self._stall[stratum] += 1
        if not stratum_metrics:
            return False
        return stop and all(self._stall[s] >= self.patience for s in stratum_metrics)
