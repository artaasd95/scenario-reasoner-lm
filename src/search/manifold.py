"""
Feasible-set constraints for action vectors (S6 manifold hook).

S6 treats a "manifold" as a constrained subset of R^d (box or simplex), not
learned Riemannian geometry. Vectors are projected before discrete action mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


class ManifoldKind(str, Enum):
    BOX = "box"  # coordinates in [0, 1]
    SIMPLEX = "simplex"  # non-negative, sum to 1


@dataclass
class ActionManifold:
    """
    Defines feasible action vectors and projection.

    Args:
        kind: ``box`` or ``simplex``.
        dim: Expected vector dimension.
    """

    kind: ManifoldKind
    dim: int

    def project(self, vector: List[float]) -> List[float]:
        """Project *vector* onto the feasible set (truncate/pad to ``dim``)."""
        v = list(vector[: self.dim])
        while len(v) < self.dim:
            v.append(0.0)

        if self.kind == ManifoldKind.BOX:
            return [min(1.0, max(0.0, x)) for x in v]

        # Simplex: ReLU then normalize; uniform if sum is zero.
        v = [max(0.0, x) for x in v]
        s = sum(v)
        if s <= 0.0:
            u = 1.0 / self.dim
            return [u] * self.dim
        return [x / s for x in v]

    def is_feasible(self, vector: List[float]) -> bool:
        """Check whether *vector* lies in the feasible set (within tolerance)."""
        p = self.project(vector)
        return all(abs(a - b) < 1e-6 for a, b in zip(p, vector[: self.dim]))
