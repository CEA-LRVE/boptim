"""Rebuilds a `Constraint` from its `toDict()` form."""

from __future__ import annotations

from typing import Any

from boptim.domain.constraints.Constraint import Constraint
from boptim.domain.constraints.LinearConstraint import LinearConstraint


def constraintFromDict(data: dict[str, Any]) -> Constraint:
    """The inverse of `Constraint.toDict`.

    Only `LinearConstraint` exists in Phase 1. Phase 2 adds
    `NonlinearConstraint` (FR17, ADR-0006): it must be added to this
    dispatcher in the same change, or a study using one could be saved but
    not reloaded.

    Raises:
        ValueError: if `data["kind"]` is missing or unknown.
    """
    kind = data.get("kind")
    if kind == "linear":
        return LinearConstraint(
            coefficients=dict(data["coefficients"]),
            bound=data["bound"],
            comparator=data["comparator"],
        )
    raise ValueError(
        f"Cannot rebuild a Constraint from kind={kind!r}; expected 'linear' "
        "(NonlinearConstraint arrives in Phase 2)."
    )
