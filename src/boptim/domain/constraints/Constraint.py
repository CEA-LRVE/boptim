"""Base class for parameter-level constraints."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict


class Constraint(BaseModel, ABC):
    """Base class for parameter-level constraints (see `OutcomeConstraint`
    for metric-level ones, which are not a `Constraint` subclass since they
    apply to a different thing entirely: an observed value, not a decision
    variable).

    `LinearConstraint` is the only concrete subclass in Phase 1 (FR9,
    Ax-native). `NonlinearConstraint` (FR17) is deferred to Phase 2, since it
    is only enforceable once `boptim`'s own custom acquisition layer exists
    (section 4.6, ADR-0006); `SearchSpace.constraints: Sequence[Constraint]`
    already accepts any future `Constraint` subclass without changes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    @abstractmethod
    def toDict(self) -> dict[str, Any]:
        """Explicit, type-tagged, JSON-compatible form of this constraint,
        rebuilt by `constraintFromDict`. Same reason as `Parameter.toDict`:
        `SearchSpace` holds a list of the abstract base type.
        """
