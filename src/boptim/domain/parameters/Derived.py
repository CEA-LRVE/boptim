"""A read-only quantity computed from other parameters."""

from __future__ import annotations

from typing import Any, Literal

from boptim.domain.parameters.Parameter import Parameter


class Derived(Parameter):
    """A read-only quantity computed from other parameters through an
    expression string (e.g. `"a + b"`), not itself searched over.

    Mirrors `ax.api.configs.DerivedParameterConfig` directly (its
    `expression_str` field is named `expression` here for readability; see
    `toAxKwargs`). Useful for referencing a computed quantity in a
    `LinearConstraint` or reading it back from a `Trial` without recomputing
    it outside boptim.

    A `Derived` parameter never has a `default`: it is computed, not chosen,
    so `default` always stays `None` (enforced by `_validate_default`).

    Attributes:
        expression: The expression computing this parameter's value from
            other parameters' names.
        parameter_type: The Python type this expression evaluates to.
    """

    expression: str
    parameter_type: Literal["float", "int", "str", "bool"]

    def __init__(
        self,
        name: str,
        expression: str,
        parameter_type: Literal["float", "int", "str", "bool"],
        default: None = None,
    ) -> None:
        # `default` exists only so `Derived.model_validate(d.model_dump())`
        # works (a dump contains `default=None`); anything but None is
        # rejected by `_validate_default`.
        super().__init__(
            name=name,
            expression=expression,
            parameter_type=parameter_type,
            default=default,
        )

    def _validate_default(self) -> None:
        if self.default is not None:
            raise ValueError(
                f"Derived parameter {self.name!r} cannot have a default value: "
                "it is computed from other parameters, not chosen."
            )
        if not self.expression.strip():
            raise ValueError(f"Derived parameter {self.name!r} has an empty expression.")

    def toAxKwargs(self) -> dict[str, Any]:
        """Field values ready to pass straight to
        `ax.api.configs.DerivedParameterConfig(**kwargs)`. Ax names the
        expression field `expression_str`; boptim's own field is named
        `expression` for readability, mapped here.
        """
        return {
            "name": self.name,
            "expression_str": self.expression,
            "parameter_type": self.parameter_type,
        }

    def toDict(self) -> dict[str, Any]:
        return {
            "kind": "derived",
            "name": self.name,
            "expression": self.expression,
            "parameter_type": self.parameter_type,
        }
