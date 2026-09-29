"""Sugar for a `Range` fixed to `parameter_type="float"`."""

from __future__ import annotations

from typing import Literal

from boptim.domain.parameters.Range import Range


class Real(Range):
    """`Real(name, min_value, max_value)` is sugar for
    `Range(name, (min_value, max_value), parameter_type="float")`.

    A class, not a function, on purpose: it is meant to be called like a type
    constructor (`Real("x", 0.0, 1.0)`), so it follows the `CamelCase`
    class-naming rule, not the `camelCase` function one (section 10).
    """

    def __init__(
        self,
        name: str,
        min_value: float,
        max_value: float,
        step_size: float | None = None,
        scaling: Literal["linear", "log"] | None = None,
        default: float | None = None,
    ) -> None:
        super().__init__(
            name=name,
            bounds=(min_value, max_value),
            parameter_type="float",
            step_size=step_size,
            scaling=scaling,
            default=default,
        )
