"""Sugar for a `Range` fixed to `parameter_type="int"`."""

from __future__ import annotations

from typing import Literal

from boptim.domain.parameters.Range import Range


class Integer(Range):
    """`Integer(name, min_value, max_value)` is sugar for
    `Range(name, (min_value, max_value), parameter_type="int")`.
    """

    def __init__(
        self,
        name: str,
        min_value: int,
        max_value: int,
        step_size: int | None = None,
        scaling: Literal["linear", "log"] | None = None,
        default: int | None = None,
    ) -> None:
        super().__init__(
            name=name,
            bounds=(float(min_value), float(max_value)),
            parameter_type="int",
            step_size=float(step_size) if step_size is not None else None,
            scaling=scaling,
            default=default,
        )
