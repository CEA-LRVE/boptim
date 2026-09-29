"""Sugar for a `Choice` of strings with no meaningful order."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from boptim.domain.parameters.Choice import Choice


class Categorical(Choice):
    """`Categorical(name, categories)` is sugar for
    `Choice(name, categories, parameter_type="str", is_ordered=False)`.
    """

    def __init__(
        self,
        name: str,
        categories: list[str],
        dependent_parameters: Mapping[str, Sequence[str]] | None = None,
        default: str | None = None,
    ) -> None:
        super().__init__(
            name=name,
            values=categories,
            parameter_type="str",
            is_ordered=False,
            dependent_parameters=dependent_parameters,
            default=default,
        )
