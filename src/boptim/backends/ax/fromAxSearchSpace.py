"""Reconstructs a boptim `SearchSpace` from Ax's own core `Parameter` objects."""

from __future__ import annotations

from typing import Any

from ax.core.parameter import ChoiceParameter, DerivedParameter, FixedParameter, RangeParameter
from ax.core.parameter import ParameterType as AxParameterType

from boptim.domain.parameters.Choice import Choice
from boptim.domain.parameters.Derived import Derived
from boptim.domain.parameters.Fixed import Fixed
from boptim.domain.parameters.Parameter import Parameter
from boptim.domain.parameters.Range import Range
from boptim.domain.SearchSpace import SearchSpace

_AX_TO_BOPTIM_TYPE: dict[AxParameterType, str] = {
    AxParameterType.FLOAT: "float",
    AxParameterType.INT: "int",
    AxParameterType.STRING: "str",
    AxParameterType.BOOL: "bool",
}


def fromAxSearchSpace(ax_parameters: list[Any]) -> SearchSpace:
    """The inverse of `toAxSearchSpace`, used when reconstructing a
    `SearchSpace` from a study loaded through `Client.load_from_json_file`
    (section 5.6).

    Accepts Ax's own *core* parameter objects (`ax.core.parameter.
    RangeParameter`/`ChoiceParameter`/`FixedParameter`/`DerivedParameter`,
    e.g. `list(client._experiment.search_space.parameters.values())`), not
    the `ax.api.configs` config objects `toAxSearchSpace` produces: those are
    write-only inputs to `configure_experiment`, while the experiment Ax
    actually holds (and serializes) is made of the core objects instead.

    Scope note (flagged for review per section 12): a `Parameter`'s `default`
    (FR13) and a `Choice`'s `dependent_parameters` (FR14) have no equivalent
    field on Ax's core parameter objects either (mirroring `ax.api.configs`'
    own lack of a `default` field), so neither round-trips through this
    function. `JsonStudyRepository` therefore treats boptim's own
    `StudySnapshot.search_space` (kept alongside, not re-derived from Ax's
    state) as the source of truth for those two pieces of metadata, and only
    uses this function where a boptim `SearchSpace` must be derived from a
    `Client` that boptim did not itself create (e.g. one built directly
    through the `axClient` escape hatch, ADR-0005).
    """
    parameters: list[Parameter] = []
    for ax_parameter in ax_parameters:
        parameters.append(_fromAxParameter(ax_parameter))
    return SearchSpace(parameters=parameters)


def _fromAxParameter(ax_parameter: Any) -> Parameter:
    if isinstance(ax_parameter, RangeParameter):
        return Range(
            name=ax_parameter.name,
            bounds=(float(ax_parameter.lower), float(ax_parameter.upper)),
            parameter_type=_axTypeToBoptim(ax_parameter.parameter_type, allow_str_bool=False),  # type: ignore[arg-type]
            step_size=getattr(ax_parameter, "step_size", None),
            scaling="log" if getattr(ax_parameter, "log_scale", False) else None,
        )
    if isinstance(ax_parameter, FixedParameter):
        return Fixed(name=ax_parameter.name, value=ax_parameter.value)
    if isinstance(ax_parameter, ChoiceParameter):
        return Choice(
            name=ax_parameter.name,
            values=list(ax_parameter.values),
            parameter_type=_axTypeToBoptim(ax_parameter.parameter_type, allow_str_bool=True),  # type: ignore[arg-type]
            is_ordered=ax_parameter.is_ordered,
        )
    if isinstance(ax_parameter, DerivedParameter):
        return Derived(
            name=ax_parameter.name,
            expression=ax_parameter.expression_str,
            parameter_type=_axTypeToBoptim(ax_parameter.parameter_type, allow_str_bool=True),  # type: ignore[arg-type]
        )
    raise TypeError(
        f"No boptim Parameter mapping is defined for Ax parameter kind "
        f"{type(ax_parameter).__name__!r}."
    )


def _axTypeToBoptim(ax_type: AxParameterType, *, allow_str_bool: bool) -> str:
    boptim_type = _AX_TO_BOPTIM_TYPE[ax_type]
    if not allow_str_bool and boptim_type not in ("float", "int"):
        raise ValueError(
            f"Ax RangeParameter reported parameter_type={ax_type!r}, which has "
            f"no float/int boptim Range equivalent."
        )
    return boptim_type
