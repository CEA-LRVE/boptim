from domain.parameters import Parameter
from dataclasses import dataclass
from typing import Any

@dataclass
class Fixed(Parameter):
    """
    Arguments

        name: A string. the name of parameter. Must be unique for each HyperParameter instance in the search space.
        value: The value to use (can be any JSON-serializable Python type).
        parent_name: Optional string, specifying the name of the parent HyperParameter to use as the condition to activate the current HyperParameter.
        parent_values: Optional list of the values of the parent HyperParameter to use as the condition to activate the current HyperParameter.
        Returns

        The value of the hyperparameter, or None if the hyperparameter is not active.
    """
    name: str
    value: Any
    parent_values: list[float] = None