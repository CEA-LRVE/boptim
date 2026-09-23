from domain.parameters import Parameter 
from dataclasses import dataclass

@dataclass
def Boolean(Parameter):
    """
    Arguments

    name: A string. the name of parameter. Must be unique for each HyperParameter instance in the search space.
    default: Boolean, the default value to return for the parameter. If unspecified, the default value will be False.
    parent_name: Optional string, specifying the name of the parent HyperParameter to use as the condition to activate the current HyperParameter.
    parent_values: Optional list of the values of the parent HyperParameter to use as the condition to activate the current HyperParameter.
    Returns

    The value of the hyperparameter, or None if the hyperparameter is not active.

    """
    name: str
    default: bool = False
    parent_name:str = None
    parent_values: list[float] = None