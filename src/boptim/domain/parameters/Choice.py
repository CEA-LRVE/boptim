from domain.parameters import Parameter 
from dataclasses import dataclass

@dataclass
def Choice(Parameter):
    """
    Arguments

    name: A string. the name of parameter. Must be unique for each HyperParameter instance in the search space.
    values: A list of possible values. Values must be int, float, str, or bool. All values must be of the same type.
    ordered: Optional boolean, whether the values passed should be considered to have an ordering. Defaults to True for float/int values. Must be False for any other values.
    default: Optional default value to return for the parameter. If unspecified, the default value will be:
    None if None is one of the choices in values
    The first entry in values otherwise.
    parent_name: Optional string, specifying the name of the parent HyperParameter to use as the condition to activate the current HyperParameter.
    parent_values: Optional list of the values of the parent HyperParameter to use as the condition to activate the current HyperParameter.
    Returns

    The value of the hyperparameter, or None if the hyperparameter is not active.
    """
    name: str
    values: list
    ordered: bool = None
    default = False
    parent_name:str = None
    parent_values: list[float] = None