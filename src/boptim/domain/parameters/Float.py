from domain.parameters import Parameter
from dataclasses import dataclass

@dataclass
class Foat(Parameter):
    """
    Example #1:


    hp.Float(
        "image_rotation_factor",
        min_value=0,
        max_value=1)
    All values in interval [0, 1] have equal probability of being sampled.

    Example #2:


    hp.Float(
        "image_rotation_factor",
        min_value=0,
        max_value=1,
        step=0.2)
    step is the minimum distance between samples. The possible values are [0, 0.2, 0.4, 0.6, 0.8, 1.0].

    Example #3:


    hp.Float(
        "learning_rate",
        min_value=0.001,
        max_value=10,
        step=10,
        sampling="log")
    When sampling="log", the step is multiplied between samples. The possible values are [0.001, 0.01, 0.1, 1, 10].

    Arguments

    name: A string. the name of parameter. Must be unique for each HyperParameter instance in the search space.
    min_value: Float, the lower bound of the range.
    max_value: Float, the upper bound of the range.
    step: Optional float, the distance between two consecutive samples in the range. If left unspecified, it is possible to sample any value in the interval. If sampling="linear", it will be the minimum additve between two samples. If sampling="log", it will be the minimum multiplier between two samples.
    sampling: String. One of "linear", "log", "reverse_log". Defaults to "linear". When sampling value, it always start from a value in range [0.0, 1.0). The sampling argument decides how the value is projected into the range of [min_value, max_value]. "linear": min_value + value * (max_value - min_value) "log": min_value * (max_value / min_value) ^ value "reverse_log": (max_value - min_value * ((max_value / min_value) ^ (1 - value) - 1))
    default: Float, the default value to return for the parameter. If unspecified, the default value will be min_value.
    parent_name: Optional string, specifying the name of the parent HyperParameter to use as the condition to activate the current HyperParameter.
    parent_values: Optional list of the values of the parent HyperParameter to use as the condition to activate the current HyperParameter.
    Returns

    The value of the hyperparameter, or None if the hyperparameter is not active.
    """
    name: str
    min_value: float
    max_value: float
    step: int = None
    sampling: str = 'linear'
    default: float = None
    parent_name:str = None
    parent_values: list[float] = None