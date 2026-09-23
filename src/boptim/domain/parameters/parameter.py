from abc import ABC

class Parameter(ABC):
    """Base class for every kind of search space parameter."""

    name: str