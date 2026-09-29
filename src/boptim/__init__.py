"""boptim: a general-purpose, domain-agnostic Bayesian Optimization library.

This module re-exports boptim's public surface, so application code should
import from here (`from boptim import BayesianOptimizer, Real, ...`) rather
than reaching into `boptim.domain.parameters.Real`, etc. directly. Exempt
from section 10's file-naming rule, like every Python dunder file.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from boptim.analysis.PredictionResult import PredictionResult
from boptim.api.BayesianOptimizer import BayesianOptimizer
from boptim.backends.ax.AxBackend import AxBackend
from boptim.backends.OptimizationBackend import OptimizationBackend
from boptim.domain.constraints.Constraint import Constraint
from boptim.domain.constraints.LinearConstraint import LinearConstraint
from boptim.domain.Metric import Metric
from boptim.domain.Objective import Objective
from boptim.domain.OutcomeConstraint import OutcomeConstraint
from boptim.domain.parameters.Boolean import Boolean
from boptim.domain.parameters.Categorical import Categorical
from boptim.domain.parameters.Choice import Choice
from boptim.domain.parameters.Derived import Derived
from boptim.domain.parameters.Fixed import Fixed
from boptim.domain.parameters.Integer import Integer
from boptim.domain.parameters.Parameter import Parameter
from boptim.domain.parameters.Range import Range
from boptim.domain.parameters.Real import Real
from boptim.domain.SearchSpace import SearchSpace
from boptim.domain.StudySnapshot import StudySnapshot
from boptim.domain.Trial import Trial
from boptim.persistence.JsonStudyRepository import JsonStudyRepository
from boptim.persistence.ReproducibilityMetadata import ReproducibilityMetadata
from boptim.persistence.StudyRepository import StudyRepository

# Single source of truth is pyproject.toml (bumped by scripts/cut_release.py);
# reading it back from the installed metadata avoids a second hand-edited copy
# drifting out of sync.
try:
    __version__ = version("boptim")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0+unknown"

__all__ = [
    "AxBackend",
    "BayesianOptimizer",
    "Boolean",
    "Categorical",
    "Choice",
    "Constraint",
    "Derived",
    "Fixed",
    "Integer",
    "JsonStudyRepository",
    "LinearConstraint",
    "Metric",
    "Objective",
    "OptimizationBackend",
    "OutcomeConstraint",
    "Parameter",
    "PredictionResult",
    "Range",
    "Real",
    "ReproducibilityMetadata",
    "SearchSpace",
    "StudyRepository",
    "StudySnapshot",
    "Trial",
    "__version__",
]
