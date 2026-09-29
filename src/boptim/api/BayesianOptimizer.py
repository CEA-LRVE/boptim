"""The public facade: `BayesianOptimizer`."""

from __future__ import annotations

import logging
import secrets
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

from boptim.analysis.PredictionResult import PredictionResult
from boptim.backends.ax.AxBackend import AxBackend
from boptim.backends.OptimizationBackend import OptimizationBackend
from boptim.domain.constraints.Constraint import Constraint
from boptim.domain.Metric import Metric
from boptim.domain.Objective import Objective
from boptim.domain.OutcomeConstraint import OutcomeConstraint
from boptim.domain.parameters.Parameter import Parameter
from boptim.domain.SearchSpace import SearchSpace
from boptim.domain.StudySnapshot import StudySnapshot
from boptim.domain.Trial import Trial
from boptim.persistence.JsonStudyRepository import JsonStudyRepository
from boptim.persistence.ReproducibilityMetadata import ReproducibilityMetadata

if TYPE_CHECKING:
    # See OptimizationBackend.py's own comment: gated behind TYPE_CHECKING so
    # importing this facade does not force a BoTorch install by itself.
    from botorch.models.model import Model

logger = logging.getLogger(__name__)

#: The metric name used for the implicit single objective built by the
#: "common case" constructor path (`objective="minimize"`/`"maximize"`).
#: Matches section 5.8's usage example (`bo.tell(x, {"objective": 0.62})`)
#: exactly.
IMPLICIT_METRIC_NAME = "objective"


class BayesianOptimizer:
    """The main entry point. Two ways to call it:

    Common case: `BayesianOptimizer(parameters=[Real(...), Integer(...)])`.
        A single, unnamed metric is assumed; `objective="minimize"` or
        `"maximize"` picks its direction.
    Advanced case: pass an already-built `SearchSpace` (carrying its own
        constraints) and an already-built `Objective` (named metrics,
        optional weights, optional outcome_constraints, single or
        multi-objective). `constraints`/`outcome_constraints` must be left
        `None` in that case (a `ValueError` is raised otherwise: the
        `SearchSpace`/`Objective` already own their own constraints, so
        passing both is an ambiguous request, not a merge).

    No other combination of `parameters`/`objective` is accepted (e.g. a
    plain parameter list together with an already-built `Objective`): the
    two `@overload` signatures above the real `__init__` are the whole
    contract, and the implementation raises a `TypeError` naming exactly
    what was passed for anything outside it, rather than silently guessing
    what a mixed call was supposed to mean.

    `backend` defaults to `AxBackend(random_seed=random_seed)`; injectable
    for testing and for a future non-Ax backend.
    """

    @overload
    def __init__(
        self,
        parameters: Sequence[Parameter],
        objective: Literal["minimize", "maximize"] = "minimize",
        constraints: Sequence[Constraint] | None = None,
        outcome_constraints: Sequence[OutcomeConstraint] | None = None,
        name: str = "study",
        random_seed: int | None = None,
        backend: OptimizationBackend | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        parameters: SearchSpace,
        objective: Objective,
        constraints: None = None,
        outcome_constraints: None = None,
        name: str = "study",
        random_seed: int | None = None,
        backend: OptimizationBackend | None = None,
    ) -> None: ...

    def __init__(
        self,
        parameters: Sequence[Parameter] | SearchSpace,
        objective: Objective | Literal["minimize", "maximize"] = "minimize",
        constraints: Sequence[Constraint] | None = None,
        outcome_constraints: Sequence[OutcomeConstraint] | None = None,
        name: str = "study",
        random_seed: int | None = None,
        backend: OptimizationBackend | None = None,
    ) -> None:
        self._search_space, self._objective = _resolveSearchSpaceAndObjective(
            parameters, objective, constraints, outcome_constraints
        )
        self._name = name
        self._trials: list[Trial] = []

        self._random_seed = (
            random_seed if random_seed is not None else secrets.randbelow(2**31 - 1)
        )
        if backend is not None:
            if random_seed is not None:
                logger.warning(
                    "Both `backend` and `random_seed` were passed to "
                    "BayesianOptimizer(): `random_seed` is recorded in "
                    "ReproducibilityMetadata, but it is not passed to the "
                    "already-constructed `backend`. If exact reproducibility "
                    "matters, construct `backend` with this same seed yourself."
                )
            self._backend = backend
        else:
            self._backend = AxBackend(random_seed=self._random_seed)

        self._backend.createExperiment(self._search_space, self._objective)

        # Captured once, at creation, not re-captured on every save(): FR12
        # asks for reproducibility metadata about the study, and `created_at`
        # in particular should mean "when this study began", not "when it was
        # last saved".
        self._reproducibility = ReproducibilityMetadata.captureCurrentEnvironment(
            self._random_seed
        )

    def tell(
        self,
        x: dict[str, float | int | str | bool],
        y: dict[str, float],
        y_std: dict[str, float] | None = None,
    ) -> Trial:
        """Manually inject an already-known point and its result(s). FR3.

        `x` maps parameter name to value. `y` maps metric name to value,
        matching however `Objective`'s metrics were named (or
        `IMPLICIT_METRIC_NAME` when the shorthand constructor was used).
        `y_std`, when known (e.g. from repeated measurements), is passed
        through as a fixed observation noise instead of being inferred by
        the surrogate model.

        If `x` exactly matches a parameterization most recently returned by
        `ask()` and not yet told, the backend completes that same pending
        trial rather than registering a new one; see
        `AxBackend.attachTrial`'s own docstring for the exact matching rule
        and its one known limitation.
        """
        pending_trial = Trial(parameters=dict(x), results=dict(y), result_std=y_std)
        trial_index = self._backend.attachTrial(pending_trial)
        recorded_trial = Trial(
            parameters=pending_trial.parameters,
            results=pending_trial.results,
            result_std=pending_trial.result_std,
            trial_index=trial_index,
        )
        self._trials.append(recorded_trial)
        return recorded_trial

    def ask(
        self,
        n_points: int = 1,
        alpha: float | None = None,
    ) -> list[dict[str, float | int | str | bool]]:
        """Ask for the next `n_points` parameterizations to evaluate.

        Args:
            n_points: batch size requested at once. FR10.
            alpha: exploration/exploitation trade-off in `[0, 1]`. `0.0`
                favors the best predicted objective, `1.0` favors the
                least-known region, `0.5` balances both. FR5. If `None`,
                delegates to the backend's own no-manual-tuning default
                strategy (FR4).

        Raises:
            ValueError: if `n_points < 1`.
            NotImplementedError: if `alpha` is not `None`. The custom
                BoTorch acquisition layer that implements FR5 (and, with it,
                the `NonlinearConstraint`-triggered automatic switch of
                ADR-0006) is Phase 2 of the roadmap (section 6); this Phase 1
                drop implements FR4's default path only, and says so clearly
                here rather than silently ignoring `alpha` or returning
                candidates that do not actually reflect it.
        """
        if n_points < 1:
            raise ValueError(f"n_points must be >= 1; got {n_points!r}.")
        if alpha is not None:
            raise NotImplementedError(
                "alpha-controlled acquisition (FR5) is not implemented in this "
                "Phase 1 drop: the custom BoTorch acquisition layer it depends "
                "on lands in Phase 2 (ADR-0001, roadmap section 6). Call "
                "ask() with alpha=None (the default) to use the Ax backend's "
                "own no-manual-tuning default generation strategy (FR4), which "
                "is fully implemented."
            )
        return self._backend.suggestDefault(n_points)

    def predict(self, x: dict[str, float | int | str | bool]) -> PredictionResult:
        """FR7, FR8."""
        raw_prediction = self._backend.predict(dict(x))
        mean = {name: mean_sem[0] for name, mean_sem in raw_prediction.items()}
        sem = {name: mean_sem[1] for name, mean_sem in raw_prediction.items()}
        return PredictionResult(mean=mean, sem=sem)

    def parameterImportance(self) -> dict[str, dict[str, float]]:
        """Refits/queries the surrogate model: has a cost, camelCase. FR6."""
        return self._backend.computeSensitivity()

    @property
    def n_trials(self) -> int:
        """`len()` over an already-held list: cheap, snake_case."""
        return len(self._trials)

    @property
    def paretoFront(self) -> list[Trial]:
        """Delegates to the backend's `getParetoFrontier()` (Ax's own
        `Client.get_pareto_frontier` under `AxBackend`, section 5.3), not a
        hand-rolled non-domination scan. Has a cost, camelCase. Collapses to
        a single-element list for a single-objective optimizer, via the
        backend's `getBestTrial()`.
        """
        if self._objective.is_multi_objective:
            return self._backend.getParetoFrontier()
        best_trial = self._backend.getBestTrial()
        return [best_trial] if best_trial is not None else []

    def save(self, path: str | Path) -> None:
        """FR11, FR12. See `JsonStudyRepository`, section 5.6."""
        snapshot = StudySnapshot(
            name=self._name,
            search_space=self._search_space,
            objective=self._objective,
            trials=list(self._trials),
            reproducibility=self._reproducibility,
            backend_state=self._backend.exportState(),
            backend_kind=_backendKind(self._backend),
        )
        JsonStudyRepository().save(snapshot, path)

    @classmethod
    def load(cls, path: str | Path) -> BayesianOptimizer:
        """FR11, FR12.

        Bypasses `__init__` (via `cls.__new__`) rather than reusing it: the
        normal constructor is for *creating* a new experiment inside the
        backend, while loading needs to *restore* a backend that already has
        its own experiment and trial history, plus boptim's own domain
        objects and reproducibility metadata exactly as they were saved, not
        rebuilt from scratch. This is documented here since it is the one
        place `BayesianOptimizer` is constructed without going through
        `__init__`.
        """
        snapshot = JsonStudyRepository().load(path)
        if snapshot.backend_kind != "ax":
            raise ValueError(
                f"Cannot load a study with backend_kind={snapshot.backend_kind!r}: "
                "Phase 1 only implements the 'ax' backend."
            )
        instance = cls.__new__(cls)
        instance._name = snapshot.name
        instance._search_space = snapshot.search_space
        instance._objective = snapshot.objective
        instance._trials = list(snapshot.trials)
        instance._reproducibility = snapshot.reproducibility
        instance._random_seed = snapshot.reproducibility.random_seed
        instance._backend = AxBackend.importState(snapshot.backend_state)
        return instance

    @property
    def axClient(self) -> Any:
        """Escape hatch (FR16, ADR-0005): the live `ax.api.client.Client`
        instance backing this optimizer. Raises `TypeError` if `backend` is
        not an `AxBackend`. Typed `Any` here to avoid forcing an Ax import on
        every caller of this file; the real return type is
        `ax.api.client.Client`. Anything reached through this property is,
        by definition, outside what boptim validates or keeps in sync with
        its own domain objects: for example, a trial attached directly via
        `axClient.attach_trial(...)` will not appear as a boptim `Trial`
        until the caller also updates the boptim side, since boptim only
        learns about it through this same escape hatch, not automatically.
        """
        if not isinstance(self._backend, AxBackend):
            raise TypeError(
                "axClient is only available when the backend is an AxBackend; "
                f"this optimizer's backend is a {type(self._backend).__name__!r}."
            )
        return self._backend.client

    def fitModel(self) -> Model:
        """Escape hatch (FR16, ADR-0005): fits and returns the current
        BoTorch surrogate model directly (the same one `acquisition/` builds
        on, from Phase 2 onward), so a caller can write and optimize their
        own acquisition function with plain BoTorch and feed the result back
        through `tell()`, without forking boptim to get an acquisition
        behavior the alpha dial does not cover.
        """
        return self._backend.fitModel()


def _resolveSearchSpaceAndObjective(
    parameters: Sequence[Parameter] | SearchSpace,
    objective: Objective | Literal["minimize", "maximize"],
    constraints: Sequence[Constraint] | None,
    outcome_constraints: Sequence[OutcomeConstraint] | None,
) -> tuple[SearchSpace, Objective]:
    is_search_space = isinstance(parameters, SearchSpace)
    is_objective = isinstance(objective, Objective)

    if is_search_space and is_objective:
        if constraints is not None or outcome_constraints is not None:
            raise ValueError(
                "When `parameters` is a SearchSpace and `objective` is an "
                "Objective (the advanced case), `constraints` and "
                "`outcome_constraints` must be left None: the SearchSpace "
                "and Objective already own their own constraints, so "
                "passing both is an ambiguous request, not a merge."
            )
        return parameters, objective

    if not is_search_space and not is_objective:
        if objective not in ("minimize", "maximize"):
            raise ValueError(
                "objective must be 'minimize' or 'maximize' when "
                f"`parameters` is a plain parameter sequence; got {objective!r}."
            )
        search_space = SearchSpace(
            parameters=list(parameters),
            constraints=list(constraints) if constraints is not None else [],
        )
        implicit_objective = Objective(
            metrics=[Metric(name=IMPLICIT_METRIC_NAME, minimize=(objective == "minimize"))],
            outcome_constraints=(
                list(outcome_constraints) if outcome_constraints is not None else []
            ),
        )
        return search_space, implicit_objective

    raise TypeError(
        "BayesianOptimizer() takes either (parameters=Sequence[Parameter], "
        "objective='minimize'|'maximize') for the common case, or "
        "(parameters=SearchSpace, objective=Objective) for the advanced case "
        "(section 5.7); got parameters as "
        f"{'a SearchSpace' if is_search_space else 'a parameter sequence'} "
        f"together with objective as "
        f"{'an Objective' if is_objective else type(objective).__name__!r}."
    )


def _backendKind(backend: OptimizationBackend) -> str:
    if isinstance(backend, AxBackend):
        return "ax"
    raise TypeError(
        f"save() does not know how to persist backend kind "
        f"{type(backend).__name__!r}: only AxBackend is supported in Phase 1."
    )
