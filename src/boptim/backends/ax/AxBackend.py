"""The `OptimizationBackend` implementation backed by `ax.api.client.Client`."""

from __future__ import annotations

import json
import logging
import secrets
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from ax.api.client import Client
from botorch.models.model import Model

from boptim.backends.ax.toAxOptimizationConfig import toAxOptimizationConfig
from boptim.backends.ax.toAxSearchSpace import toAxSearchSpace
from boptim.backends.OptimizationBackend import OptimizationBackend
from boptim.domain.constraints.LinearConstraint import LinearConstraint
from boptim.domain.Objective import Objective
from boptim.domain.SearchSpace import SearchSpace
from boptim.domain.Trial import Trial

logger = logging.getLogger(__name__)

_AxValue = float | int | str | bool
_AxRawDataValue = float | tuple[float, float]


class AxBackend(OptimizationBackend):
    """The `OptimizationBackend` implementation backed by
    `ax.api.client.Client`.

    Implements every `OptimizationBackend` method by delegating to a private
    `ax.api.client.Client` instance; exact accessor names confirmed against
    https://ax.readthedocs.io/en/stable/api.html for this draft, re-check
    against whatever Ax version `pyproject.toml` ends up pinning.

    `random_seed`, `method`, `initialization_budget`, `torch_device` are
    passed straight through to `Client.configure_generation_strategy`
    (section 4.4); not reinvented, just exposed at construction time.
    """

    def __init__(
        self,
        random_seed: int | None = None,
        method: Literal["quality", "fast", "random_search"] = "fast",
        initialization_budget: int | None = None,
        torch_device: str | None = None,
    ) -> None:
        # A concrete seed is always resolved and recorded, even when the
        # caller leaves `random_seed=None`, so ReproducibilityMetadata (FR12)
        # never has to record "unknown". See BayesianOptimizer.__init__.
        self._random_seed: int = (
            random_seed if random_seed is not None else secrets.randbelow(2**31 - 1)
        )
        self._method: Literal["quality", "fast", "random_search"] = method
        self._initialization_budget = initialization_budget
        self._torch_device = torch_device

        self._client: Client = Client(random_seed=self._random_seed)

        # Maps a canonical (sorted-items-tuple) parameterization to the Ax
        # trial_index that `suggestDefault` created it under, so that
        # `attachTrial` can `complete_trial` the trial Ax already has RUNNING
        # for a point returned by `ask()`, instead of creating a second,
        # orphaned one via `attach_trial`. Not part of section 5.3's ABC:
        # a purely internal bookkeeping detail of this backend. See
        # `attachTrial`'s own docstring for the full rationale.
        self._pending_trials: dict[tuple[tuple[str, _AxValue], ...], int] = {}

        self._search_space: SearchSpace | None = None
        self._objective: Objective | None = None
        self._metric_names: list[str] = []

    def createExperiment(self, search_space: SearchSpace, objective: Objective) -> None:
        self._search_space = search_space
        self._objective = objective
        self._metric_names = [metric.name for metric in objective.metrics]

        ax_parameters = toAxSearchSpace(search_space)
        linear_constraint_strings = [
            constraint.toAxParameterConstraintString()
            for constraint in search_space.constraints
            if isinstance(constraint, LinearConstraint)
        ]
        self._client.configure_experiment(
            parameters=ax_parameters,
            parameter_constraints=linear_constraint_strings or None,
        )

        objective_string, outcome_constraint_strings = toAxOptimizationConfig(objective)
        self._client.configure_optimization(
            objective=objective_string,
            outcome_constraints=outcome_constraint_strings or None,
        )

        self._client.configure_generation_strategy(
            method=self._method,
            initialization_budget=self._initialization_budget,
            initialization_random_seed=self._random_seed,
            torch_device=self._torch_device,
        )

    def attachTrial(self, trial: Trial) -> int:
        """Registers `trial`, returning its Ax trial index.

        If `trial.parameters` exactly matches a parameterization most
        recently returned by `suggestDefault` (and not yet told), the
        already-RUNNING Ax trial for it is completed in place. Otherwise (the
        FR3 case: prior data, or a point evaluated outside boptim entirely)
        a brand-new trial is attached and immediately completed.

        Matching is by exact equality of the parameterization dict. If the
        caller mutates a point between `ask()` and `tell()`, a new trial is
        attached instead of completing the pending one, leaving the original
        Ax trial stuck at RUNNING; this is a known, accepted limitation of
        keying by value rather than by an explicit trial handle (the
        `OptimizationBackend.attachTrial(trial: Trial)` signature has no
        trial-handle parameter to key on instead).
        """
        raw_data = _toAxRawData(trial.results, trial.result_std)
        key = _canonicalParameters(trial.parameters)
        pending_index = self._pending_trials.pop(key, None)
        trial_index = (
            pending_index
            if pending_index is not None
            else self._client.attach_trial(parameters=trial.parameters)
        )
        self._client.complete_trial(trial_index=trial_index, raw_data=raw_data)
        return trial_index

    def suggestDefault(self, n_points: int) -> list[dict[str, _AxValue]]:
        next_trials = self._client.get_next_trials(max_trials=n_points)
        suggestions: list[dict[str, _AxValue]] = []
        for trial_index, parameterization in next_trials.items():
            parameters: dict[str, _AxValue] = dict(parameterization)
            self._pending_trials[_canonicalParameters(parameters)] = trial_index
            suggestions.append(parameters)
        return suggestions

    def fitModel(self) -> Model:
        """Best-effort extraction of the BoTorch `Model` currently fitted
        inside Ax's `GenerationStrategy`.

        This walks Ax's internal object graph (`GenerationStrategy` ->
        `Adapter`/`ModelBridge` -> `Surrogate` -> BoTorch `Model`), which is
        *not* part of the stable `ax.api` surface this file otherwise
        restricts itself to. It is deliberately isolated to this one method
        so that, per ADR-0005's own accepted trade-off ("anything reached
        through this escape hatch is outside what boptim validates"), a
        future Ax version needing a different attribute chain only requires
        editing this one method.
        """
        try:
            generation_strategy = self._client._generation_strategy  # noqa: SLF001
            adapter = generation_strategy.model
            surrogate = getattr(adapter, "surrogate", None)
            model = getattr(surrogate, "model", None) if surrogate is not None else None
            if model is None:
                model = getattr(adapter, "model", None)
            if not isinstance(model, Model):
                raise TypeError(
                    f"resolved object is a {type(model).__name__!r}, not a botorch Model"
                )
            return model
        except Exception as error:  # noqa: BLE001 - deliberately broad, see docstring
            raise RuntimeError(
                "Could not extract a fitted BoTorch model from Ax's current "
                "GenerationStrategy. This may mean there are not yet enough "
                "trials for Ax to have fit a model (see `initialization_budget`), "
                "or that the pinned Ax version restructured its internal "
                "GenerationStrategy/Adapter/Surrogate object graph, which "
                "`fitModel()` walks outside of Ax's own `ax.api` stability "
                "guarantees (ADR-0005)."
            ) from error

    def predict(self, x: dict[str, _AxValue]) -> dict[str, tuple[float, float]]:
        return self._client.predict([x])[0]

    def computeSensitivity(self) -> dict[str, dict[str, float]]:
        """Delegates to Ax's own `Client.compute_analyses` with a
        `SensitivityAnalysisPlot` per metric.

        Ax's `Analysis` module is explicitly documented as not part of the
        stable `ax.api` surface ("its methods are subject to change
        incompatibly between minor versions"). This method is therefore
        defensive: a metric whose sensitivity cannot be extracted (too few
        trials, an unrecognized `AnalysisCard` shape for the pinned Ax
        version, etc.) logs a warning and contributes an empty `{}` rather
        than raising, so one metric's failure does not take down FR6 for
        every other metric.
        """
        metric_names = self._metric_names or self._resolveMetricNames()
        try:
            from ax.analysis.plotly.sensitivity import SensitivityAnalysisPlot
        except ImportError as error:
            logger.warning(
                "computeSensitivity() could not import Ax's "
                "SensitivityAnalysisPlot (its module path may have moved for "
                "the pinned Ax version): %s", error,
            )
            return {metric_name: {} for metric_name in metric_names}

        result: dict[str, dict[str, float]] = {}
        for metric_name in metric_names:
            try:
                cards = self._client.compute_analyses(
                    analyses=[SensitivityAnalysisPlot(metric_name=metric_name)],
                    display=False,
                )
                result[metric_name] = _extractParameterImportance(cards)
            except Exception as error:  # noqa: BLE001 - see docstring
                logger.warning(
                    "computeSensitivity() could not extract sensitivity for "
                    "metric %r: %s", metric_name, error,
                )
                result[metric_name] = {}
        return result

    def getParetoFrontier(self) -> list[Trial]:
        try:
            entries = self._client.get_pareto_frontier(use_model_predictions=True)
        except Exception as error:  # noqa: BLE001
            logger.warning("getParetoFrontier() failed: %s", error)
            return []
        return [
            _axBestEntryToTrial(parameters, means, trial_index)
            for parameters, means, trial_index, _arm_name in entries
        ]

    def getBestTrial(self) -> Trial | None:
        try:
            parameters, means, trial_index, _arm_name = self._client.get_best_parameterization(
                use_model_predictions=True
            )
        except Exception as error:  # noqa: BLE001
            logger.warning("getBestTrial() failed (no completed trials yet?): %s", error)
            return None
        return _axBestEntryToTrial(parameters, means, trial_index)

    def exportState(self) -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "ax_client_snapshot.json"
            self._client.save_to_json_file(str(tmp_path))
            return dict(json.loads(tmp_path.read_text(encoding="utf-8")))

    @classmethod
    def importState(cls, state: dict[str, Any]) -> AxBackend:
        backend = cls()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "ax_client_snapshot.json"
            tmp_path.write_text(json.dumps(state), encoding="utf-8")
            backend._client = Client.load_from_json_file(str(tmp_path))
        backend._metric_names = backend._resolveMetricNames()
        return backend

    def _resolveMetricNames(self) -> list[str]:
        """Falls back to Ax's own `Experiment.optimization_config` when
        `_metric_names` was not populated by `createExperiment` (i.e. after
        `importState`, where boptim's own `Objective` is supplied separately
        by the caller and this backend only has the raw Ax state to go on).
        """
        try:
            optimization_config = self._client._experiment.optimization_config  # noqa: SLF001
            return list(optimization_config.metrics.keys())
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "Could not resolve metric names from the Ax experiment; "
                "computeSensitivity() will return an empty result: %s", error,
            )
            return []

    @property
    def client(self) -> Any:
        """The live `ax.api.client.Client`. Returned as `Any` here to avoid
        leaking an Ax import into this file's own public signature; exposed
        to end users, typed as Ax's own `Client`, via
        `BayesianOptimizer.axClient` (section 5.7, ADR-0005).
        """
        return self._client


def _canonicalParameters(
    parameters: Mapping[str, _AxValue],
) -> tuple[tuple[str, _AxValue], ...]:
    return tuple(sorted(parameters.items()))


def _toAxRawData(
    results: dict[str, float], result_std: dict[str, float] | None
) -> dict[str, _AxRawDataValue]:
    raw_data: dict[str, _AxRawDataValue] = {}
    for name, mean in results.items():
        if result_std is not None and name in result_std:
            raw_data[name] = (mean, result_std[name])
        else:
            raw_data[name] = mean
    return raw_data


def _meanAndSem(value: float | tuple[float, float]) -> tuple[float, float | None]:
    if isinstance(value, tuple):
        return value[0], value[1]
    return value, None


def _axBestEntryToTrial(
    parameters: Mapping[str, _AxValue],
    means: Mapping[str, float | tuple[float, float]],
    trial_index: int,
) -> Trial:
    results: dict[str, float] = {}
    result_std: dict[str, float] = {}
    for metric_name, value in means.items():
        mean, sem = _meanAndSem(value)
        results[metric_name] = mean
        if sem is not None:
            result_std[metric_name] = sem
    return Trial(
        parameters=dict(parameters),
        results=results,
        result_std=result_std or None,
        trial_index=trial_index,
    )


def _extractParameterImportance(cards: list[Any]) -> dict[str, float]:
    """Best-effort extraction of `{parameter_name: importance}` from the
    `AnalysisCardBase` list `Client.compute_analyses` returns for a
    `SensitivityAnalysisPlot`.

    `AnalysisCard`'s own shape is explicitly not part of Ax's stable API
    (see `computeSensitivity`'s docstring), so this tries a couple of
    plausible attribute/column names rather than committing to one; it
    returns `{}` if none match, which `computeSensitivity` logs as a
    per-metric warning rather than a hard failure.
    """
    importance: dict[str, float] = {}
    for card in cards:
        data_frame = getattr(card, "df", None)
        if data_frame is None:
            continue
        columns = set(data_frame.columns)
        parameter_column = next(
            (c for c in ("parameter_name", "parameter") if c in columns), None
        )
        value_column = next(
            (c for c in ("sensitivity", "importance", "value") if c in columns), None
        )
        if parameter_column is None or value_column is None:
            continue
        for _, row in data_frame.iterrows():
            importance[str(row[parameter_column])] = float(row[value_column])
    return importance
