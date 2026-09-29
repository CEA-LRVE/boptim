"""Unit tests for `BayesianOptimizer`'s facade logic.

Uses `FakeBackend` (an in-memory `OptimizationBackend`) throughout, so these
tests exercise `BayesianOptimizer`'s own logic (constructor-case resolution,
`tell`/`ask` bookkeeping, property behavior, error handling) without needing
a real Ax install.
"""

from __future__ import annotations

import pytest

from boptim import BayesianOptimizer, Metric, Objective, OutcomeConstraint, Real, SearchSpace
from fake_backend import FakeBackend  # local test double, tests/unit/api/fake_backend.py


def _searchSpace() -> SearchSpace:
    return SearchSpace(parameters=[Real("x", 0.0, 1.0)])


class TestConstructorCaseResolution:
    def test_common_case_builds_implicit_single_metric_objective(self) -> None:
        backend = FakeBackend()
        BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], objective="minimize", backend=backend)

        assert backend.created_objective is not None
        assert [m.name for m in backend.created_objective.metrics] == ["objective"]
        assert backend.created_objective.metrics[0].minimize is True

    def test_common_case_maximize(self) -> None:
        backend = FakeBackend()
        BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], objective="maximize", backend=backend)

        assert backend.created_objective is not None
        assert backend.created_objective.metrics[0].minimize is False

    def test_common_case_rejects_other_objective_strings(self) -> None:
        with pytest.raises(ValueError, match="minimize.*maximize"):
            BayesianOptimizer(
                parameters=[Real("x", 0.0, 1.0)],
                objective="maximise",  # a plausible typo, deliberately not accepted
                backend=FakeBackend(),
            )

    def test_advanced_case_uses_search_space_and_objective_as_is(self) -> None:
        search_space = _searchSpace()
        objective = Objective(metrics=[Metric(name="loss", minimize=True)])
        backend = FakeBackend()

        BayesianOptimizer(parameters=search_space, objective=objective, backend=backend)

        assert backend.created_search_space is search_space
        assert backend.created_objective is objective

    def test_advanced_case_rejects_extra_constraints(self) -> None:
        search_space = _searchSpace()
        objective = Objective(metrics=[Metric(name="loss", minimize=True)])

        with pytest.raises(ValueError, match="ambiguous"):
            BayesianOptimizer(
                parameters=search_space,
                objective=objective,
                outcome_constraints=[OutcomeConstraint("loss", 1.0, "<=")],
                backend=FakeBackend(),
            )

    def test_rejects_search_space_with_string_objective(self) -> None:
        with pytest.raises(TypeError):
            BayesianOptimizer(parameters=_searchSpace(), objective="minimize", backend=FakeBackend())

    def test_rejects_parameter_list_with_objective_instance(self) -> None:
        with pytest.raises(TypeError):
            BayesianOptimizer(
                parameters=[Real("x", 0.0, 1.0)],
                objective=Objective(metrics=[Metric(name="loss", minimize=True)]),
                backend=FakeBackend(),
            )


class TestTellAndAsk:
    def test_tell_returns_trial_with_backend_assigned_index(self) -> None:
        backend = FakeBackend()
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=backend)

        trial = bo.tell({"x": 0.5}, {"objective": 1.0})

        assert trial.trial_index == 0
        assert trial.parameters == {"x": 0.5}
        assert trial.results == {"objective": 1.0}
        assert bo.n_trials == 1

    def test_n_trials_increments_across_multiple_tells(self) -> None:
        backend = FakeBackend()
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=backend)

        bo.tell({"x": 0.1}, {"objective": 1.0})
        bo.tell({"x": 0.2}, {"objective": 2.0})

        assert bo.n_trials == 2

    def test_ask_delegates_to_backend_suggest_default(self) -> None:
        backend = FakeBackend()
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=backend)

        points = bo.ask(n_points=3)

        assert points == [{"x": 0.0}, {"x": 1.0}, {"x": 2.0}]

    def test_ask_rejects_non_positive_n_points(self) -> None:
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=FakeBackend())
        with pytest.raises(ValueError, match="n_points"):
            bo.ask(n_points=0)

    def test_ask_with_alpha_is_not_yet_implemented(self) -> None:
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=FakeBackend())
        with pytest.raises(NotImplementedError, match="Phase 2"):
            bo.ask(alpha=0.5)


class TestPredictAndImportance:
    def test_predict_reshapes_backend_mean_sem_tuples(self) -> None:
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=FakeBackend())

        prediction = bo.predict({"x": 0.5})

        assert prediction.mean == {"objective": 0.5}
        assert prediction.sem == {"objective": 0.1}
        assert prediction.variance == pytest.approx({"objective": 0.01})

    def test_parameter_importance_delegates_to_backend(self) -> None:
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=FakeBackend())
        assert bo.parameterImportance() == {"objective": {"x": 1.0}}


class TestParetoFront:
    def test_single_objective_collapses_to_best_trial(self) -> None:
        backend = FakeBackend()
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=backend)
        bo.tell({"x": 0.5}, {"objective": 1.0})

        front = bo.paretoFront

        assert len(front) == 1
        assert front[0] is backend.attached_trials[-1]

    def test_single_objective_empty_before_any_trial(self) -> None:
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=FakeBackend())
        assert bo.paretoFront == []

    def test_multi_objective_uses_pareto_frontier(self) -> None:
        search_space = _searchSpace()
        objective = Objective(metrics=[Metric(name="a", minimize=True), Metric(name="b", minimize=False)])
        backend = FakeBackend()
        bo = BayesianOptimizer(parameters=search_space, objective=objective, backend=backend)

        bo.tell({"x": 0.1}, {"a": 1.0, "b": 2.0})
        bo.tell({"x": 0.2}, {"a": 0.5, "b": 1.0})

        assert bo.paretoFront == backend.attached_trials


class TestEscapeHatches:
    def test_ax_client_raises_for_non_ax_backend(self) -> None:
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=FakeBackend())
        with pytest.raises(TypeError, match="AxBackend"):
            _ = bo.axClient

    def test_fit_model_propagates_backend_error(self) -> None:
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=FakeBackend())
        with pytest.raises(RuntimeError):
            bo.fitModel()


class TestSave:
    def test_save_rejects_non_ax_backend(self) -> None:
        # Fails before any file I/O happens (StudySnapshot's backend_kind
        # argument is evaluated, and raises, before save() ever calls
        # JsonStudyRepository().save()), so no path/tmp_path is needed here.
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=FakeBackend())
        with pytest.raises(TypeError, match="AxBackend"):
            bo.save("unused-path.json")
