# boptim

A general-purpose, domain-agnostic Bayesian Optimization library for Python, built on top of
[Ax](https://ax.dev) and [BoTorch](https://botorch.org).

`boptim` is equally comfortable with a campaign of 4-5 extremely costly physical-experiment
evaluations, a few hundred ML hyperparameter-tuning trials, or thousands of cheap, fast
evaluations, without the caller having to pre-classify which case they are in. See
[`PROJECT_SPECIFICATION.md`](./PROJECT_SPECIFICATION.md) for the full design.

## Status

**Phase 1 of 4 (Foundation)** — see the [roadmap](./PROJECT_SPECIFICATION.md#6-roadmap).

This drop implements:

- The full domain model: `Range`/`Real`/`Integer`, `Choice`/`Categorical`/`Boolean`/`Fixed`/
  `Derived` (with `dependent_parameters`, FR14's definition side), `LinearConstraint` (FR9),
  `Objective` with `N >= 1` metrics and optional weights, `OutcomeConstraint` (FR15).
- `AxBackend`: search space and optimization-config mapping, manual trial injection (FR3),
  `ask()` via Ax's own no-manual-tuning default generation strategy (FR4), Ax's built-in
  sensitivity analysis (FR6), prediction with uncertainty (FR7, FR8), Pareto-frontier/
  best-trial lookup, and the `axClient`/`fitModel()` escape hatches (FR16, ADR-0005).
- `JsonStudyRepository`: save/load a study to/from a single JSON file, with reproducibility
  metadata (FR11, FR12).

**Not yet implemented** (Phase 2 onward): the `alpha`-controlled exploration/exploitation
acquisition layer (FR5) — `ask(alpha=...)` currently raises `NotImplementedError` — and
`NonlinearConstraint` (FR17). See `PROJECT_SPECIFICATION.md` section 6.

> **Verification status.** Written against Ax's current API reference
> (<https://ax.readthedocs.io/en/stable/api.html>), but **never run against the real
> `pydantic` / `ax-platform` / `botorch`**: the authoring sandbox blocks PyPI
> (`host_not_allowed`). What *was* done: `python -m py_compile` on every file, the
> naming-convention script, and a run of the whole test suite (104 tests), both examples and
> an ask/tell/save/load check against throwaway stand-ins for Pydantic, pytest and Ax's
> `Client` (kept outside this repo). That exercises the Python logic of this code, **not**
> Pydantic's or Ax's real behaviour. Still unverified: Pydantic v2 semantics (custom
> `__init__` + validators/serializers, union matching, the stdlib dataclass inside
> `StudySnapshot`), Ax's real `Client` behaviour (`AxBackend.fitModel` and
> `.computeSensitivity` walk non-`ax.api` internals), `mypy --strict` and `ruff`.
> Next step: `uv sync --extra dev && uv run pytest && uv run mypy && uv run ruff check .`

## Install

```bash
uv sync
```

This installs `boptim` and its pinned dependencies (`ax-platform`, `botorch`, `gpytorch`,
`torch`, `pydantic`) into a local virtual environment. For development (tests, lint, type
checking, docs):

```bash
uv sync --extra dev --extra docs
```

## Quickstart

```python
from boptim import BayesianOptimizer, Real, Integer, Categorical, Boolean

bo = BayesianOptimizer(
    parameters=[
        Real("temperature", 20.0, 120.0),
        Integer("num_layers", 1, 8),
        Categorical("solvent", ["water", "ethanol", "toluene"]),
        Boolean("use_catalyst", default=True),
    ],
    objective="maximize",
)

# seed with a point you already ran, if you have one (FR3)
bo.tell(
    {"temperature": 80.0, "num_layers": 3, "solvent": "water", "use_catalyst": True},
    {"objective": 0.62},
)

# ask for the next point, using Ax's own no-manual-tuning default strategy (FR4)
x = bo.ask()[0]
y = run_my_experiment(**x)  # your own code: a real experiment or a training run
bo.tell(x, {"objective": y})

# what does the model currently believe, with uncertainty? (FR7, FR8)
prediction = bo.predict({"temperature": 100.0, "num_layers": 4, "solvent": "ethanol", "use_catalyst": False})
print(prediction.mean, prediction.sem)

bo.parameterImportance()  # FR6
bo.save("study.json")  # FR11, FR12
bo_reloaded = BayesianOptimizer.load("study.json")
```

See [`examples/`](./examples) for two complete, runnable scripts:
[`lab_experiment.py`](./examples/lab_experiment.py) (a small-budget, high-cost-per-evaluation
campaign, seeded via `tell()`) and [`escape_hatch.py`](./examples/escape_hatch.py)
(`axClient`/`fitModel()` used directly, ADR-0005).

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run python scripts/check_naming_convention.py
```

## Documentation

Architectural decisions are recorded as ADRs under [`docs/adr/`](./docs/adr); see in
particular [ADR-0001](./docs/adr/0001-hybrid-ax-botorch-architecture.md) (why Ax + BoTorch,
not either alone) and [ADR-0005](./docs/adr/0005-escape-hatches-to-ax-and-botorch.md) (why
`axClient`/`fitModel()` exist). The naming-convention rationale (camelCase methods, `CamelCase`
file names) lives in [`docs/contributing.md`](./docs/contributing.md).

## License

TBD (section 1.4): not yet decided whether this stays internal or gets published.
