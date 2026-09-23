# boptim: Project Specification

**A general-purpose, domain-agnostic Bayesian Optimization library for Python.**

- Version: 0.1 (draft)
- Date: 2026-09-23
- Status: Draft, pending review by the project owner

> Scope note: this document fixes the technology stack, the architecture, the functional
> specification, the roadmap, and the signatures of the main files, classes, methods and
> functions. It deliberately stops at signatures and short docstrings, not full
> implementations: the goal is to agree on the contract before writing code. `boptim` is a
> working name for the package; check PyPI availability before publishing and rename freely.

---

## 0. Language & conventions

- Conversations about the project may happen in any language (typically French).
- All code, docstrings, comments, config keys, commit messages, and technical documentation
  **must be written in English** (research/industry standard, needed for eventual publication
  and collaboration).

---

## 1. Project overview

### 1.1 What this is

`boptim` is a standalone Python library implementing Bayesian Optimization (BO) as a
**general-purpose, domain-agnostic** tool. It is a generic black-box optimization engine 
that is used, among other
things, by a laboratory and by machine-learning practitioners. Concretely, this means the
library must be equally comfortable with:

- A campaign of **4-5 evaluations total**, each one extremely costly (physical experiments
  that can cost on the order of EUR 1,000,000 each), where every single suggested point
  matters and no evaluation can be "wasted" on uninformed exploration.
- A campaign of a **few hundred evaluations** (typical ML hyperparameter tuning).
- Campaigns that may run into the **thousands** of evaluations for cheap, fast objectives.
- Observations that are **noisy** (repeating the same point gives a different result) or
  **effectively deterministic**, without the caller having to pre-classify which case they
  are in.

### 1.2 What this is not (for now)

- Not a REST/HTTP service. `boptim` is a pure Python library, imported directly into whatever
  code calls it. A service layer is an explicit non-goal for this phase.
- Not the end-user application itself. The "application for the lab" mentioned in the original
  request is a separate, later project that will sit on top of `boptim`.

### 1.3 Design philosophy

1. **Flexibility over convention.** Every numeric default (initial design size, noise
   assumption, exploration/exploitation blend, batch size, etc.) is a *default*, never a
   hardcoded assumption. Nothing in the domain layer encodes "this is for a lab" or "this is
   for ML".
2. **`boptim`'s public API is its own.** Ax and BoTorch are implementation details behind an
   adapter, not something a caller has to understand. This is not academic caution: Ax's own
   recommended entry point changed from `AxClient` to the newer `ax.api.client.Client` in the
   last couple of years, and we do not want that kind of upstream churn to leak into
   `boptim`'s public API.
3. **Single-objective is the N = 1 case of multi-objective**, not a separate code path. There
   is one `Objective` concept everywhere in the codebase.
4. **The exploration/exploitation dial is orthogonal to everything else.** It is not tied to
   a specific number of objectives, a specific noise model, or a specific budget size.

### 1.4 Defaults chosen for this draft (flag for review)

These are places where a decision was needed to keep the document concrete. None of them are
expensive to change later; they are called out so they can be corrected in one pass.

| Item | Default chosen | Why |
|---|---|---|
| Package name | `boptim` | Placeholder. Short, descriptive, likely available; verify on PyPI. |
| `mypy` strictness | `--strict` | Matches "type hints mandatory everywhere" from section 10. |
| License | Not set (TBD) | Depends on whether this stays internal or gets published; not an architectural decision. |
| Build backend | `uv_build` (or `hatchling`) under `uv` | See ADR-0002 in section 4.6. |

---

## 2. Functional requirements

| ID | Requirement | Notes |
|---|---|---|
| FR1 | Define a search space with `float`, `int`, categorical and boolean parameters, with bounds and linear/log scales. | Log scale applies to `float`/`int` parameters. |
| FR2 | Define one or more objectives, each minimized or maximized, with optional relative importance weights. | N = 1 must behave identically to a "simple" single-objective API; N > 1 supports both weighted and unweighted (Pareto) preference. |
| FR3 | Manually inject an arbitrary (parameters, result) pair, bypassing `ask()`. | Needed to seed a study from prior data, and to record points evaluated outside `boptim`. |
| FR4 | Suggest the next point(s) automatically with a high-performing strategy, with no manual tuning of surrogate-model or acquisition-function hyperparameters required. | This is the "no manual tuning" requirement; it governs the *default* path, not the alpha-controlled path (FR5). |
| FR5 | Control the exploration/exploitation trade-off with a single continuous parameter `alpha` in `[0, 1]`: `0.0` favors the best predicted objective (pure exploitation), `1.0` favors the least-known region of the search space (pure exploration), `0.5` balances both. | This is the requirement Ax does not expose natively; see section 4.3. |
| FR6 | Report each parameter's importance with respect to each objective. | |
| FR7 | Predict the expected result for an arbitrary (not necessarily evaluated) parameterization. | |
| FR8 | Report how reliable a prediction or a suggestion is (calibrated uncertainty), not just a point estimate. | Applies to FR7 and, where meaningful, to FR6. |
| FR9 | Support constraints across parameters (e.g. a mixture's proportions summing to 1, or linear inequalities between parameters). | Linear constraints for V1; see section 6 for nonlinear constraints as a later extension. |
| FR10 | Suggest a batch of `n >= 1` points in a single call, jointly optimized (not `n` copies of the same point). | Needed for parallel evaluations (e.g. several reactors, a well plate, several ML training jobs at once). |
| FR11 | Save a study to disk and reload it later, resuming exactly where it left off. | A campaign can span days or weeks with the process restarting in between. |
| FR12 | Make a saved study reproducible: the random seed, library versions, and full trial history are recorded, not just the final state. | |
| FR13 | Support a per-parameter default value and, for `Real`/`Integer`, an optional fixed sampling step. | Ergonomics, not core optimization logic; added after seeing Keras Tuner's `HyperParameters` API. |
| FR14 | Support conditional (hierarchical) parameters: a parameter that only applies when another parameter takes a specific value. | Added after seeing Keras Tuner's `conditional_scope`. Definition/persistence is solid; genuine optimization-quality benefit from the structure is weaker and tracked separately, see section 4.5. |

---

## 3. Technology stack

| Layer | Technology | Why |
|---|---|---|
| Language / runtime | Python 3.11+ | Given. |
| Optimization core | Ax (`ax-platform`, the `ax.api.client.Client` API) | Search space definition, trial/experiment bookkeeping, a strong default generation strategy, built-in sensitivity analysis and prediction, JSON persistence primitives. Covers most of FR1-FR4, FR6, FR7, FR9-FR11 largely "for free". |
| Custom acquisition | BoTorch | Ax's automatic strategy selection is objective-driven (EI/NEI/UCB-family); it does not expose a pure-exploration, objective-independent acquisition function. BoTorch does (`qNegIntegratedPosteriorVariance` and friends), so FR5 is built as a custom acquisition layer on top of BoTorch models, orchestrated around Ax's data. See ADR-0001. |
| Tensor / GP backend | PyTorch, GPyTorch | Transitive dependencies of Ax/BoTorch, not called directly except where the custom acquisition layer needs raw posterior access. |
| Domain / config models | Pydantic v2 | Typed, validated, JSON-serializable-by-construction models for `boptim`'s own public types (parameters, objectives, constraints, trials). Keeps the public API decoupled from Ax's internal types (design philosophy #2). |
| Packaging / environment | `uv` | See ADR-0002. |
| Lint / format | `ruff` | Given. |
| Type checking | `mypy --strict` | Given ("type hints mandatory everywhere"), strict mode chosen as the concrete default. |
| Testing | `pytest` | Given. |
| Logging | standard `logging` module | Given. |
| Docs | `mkdocs` + `mkdocstrings` (Material theme) | Given. |
| CI | GitHub Actions | Given. |

---

## 4. Architecture

### 4.1 Component overview

```
api/  BayesianOptimizer                     <- the single public entry point
  |
  +-- domain/         pure Pydantic models, zero ML dependencies
  |
  +-- backends/ax/     Ax Client adapter: search space, trial bookkeeping,
  |                    default generation strategy, sensitivity, predictions   --+
  |                                                                              |
  +-- acquisition/     custom BoTorch acquisition layer: the alpha dial,        +-- Ax / BoTorch /
  |                    batch generation, multi-objective variant                |   GPyTorch
  |                                                                              |
  +-- analysis/        sensitivity + prediction wrappers (Ax-backed, with      --+
  |                    a non-Ax fallback for the BoTorch-only code path)
  |
  +-- persistence/     JSON snapshots + reproducibility metadata (seed,
                       library versions, full trial history)
```

- **`domain`** has zero ML dependencies. It is what makes the library's public surface stable
  even if the backend changes.
- **`backends/ax`** is the only place that imports Ax. It translates `domain` objects to and
  from Ax's `Client`, and runs the *default* (non-alpha-controlled) generation strategy,
  predictions, and sensitivity analysis.
- **`acquisition`** is the only place that imports BoTorch acquisition machinery directly. It
  implements FR5 (the alpha dial) and its batch and multi-objective variants, working off the
  fitted model that the Ax backend exposes.
- **`analysis`** wraps whichever of Ax's built-in analyses we use, plus a fallback
  implementation that does not depend on Ax's internals, so the library remains usable even in
  a code path that only calls BoTorch directly.
- **`persistence`** owns the on-disk format and reproducibility metadata (FR11, FR12).

### 4.2 Data flow for a typical call

1. Caller constructs a `BayesianOptimizer`, either from a plain list of `Parameter`s (the
   common case) or from an explicit `SearchSpace` + `Objective` (the case that needs
   constraints or multi-objective weighting spelled out in full). See section 5.7.
2. Caller calls `bo.tell(x, y)` zero or more times to inject prior data (FR3).
3. Caller calls `bo.ask(n_points=..., alpha=...)`.
   - If `alpha` is left at its default (see 4.3), `BayesianOptimizer` delegates to the Ax
     backend's default generation strategy.
   - Otherwise, `BayesianOptimizer` asks the Ax backend to fit a surrogate model on the current
     data, then hands that model to the `acquisition` layer, which builds and optimizes the
     custom alpha-blended acquisition function to produce `n_points` candidates.
4. Caller evaluates the suggested point(s) (outside `boptim`, this is where a real experiment
   or a training run happens) and calls `bo.tell(x, y)` again with the result(s).
5. At any point, caller can call `bo.predict(x)`, `bo.parameterImportance()`,
   `bo.paretoFront`, `bo.save(path)`.

### 4.3 The exploration/exploitation dial (FR5)

This is the part Ax does not give us for free, so it deserves its own explanation.

Ax's automatic strategy selection picks a generation strategy based on the structure of the
problem (search space size, number of trials so far, etc.), but the models it picks from are
all objective-driven acquisition functions (Expected Improvement, Noisy EI, UCB and similar).
There is no "ignore the objective, just reduce my uncertainty everywhere" mode exposed through
the high-level API. BoTorch has exactly that building block
(`qNegIntegratedPosteriorVariance`, an acquisition function designed purely for active
learning / exploration), but wiring it into Ax requires stepping outside the high-level flow
and writing a custom generation step, which is exactly why this project needs its own
acquisition layer rather than relying on Ax alone.

`boptim`'s acquisition layer defines two terms over the fitted surrogate model, for a
candidate point `x`:

- **Exploitation term:** the posterior mean at `x` (direction-adjusted for minimize/maximize),
  min-max normalized over a Sobol-sampled reference set spanning the search space. At
  `alpha = 0`, the acquisition function reduces to "go to the point the model currently
  believes is best", independent of how uncertain that belief is. This is deliberately not
  Expected Improvement, which already blends in some uncertainty-seeking behavior; keeping the
  two extremes clean (mean-only vs. variance-only) is what makes `alpha` behave the way the
  original spec describes it.
- **Exploration term:** the posterior predictive variance at `x` (the same quantity
  `qNegIntegratedPosteriorVariance` integrates over the search space), normalized the same
  way.

The blended score is `score(x) = (1 - alpha) * exploitation(x) + alpha * exploration(x)`,
maximized by BoTorch's standard `optimize_acqf`. For a batch of `n_points > 1`, points are
chosen sequentially using BoTorch's fantasization mechanism (each already-chosen point in the
batch is treated as a pending, model-predicted observation before picking the next one), which
is BoTorch's standard way of getting a diverse batch instead of `n_points` copies of the same
optimum.

For multi-objective studies, the same blend applies, but the exploitation term becomes a
(weighted, if `Objective.weights` is set) hypervolume-improvement-style term over the current
Pareto front, and the exploration term becomes an aggregate of the per-objective posterior
variance. `Objective.weights` controls *preference between objectives*; `alpha` controls
*exploration vs. exploitation given that preference*. The two are orthogonal by design.

### 4.4 Handling the full budget range (FR: implicit, from the "4-5 to thousands" requirement)

- The number of purely space-filling initial points (Sobol design) before model-based
  suggestions start is a configurable parameter of `BayesianOptimizer`, defaulting to a small,
  data-driven heuristic, and can be set to `0` when the caller intends to seed the study
  entirely through `tell()` and cannot afford to "spend" evaluations on a random design.
- Below a configurable minimum number of observed points (a handful), a full marginal
  likelihood fit of the GP hyperparameters is not reliable; the surrogate model factory falls
  back to fixed/weakly-informative priors on lengthscales and noise rather than free
  optimization, and `BayesianOptimizer` surfaces this state so a caller building a UI on top can show an
  explicit "low-confidence model" indicator rather than silently reporting an overconfident
  uncertainty estimate.
- Scaling to very large budgets (thousands of points, where a plain GP's O(n^3) fitting
  cost becomes a real bottleneck) is out of scope for V1, but the `models/` layer is a single,
  swappable factory function specifically so that a sparse/scalable GP can be plugged in later
  without touching the acquisition or API layers. See section 6.

### 4.5 Conditional (hierarchical) parameters

Prompted by Keras Tuner's `conditional_scope`/`parent_name` pattern: a parameter that only
makes sense given another parameter's value (e.g. `num_filters` only matters when
`model_type == "cnn"`). Kept declarative rather than callback-based (`Parameter.active_when`,
section 5.2), because a static, inspectable search space is what `persistence/` needs to save
and reload a study exactly, and what a future GUI would need to render a form that shows or
hides fields as the user picks values, neither of which is as natural if the space is only
discovered by actually running a build function the way Keras Tuner's `hp` object is.

Honesty check on the backend: Ax does support this (`HierarchicalSearchSpace`, exposed as a
`dependents` mapping on a choice parameter), so `active_when` has a real place to map to in
`backends/ax`. Ax's own team describes the current default behavior as flattening the space
under the hood for model fitting rather than doing fully tree-aware optimization, noting that
this "works shockingly well" empirically but is still an active area of their research. Two
consequences worth being upfront about: (1) the definition, persistence, and introspection of
a conditional search space is solid and can be built as part of the domain/backend work in
Phase 1; (2) genuinely exploiting the structure during optimization (rather than just not
crashing on it) is weaker today than a mature, flat search space, both in Ax's default
strategy and in `boptim`'s own custom acquisition layer (section 4.3), which does not yet
reason about `active_when` at all. Treat (1) as a Phase 1/2 item and (2) as a distinct,
later roadmap item (section 6), not something the `Condition` class alone solves.

### 4.6 Key decisions (ADRs)

Full ADRs live under `docs/adr/`, one file per decision, following the format below. Two are
written out in full here as the two decisions with the most day-to-day impact; the rest are
listed with the context that should seed them when they are written, following the same
format.

#### ADR-0001: Hybrid Ax + BoTorch architecture

**Status:** Accepted
**Date:** 2026-09-23
**Deciders:** [project owner]

**Context**

The project needs (a) a robust, low-maintenance implementation of the standard building
blocks of Bayesian optimization (search space definition, trial bookkeeping, a
no-manual-tuning default strategy, sensitivity analysis, persistence), and (b) a genuinely
custom, objective-independent exploration/exploitation control (FR5) that is not exposed by
any existing high-level BO library.

**Decision**

Use Ax's `Client` API for everything in (a), and a custom BoTorch-based acquisition layer for
(b), connected through a fitted-model handoff rather than through Ax's generation-strategy
plugin mechanism.

**Options considered**

| Dimension | Ax only | BoTorch only | Hybrid (chosen) |
|---|---|---|---|
| Effort for FR1-FR4, FR6, FR7, FR9-FR11 | Low (native) | High (all hand-rolled) | Low (native) |
| Effort for FR5 (alpha dial) | High (fighting the framework) | Low (native building blocks) | Low (native building blocks) |
| Long-term maintenance | Low | High | Medium |
| Risk from upstream API churn | Medium (already changed once) | Low (BoTorch is lower-level, more stable) | Medium, mitigated by the domain-layer boundary (design philosophy #2) |

**Trade-off analysis**

Ax-only would mean either not delivering FR5 properly, or fighting Ax's generation-strategy
internals to inject a custom acquisition function, which is possible but brittle and poorly
documented for this specific use case. BoTorch-only would mean re-implementing trial
bookkeeping, persistence, and sensitivity analysis that Ax already provides solidly. The
hybrid keeps each library doing what it is strongest at.

**Consequences**

- `backends/ax` is the only module allowed to import from `ax.*`.
- `acquisition` is the only module allowed to build custom `botorch.acquisition.*` subclasses.
- Any future backend swap (e.g. moving off Ax entirely) only requires rewriting
  `backends/ax` and the parts of `acquisition` that read a fitted model, not `domain` or `api`.

**Action items**

1. [ ] Implement `backends/ax/ax_backend.py` against the pinned Ax version.
2. [ ] Implement `acquisition/exploration_exploitation_acquisition.py` against the pinned
   BoTorch version.
3. [ ] Pin compatible Ax/BoTorch/PyTorch/GPyTorch versions together in `pyproject.toml`.

#### ADR-0002: `uv` for project and dependency management

**Status:** Accepted
**Date:** 2026-09-23
**Deciders:** [project owner]

**Context**

The project has a heavy, sometimes fragile dependency tree (PyTorch, BoTorch, Ax, GPyTorch),
several of which have platform-specific wheels. FR12 explicitly requires reproducibility. The
project owner's default on other projects is `setuptools`.

**Decision**

Use `uv` for environment creation, dependency resolution, locking, and running dev commands
(`uv run`, `uv sync`). Use a simple build backend (`uv_build`, or `hatchling` if a Ax/BoTorch
edge case needs it) rather than `setuptools`, since this is a pure-Python package with no C
extensions to compile.

**Options considered**

`uv` and `setuptools` are not strictly alternatives: `setuptools` is a *build backend* (it
knows how to turn source into a wheel/sdist); `uv` is a *project and dependency manager*
(virtual environments, dependency resolution, lockfiles, Python version management) that can
itself use `setuptools`, `hatchling`, or its own `uv_build` as the build backend underneath.
The real comparison is "pip + setuptools as the whole day-to-day workflow" vs. "`uv` as the
whole day-to-day workflow".

| Dimension | pip + setuptools workflow | `uv` workflow (chosen) |
|---|---|---|
| Install speed on a torch/botorch-sized dependency tree | Minutes | Seconds to low tens of seconds |
| Reproducible installs | Requires adding `pip-tools` or similar; no native lockfile | Native, cross-platform `uv.lock` |
| Virtual env + Python version management | Separate tools (`venv`, `pyenv`) | Built in |
| Maturity | 20+ years, universal | Newer (~2 years), but backed by Astral (same team as `ruff`) and already a common choice for ML projects |
| Handles C extensions / Cython | Yes (this is `setuptools`' particular strength) | Not needed here: `boptim` is pure Python |

**Trade-off analysis**

For a pure-Python package, `setuptools`' specific strength (compiled extensions) is not
relevant here, so it offers no advantage that matters for this project. `uv`'s advantages
(install speed, native lockfile) map directly onto two things this project actually has: a
heavy dependency tree and an explicit reproducibility requirement. This is not a marginal
call; it is a real, project-specific fit, not a general endorsement of `uv` over `setuptools`
in every context (a project that genuinely needs custom build steps or C extensions would
weigh this differently). If `setuptools` is preferred for familiarity, it can still be used as
just the build backend under `uv` (`uv init --build-backend setuptools`) without giving up any
of `uv`'s other benefits; that remains a low-cost fallback.

**Consequences**

- Contributors run `uv sync` once instead of manually managing a virtualenv.
- `uv.lock` is committed to version control.
- CI uses `uv`'s official GitHub Action instead of a manual `pip install` step.

**Action items**

1. [ ] `uv init`, confirm the pinned dependency set resolves cleanly.
2. [ ] Commit `uv.lock`.
3. [ ] Update the CI workflow to use `astral-sh/setup-uv`.

#### ADR-0003 (to write): Pydantic domain models decoupled from Ax's internal types

Context to seed it: design philosophy #2 (section 1.3) and the fact that Ax's own public API
surface has already changed once. Follow the ADR-0001 format above.

#### ADR-0004 (to write): JSON as the persistence format, with embedded reproducibility metadata

Context to seed it: FR11/FR12, and that Ax already ships JSON (de)serialization for its own
experiment objects, which `persistence/json_study_repository.py` can wrap rather than
reinvent. Follow the ADR-0001 format above.

---

## 5. Detailed design

This section names the main files, classes, methods and functions with typed signatures.
Docstrings are shortened to one line here for space; the actual code requires full
Google-style docstrings per section 10. Signatures are Python 3.11+ (`X | None`,
built-in generics).

### 5.1 Package layout

```
boptim/
├── pyproject.toml
├── README.md
├── CHANGELOG.md
├── mkdocs.yml
├── docs/
│   ├── index.md
│   ├── contributing.md              # incl. the naming-convention rationale from section 10
│   ├── changelogs
│   │   ├── changelog-v0.2.3.md
│   │   ├── changelog-v1.0.0.md
│   │   └── CHANGELOG.md
│   └── adr/
│       ├── 0001-hybrid-ax-botorch-architecture.md
│       ├── 0002-uv-over-setuptools.md
│       ├── 0003-pydantic-domain-models.md
│       └── 0004-json-persistence-format.md
├── src/boptim/
│   ├── __init__.py                  # re-exports the public surface (BayesianOptimizer, parameter
│   │                                 # classes, Objective, Metric, constraints, ...)
│   ├── domain/
│   │   ├── parameters/
│   │   │   ├── parameter.py             # Parameter (ABC)
│   │   │   ├── real.py                  # Real
│   │   │   ├── integer.py               # Integer
│   │   │   ├── categorical.py           # Categorical
│   │   │   ├── boolean.py               # Boolean
│   │   │   └── condition.py             # Condition (drives active_when, section 5.2)
│   │   ├── constraints/
│   │   │   ├── constraint.py            # Constraint (ABC)
│   │   │   └── linear_constraint.py     # LinearConstraint
│   │   ├── search_space.py              # SearchSpace
│   │   ├── metric.py                    # Metric
│   │   ├── objective.py                 # Objective (handles N >= 1 uniformly)
│   │   ├── trial.py                     # Trial
│   │   └── study_snapshot.py            # StudySnapshot (full persisted state)
│   ├── backends/
│   │   ├── optimization_backend.py      # OptimizationBackend (ABC)
│   │   └── ax/
│   │       ├── ax_backend.py             # AxBackend
│   │       ├── search_space_mapper.py    # toAxSearchSpace / fromAxSearchSpace
│   │       └── objective_mapper.py       # toAxOptimizationConfig
│   ├── models/
│   │   └── surrogate_model_factory.py    # buildSurrogateModel(...)
│   ├── acquisition/
│   │   ├── acquisition_strategy.py                  # AcquisitionStrategy (ABC)
│   │   ├── exploration_exploitation_acquisition.py  # ExplorationExploitationAcquisition
│   │   └── multi_objective_acquisition.py           # MultiObjectiveExplorationExploitationAcquisition
│   ├── analysis/
│   │   ├── sensitivity_analyzer.py          # SensitivityAnalyzer (ABC)
│   │   ├── sobol_sensitivity_analyzer.py    # SobolSensitivityAnalyzer
│   │   └── prediction_result.py             # PredictionResult
│   ├── persistence/
│   │   ├── study_repository.py           # StudyRepository (ABC)
│   │   ├── json_study_repository.py      # JsonStudyRepository
│   │   └── reproducibility_metadata.py   # ReproducibilityMetadata
│   ├── api/
│   │   └── bayesian_optimizer.py          # BayesianOptimizer (the public facade)
│   └── logging_config.py                 # configureLogging(...)
├── scripts/
├── exemples/
└── tests/
    ├── unit/                             # mirrors src/boptim/*
    └── integration/
        └── test_end_to_end.py            # dummy objective, full tell/ask/save/load loop
```

### 5.2 Domain layer

```python
# domain/parameters/parameter.py
class Parameter(ABC):
    """Base class for every kind of search space parameter."""

    name: str
    default: float | int | str | bool | None
    active_when: Condition | None
    """None means always active. Set to make this parameter only relevant
    when another parameter takes specific value(s): a conditional /
    hierarchical search space, e.g. "num_filters" only matters when
    "model_type" == "cnn". See section 4.5 for how far this is actually
    supported end to end."""


# domain/parameters/condition.py
class Condition:
    def __init__(self, parameter_name: str, values: list[float | int | str | bool]) -> None:
        """True when `parameter_name`'s current value is one of `values`.

        Attach to another Parameter's `active_when` to make it conditional.
        Named after, and directly inspired by, Keras Tuner's
        parent_name/parent_values pattern, kept declarative (a plain value
        object) rather than a callback so the search space stays fully
        introspectable for persistence and for a future GUI.
        """


# domain/parameters/real.py
class Real(Parameter):
    def __init__(
        self,
        name: str,
        lower_bound: float,
        upper_bound: float,
        log_scale: bool = False,
        step: float | None = None,
        default: float | None = None,
        active_when: Condition | None = None,
    ) -> None:
        """A continuous parameter with a lower and upper bound.

        step, if set, restricts sampling to a fixed grid within the bounds
        (e.g. a dial that only turns in 5-degree increments).
        """


# domain/parameters/integer.py
class Integer(Parameter):
    def __init__(
        self,
        name: str,
        lower_bound: int,
        upper_bound: int,
        log_scale: bool = False,
        step: int | None = None,
        default: int | None = None,
        active_when: Condition | None = None,
    ) -> None:
        """An integer parameter with a lower and upper bound."""


# domain/parameters/categorical.py
class Categorical(Parameter):
    def __init__(
        self,
        name: str,
        categories: list[str],
        default: str | None = None,
        active_when: Condition | None = None,
    ) -> None:
        """A parameter taking one of a fixed, unordered set of values."""


# domain/parameters/boolean.py
class Boolean(Parameter):
    def __init__(
        self,
        name: str,
        default: bool | None = None,
        active_when: Condition | None = None,
    ) -> None:
        """A True/False parameter."""


# domain/constraints/constraint.py
class Constraint(ABC):
    """Base class for constraints across one or more parameters."""


# domain/constraints/linear_constraint.py
class LinearConstraint(Constraint):
    def __init__(
        self,
        coefficients: dict[str, float],
        bound: float,
        comparator: Literal["<=", ">=", "="],
    ) -> None:
        """sum(coefficients[name] * value[name]) <comparator> bound.

        Example: a 3-component mixture summing to 1 is
        LinearConstraint({"a": 1.0, "b": 1.0, "c": 1.0}, bound=1.0, comparator="=").
        """


# domain/search_space.py
class SearchSpace:
    def __init__(
        self,
        parameters: Sequence[Parameter],
        constraints: Sequence[Constraint] | None = None,
    ) -> None: ...

    def addParameter(self, parameter: Parameter) -> None: ...

    def addConstraint(self, constraint: Constraint) -> None: ...

    @property
    def parameter_names(self) -> list[str]:
        """Cheap (list comprehension over an already-held list): snake_case."""


# domain/metric.py
class Metric:
    def __init__(self, name: str, minimize: bool) -> None:
        """One measured quantity and its optimization direction."""


# domain/objective.py
class Objective:
    def __init__(
        self,
        metrics: Sequence[Metric],
        weights: Sequence[float] | None = None,
    ) -> None:
        """One or more metrics with optional relative importance weights.

        `weights=None` with a single metric is the plain single-objective case.
        `weights=None` with several metrics means no preference between them
        (a standard Pareto multi-objective problem). Explicit weights scalarize
        the preference between objectives; they do not control exploration
        versus exploitation, which is `BayesianOptimizer.ask`'s `alpha` (section 4.3).
        """

    @property
    def is_multi_objective(self) -> bool:
        """Cheap (len check): snake_case."""


# domain/trial.py
class Trial:
    def __init__(
        self,
        parameters: dict[str, float | int | str | bool],
        results: dict[str, float],
        result_std: dict[str, float] | None = None,
        trial_index: int | None = None,
    ) -> None:
        """One evaluated point: its parameters and its observed metric value(s).

        `result_std`, when known (e.g. from repeated measurements), is passed
        through as a fixed observation noise instead of being inferred by the
        surrogate model.
        """


# domain/study_snapshot.py
class StudySnapshot:
    """Everything needed to fully reconstruct a BayesianOptimizer: its search space,
    objective, constraints, trial history, and reproducibility metadata.
    The unit that persistence/ reads and writes."""
```

### 5.3 Backend layer

```python
# backends/optimization_backend.py
class OptimizationBackend(ABC):
    @abstractmethod
    def createExperiment(self, search_space: SearchSpace, objective: Objective) -> None: ...

    @abstractmethod
    def attachTrial(self, trial: Trial) -> int:
        """Registers an already-evaluated trial, returns its backend-assigned index."""

    @abstractmethod
    def suggestDefault(self, n_points: int) -> list[dict[str, float | int | str | bool]]:
        """The no-manual-tuning default path (FR4), delegated entirely to the backend."""

    @abstractmethod
    def fitModel(self) -> Model:
        """Fits and returns the current surrogate model, for the acquisition layer to use."""

    @abstractmethod
    def predict(
        self, parameters: dict[str, float | int | str | bool]
    ) -> dict[str, tuple[float, float]]:
        """Returns {metric_name: (mean, variance)}."""

    @abstractmethod
    def computeSensitivity(self) -> dict[str, dict[str, float]]:
        """Returns {metric_name: {parameter_name: importance}}."""


# backends/ax/ax_backend.py
class AxBackend(OptimizationBackend):
    def __init__(self, random_seed: int | None = None) -> None: ...
    # implements every method above by delegating to an ax.api.client.Client instance
    # held as a private attribute; exact accessor names (e.g. for sensitivity analysis)
    # to be confirmed against the Ax version pinned in pyproject.toml.


# backends/ax/search_space_mapper.py
def toAxSearchSpace(search_space: SearchSpace) -> list[Any]:
    """Maps boptim Parameters/Constraints to Ax RangeParameterConfig /
    ChoiceParameterConfig / parameter_constraints. Return type is Ax's own
    config list type; kept as Any here to avoid leaking an Ax import into
    this file's public signature."""


# backends/ax/objective_mapper.py
def toAxOptimizationConfig(objective: Objective) -> str | list[str]:
    """Maps a boptim Objective to the objective expression(s) Ax's
    configure_optimization expects, weighted or unweighted as appropriate."""
```

### 5.4 Surrogate model and acquisition layer

```python
# models/surrogate_model_factory.py
def buildSurrogateModel(
    train_x: Tensor,
    train_y: Tensor,
    train_yvar: Tensor | None = None,
    minimum_points_for_free_fit: int = 5,
) -> Model:
    """Builds a SingleTaskGP (or a multi-output model for multi-objective).

    Infers observation noise when train_yvar is None. Below
    minimum_points_for_free_fit observations, falls back to weakly informative
    priors instead of a free marginal-likelihood fit (section 4.4). The single
    swap point for a scalable/sparse GP in a future version.
    """


# acquisition/acquisition_strategy.py
class AcquisitionStrategy(ABC):
    @abstractmethod
    def buildAcquisitionFunction(
        self,
        model: Model,
        objective: Objective,
        alpha: float,
        reference_points: Tensor,
    ) -> AcquisitionFunction: ...

    @abstractmethod
    def suggest(
        self,
        model: Model,
        objective: Objective,
        search_space: SearchSpace,
        alpha: float,
        n_points: int,
    ) -> list[dict[str, float | int | str | bool]]: ...


# acquisition/exploration_exploitation_acquisition.py
class ExplorationExploitationAcquisition(MCAcquisitionFunction):
    """Blends a posterior-mean exploitation term and a posterior-variance
    exploration term: score(x) = (1 - alpha) * exploitation(x) + alpha *
    exploration(x), both min-max normalized over reference_points. See
    section 4.3."""

    def __init__(
        self,
        model: Model,
        alpha: float,
        minimize: bool,
        reference_points: Tensor,
        sampler: MCSampler | None = None,
    ) -> None: ...

    def forward(self, X: Tensor) -> Tensor: ...


# acquisition/multi_objective_acquisition.py
class MultiObjectiveExplorationExploitationAcquisition(MCAcquisitionFunction):
    """Multi-objective generalization: exploitation is a (weighted)
    hypervolume-improvement term over the current Pareto front, exploration
    is an aggregate of per-objective posterior variance."""

    def __init__(
        self,
        model: Model,
        alpha: float,
        objective_weights: Tensor | None,
        ref_point: Tensor,
        reference_points: Tensor,
        sampler: MCSampler | None = None,
    ) -> None: ...

    def forward(self, X: Tensor) -> Tensor: ...
```

### 5.5 Analysis layer

```python
# analysis/sensitivity_analyzer.py
class SensitivityAnalyzer(ABC):
    @abstractmethod
    def computeSensitivity(
        self, model: Model, search_space: SearchSpace, metric_names: Sequence[str]
    ) -> dict[str, dict[str, float]]: ...


# analysis/sobol_sensitivity_analyzer.py
class SobolSensitivityAnalyzer(SensitivityAnalyzer):
    def __init__(self, num_mc_samples: int = 1024) -> None: ...
    def computeSensitivity(
        self, model: Model, search_space: SearchSpace, metric_names: Sequence[str]
    ) -> dict[str, dict[str, float]]:
        """Fallback used when not delegating to Ax's built-in analysis, so the
        acquisition-only code path (no AxBackend involved) still has parameter
        importance available."""


# analysis/prediction_result.py
@dataclass(frozen=True)
class PredictionResult:
    mean: dict[str, float]
    variance: dict[str, float]

    @property
    def std(self) -> dict[str, float]:
        """Square root of variance over a handful of metrics: negligible cost,
        snake_case even though it "computes" something, per section 10."""
```

### 5.6 Persistence layer

```python
# persistence/study_repository.py
class StudyRepository(ABC):
    @abstractmethod
    def save(self, snapshot: StudySnapshot, path: str | Path) -> None: ...

    @abstractmethod
    def load(self, path: str | Path) -> StudySnapshot: ...


# persistence/json_study_repository.py
class JsonStudyRepository(StudyRepository):
    def save(self, snapshot: StudySnapshot, path: str | Path) -> None: ...
    def load(self, path: str | Path) -> StudySnapshot: ...


# persistence/reproducibility_metadata.py
@dataclass(frozen=True)
class ReproducibilityMetadata:
    random_seed: int
    library_versions: dict[str, str]
    created_at: datetime
    boptim_version: str
```

### 5.7 Public API facade

`ask`/`tell` is the naming used by scikit-optimize and Optuna for exactly this pattern, so it
was adopted here over the earlier `suggest`/`observe` names for the sake of matching an
established convention that both target audiences (ML practitioners and, increasingly, the
BO-literate lab) are likely to already recognize. The constructor accepts either a plain list
of `Parameter`s (the common case) or a fully-built `SearchSpace`/`Objective` pair (the case
that needs constraints or multi-objective weighting spelled out), so simple use stays terse
without losing access to the advanced path.

```python
# api/bayesian_optimizer.py
class BayesianOptimizer:
    def __init__(
        self,
        parameters: Sequence[Parameter] | SearchSpace,
        objective: Objective | Literal["minimize", "maximize"] = "minimize",
        constraints: Sequence[Constraint] | None = None,
        name: str = "study",
        random_seed: int | None = None,
        backend: OptimizationBackend | None = None,
    ) -> None:
        """The main entry point. Two ways to call it:

        Common case: BayesianOptimizer(parameters=[Real(...), Integer(...)]).
            A single, unnamed metric is assumed; objective="minimize" or
            "maximize" picks its direction.
        Advanced case: pass an already-built SearchSpace (carrying its own
            constraints) and an already-built Objective (named metrics,
            optional weights, single or multi-objective). `constraints` must
            be left None in that case (a ValueError is raised otherwise: the
            SearchSpace already owns its own constraints, so passing both is
            an ambiguous request, not a merge).

        backend defaults to AxBackend(random_seed=random_seed); injectable for
        testing and for a future non-Ax backend. The real implementation will
        likely type this with @overload for the two cases rather than the
        single Union shown here, kept simple for this spec.
        """

    def tell(
        self,
        x: dict[str, float | int | str | bool],
        y: dict[str, float],
        y_std: dict[str, float] | None = None,
    ) -> Trial:
        """Manually inject an already-known point and its result(s). FR3.

        x maps parameter name to value. y maps metric name to value, matching
        however Objective's metrics were named (or the single default metric
        name when the shorthand constructor was used). y_std, when known
        (e.g. from repeated measurements), is passed through as a fixed
        observation noise instead of being inferred by the surrogate model.
        """

    def ask(
        self,
        n_points: int = 1,
        alpha: float | None = None,
    ) -> list[dict[str, float | int | str | bool]]:
        """Ask for the next n_points parameterizations to evaluate.

        Args:
            n_points: batch size requested at once. FR10.
            alpha: exploration/exploitation trade-off in [0, 1]. 0.0 favors the
                best predicted objective, 1.0 favors the least-known region,
                0.5 balances both. FR5. If None, delegates to the backend's
                own no-manual-tuning default strategy (FR4) instead of the
                custom acquisition layer.
        """

    def predict(self, x: dict[str, float | int | str | bool]) -> PredictionResult:
        """FR7, FR8."""

    def parameterImportance(self) -> dict[str, dict[str, float]]:
        """Refits/queries the surrogate model: has a cost, camelCase. FR6."""

    @property
    def n_trials(self) -> int:
        """len() over an already-held list: cheap, snake_case."""

    @property
    def paretoFront(self) -> list[Trial]:
        """Recomputed by scanning all trials on each access: has a cost,
        camelCase. Collapses to a single-element list for a single-objective
        optimizer; `bo.paretoFront[0]` is then the best trial found so far."""

    def save(self, path: str | Path) -> None:
        """FR11, FR12."""

    @classmethod
    def load(cls, path: str | Path) -> BayesianOptimizer:
        """FR11, FR12."""
```

### 5.8 Usage example

The common case, start to finish. Parameter and method names are chosen so this reads close
to plain English, per the "clean but simple" ask that shaped this section.

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

# ask for a single point, leaning toward exploitation
x = bo.ask(alpha=0.2)[0]
y = run_my_experiment(**x)          # your own code: a real experiment or a training run
bo.tell(x, {"objective": y})

# ask for a batch of 3, balanced exploration/exploitation (FR10)
batch = bo.ask(n_points=3, alpha=0.5)

# what does the model currently believe, with uncertainty? (FR7, FR8)
prediction = bo.predict({"temperature": 100.0, "num_layers": 4, "solvent": "ethanol", "use_catalyst": False})
print(prediction.mean, prediction.std)

bo.parameterImportance()            # FR6
bo.save("study.json")               # FR11, FR12
bo_reloaded = BayesianOptimizer.load("study.json")
```

A conditional search space (section 4.5), kept out of the main example above to keep that one
short:

```python
model_type = Categorical("model_type", ["mlp", "cnn"])
bo = BayesianOptimizer(parameters=[
    model_type,
    Integer("hidden_units", 8, 512, active_when=Condition("model_type", ["mlp"])),
    Integer("num_filters", 8, 256, active_when=Condition("model_type", ["cnn"])),
])
```

---

## 6. Roadmap

The phases are sequenced by technical risk, not by feature importance: every functional
requirement from section 2 is in scope, but the genuinely novel part (the alpha dial) is
built once the foundation under it is solid, rather than the other way around.

**Phase 1: Foundation (domain model + Ax-backed default flow)**
Full domain model including constraints, multi-objective (`Objective` with N >= 1 metrics and
optional weights), and `default`/`step`/`active_when` on parameters (FR13, and FR14's
definition/persistence/introspection side) from the start, since Ax's `Client` natively
supports multi-objective, constraints, batch, and (per section 4.5) a hierarchical search
space, so this is realistically a mapping-layer effort, not a re-design later. `AxBackend`,
manual injection, `ask()` via Ax's own default strategy, `predict()`, Ax's built-in sensitivity
analysis, JSON persistence and reproducibility metadata. Deliverable: a usable, if not yet
alpha-controllable, end-to-end loop.

**Phase 2: The exploration/exploitation acquisition layer**
`ExplorationExploitationAcquisition` and its multi-objective variant, with batch support
(fantasization-based sequential selection) built in from the start rather than retrofitted.
This is FR5, the requirement Ax does not cover, and the reason this project exists rather
than "just use Ax".

**Phase 3: Robustness across the full budget range, and deferred hard problems**
Weakly-informative-prior fallback for very small trial counts (down to the 4-5-evaluation
case), explicit low-confidence signaling; the scalable/sparse-GP extension point in
`models/surrogate_model_factory.py` documented (not necessarily implemented) for very large
trial counts; and FR14's harder half, making the custom acquisition layer actually reason
about `active_when` instead of just not breaking on it, documented as a known limitation
rather than implemented, in step with where Ax's own support for this sits today (section
4.5).

**Phase 4: Polish, docs, packaging**
Full `mkdocs` site, ADR-0003 and ADR-0004 written, CI green on lint/type-check/tests,
`uv`-based packaging finalized, integration test covering the full
tell/ask/save/load loop on a dummy objective, and two worked examples in the docs: one
framed as a physical-experiment campaign, one as an ML hyperparameter search, to make the
"general-purpose, not lab-specific" intent (section 1.1) concrete rather than just stated.

---

## 10. Coding standards

*(Section numbers 7-9 and 11 are intentionally left open, consistent with the numbering style
of the source template this document extends.)*

- **Language/runtime:** Python 3.11+
- **Formatting/linting:** `ruff` (lint + format), consistent import ordering.
- **Naming convention (custom, overrides PEP8 default for callables):**
  - Classes -> `CamelCase` (e.g. `SearchSpace`, `Real`, `ExplorationExploitationAcquisition`).
  - Variables (including function/method parameters) -> `snake_case` (e.g. `n_points`, `random_seed`).
  - Functions and methods -> same rule as classes but starting lowercase, i.e. `camelCase`
    (e.g. `suggestDefault`, `computeSensitivity`, `buildAcquisitionFunction`), not PEP8's usual
    `snake_case` for callables. Python's required dunder methods (`__init__`, `__repr__`, ...)
    are exempt: they keep their mandatory spelling.
  - Properties (`@property`) are named by **cost, not by whether they compute anything**:
    - If accessing the property does real work (recomputes something from the full trial
      history, calls into a model, is not O(1)-ish) -> `camelCase`, exactly like a method,
      even though it is called without parentheses. Example: `BayesianOptimizer.paretoFront`.
    - If it just returns an already-stored or negligible-cost value -> `snake_case`, like a
      variable. Example: `BayesianOptimizer.n_trials`, `PredictionResult.std`.
    - The point is that the caller can tell, from the name alone, whether touching this
      attribute is free or not.
  - Since this deviates from PEP8, disable/adjust `ruff`'s naming rules (`N802`, `N803`,
    `N806`) in `pyproject.toml` and note the exception in `docs/contributing.md`, so linting
    doesn't silently "fix" it back to snake_case later.
- **Typography:** no em dashes (`—`) in code, comments, docstrings, commit messages, or project
  documentation. Use a period, a colon, parentheses, or two sentences instead. This is a house
  style rule, not a technical one, so there is no linter for it; review for it like any other
  style note. Same for arrows (`→`), use `->` instead.
- **Typing:** type hints mandatory everywhere; `mypy --strict` run in CI.
- **Docstrings:** Google-style, mandatory on every public class/function: purpose, `Args`,
  `Returns`, `Raises`.
- **Modularity:** one responsibility per file; one class per file. No god-files: a base class,
  its registry, and every concrete strategy each get their own file (see section 5.1's package
  layout for what this looks like in practice: `parameters/`, `constraints/`, `acquisition/`
  each split this way).
- **Testing:** `pytest`. Unit tests per module, plus an integration test that runs the full
  tell/ask/save/load loop end to end on a dummy objective (shape and gradient sanity
  checks where tensors are involved).
- **Logging:** standard `logging` module, no bare `print`.
- **Version control:** Conventional Commits, semantic versioning, maintained `CHANGELOG.md`.
  Architectural decisions get a new ADR when they change, rather than an old ADR being edited
  in place (see `docs/adr/0002-*.md` for an example of a decision recorded this way).
- **CI:** GitHub Actions running lint, type-check, and tests on every push.
- **Documentation:** `mkdocs` + `mkdocstrings` built from docstrings; major architectural
  choices (e.g. "why a hybrid Ax + BoTorch architecture", "why Pydantic domain models
  decoupled from Ax's own types") recorded as short ADRs (`docs/adr/NNNN-title.md`).
- **Flexibility:** the code should be as flexible and generalist as possible so it can be
  adapted to any user, lab or otherwise. Avoid hard-coded values that may be significant to a
  particular user; define sensible defaults instead (see section 1.4 for the defaults chosen
  in this draft, and section 4.4 for how this plays out for very small vs. very large trial
  budgets).

---

## 12. How future Claude conversations (and contributors) should use this document

- Treat sections 0, 2, 3, 5 and 10 as the contract: search-space types, functional
  requirements, the technology choices and their rationale (section 4.6's ADRs), the package
  layout, and the naming/style rules. Do not silently deviate from them; propose a change and,
  if accepted, update the relevant ADR or section rather than drifting from it in code.
- Section 1.4 ("Defaults chosen for this draft") lists the handful of decisions made to keep
  this document concrete rather than because they were architecturally required. These are
  safe to revisit without re-opening the rest of the design.
- Section 5's signatures are the target shape of the code, not yet the code. When
  implementing, keep the signatures; if a signature turns out to be wrong once real code is
  written against it, update this document in the same change, not after.
- If a request does not have a clear answer in this document (a new feature, a changed
  requirement, an ambiguous naming case not covered by section 10), do not guess: ask, the way
  this document itself was produced through a round of clarifying questions before being
  written. Once answered, fold the answer back into this document so the next conversation
  does not have to ask again.