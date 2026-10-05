# boptim: Project Specification

**A general-purpose, domain-agnostic Bayesian Optimization library for Python.**

- Version: 0.2 (draft)
- Date: 2026-09-23
- Status: Draft, pending review by the project owner

> Scope note: this document fixes the technology stack, the architecture, the functional
> specification, the roadmap, and the signatures of the main files, classes, methods and
> functions. It deliberately stops at signatures and short docstrings, not full
> implementations: the goal is to agree on the contract before writing code. `boptim` is a
> working name for the package; check PyPI availability before publishing and rename freely.
> This revision grounds the parameter model directly in Ax's own current API reference
> (https://ax.readthedocs.io/en/stable/api.html), rather than in earlier, partly-inferred
> assumptions about it; where the two conflict, this revision wins.

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
**general-purpose, domain-agnostic** tool. It is a generic black-box optimization engine that
is used, among other things, by a laboratory and by machine-learning practitioners.
Concretely, this means the library must be equally comfortable with:

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
- Not a reimplementation of what Ax already does well. Where Ax's own API
  (`ax.api.configs`, `ax.api.client.Client`) already has a clean shape for something,
  `boptim`'s domain layer mirrors that shape and its backend delegates to it, rather than
  inventing a parallel abstraction. Section 4.5 and 4.6 are about exactly this.

### 1.3 Design philosophy

1. **Flexibility over convention.** Every numeric default (initial design size, noise
   assumption, exploration/exploitation blend, batch size, etc.) is a *default*, never a
   hardcoded assumption. Nothing in the domain layer encodes "this is for a lab" or "this is
   for ML".
2. **`boptim`'s public API is its own, but it is not a cage.** Ax and BoTorch are
   implementation details behind an adapter for the common case, not something a caller has
   to understand to get started. But a high-level facade will always fall short of some
   power-user need, so `BayesianOptimizer` also exposes the underlying `ax.api.client.Client`
   and the current fitted BoTorch model directly (`axClient`, `fitModel()`, section 5.7,
   ADR-0005). The point of the facade is a clean default path, not a locked door.
3. **Single-objective is the N = 1 case of multi-objective**, not a separate code path. There
   is one `Objective` concept everywhere in the codebase.
4. **The exploration/exploitation dial is orthogonal to everything else.** It is not tied to
   a specific number of objectives, a specific noise model, or a specific budget size.
5. **Don't recode what Ax already codes.** Search space definition, conditional/hierarchical
   parameters, sensitivity analysis, Pareto-frontier and best-trial lookup, and JSON
   persistence all have a native Ax primitive; `boptim`'s job there is a thin, typed,
   boptim-flavored pass-through, not a reimplementation. FR5 (the alpha dial) is the one
   genuine exception, and the whole reason this project is more than a thin Ax wrapper.

### 1.4 Defaults chosen for this draft (flag for review)

| Item | Default chosen | Why |
|---|---|---|
| Package name | `boptim` | Placeholder. Short, descriptive, likely available; verify on PyPI. |
| `mypy` strictness | `--strict` | Matches "type hints mandatory everywhere" from section 10. |
| License | Not set (TBD) | Depends on whether this stays internal or gets published; not an architectural decision. |
| Build backend | `uv_build` (or `hatchling`) under `uv` | See ADR-0002 in section 4.8. |

---

## 2. Functional requirements

| ID | Requirement | Notes |
|---|---|---|
| FR1 | Define a search space with `float`, `int`, categorical and boolean parameters, with bounds and linear/log scales, plus fixed and derived (computed) parameters. | See section 5.2: `Range`/`Real`/`Integer`, `Choice`/`Categorical`/`Boolean`, `Fixed`, `Derived`. |
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
| FR14 | Support conditional (hierarchical) parameters: a parameter that only applies when another parameter takes a specific value. | Directly exposes Ax's own `dependent_parameters` mechanism on `Choice` (section 4.5, 5.2); not a boptim-invented mechanism. Definition/persistence is solid; genuine optimization-quality benefit from the structure is weaker, tracked separately. |
| FR15 | Support constraints on an observed metric, not just on parameters (e.g. "the result must reach at least X"). | Mirrors Ax's own `outcome_constraints`. See `OutcomeConstraint`, section 5.2. |
| FR16 | Give a documented way to use Ax and BoTorch directly for anything the high-level facade does not cover, without forking the library. | `BayesianOptimizer.axClient` / `.fitModel()`. See design philosophy #2, ADR-0005. |
| FR17 | Support a feasibility constraint across parameters expressed as an arbitrary (nonlinear) expression, e.g. `var * x ** z <= n`. | Not expressible through Ax's own (linear-only) `parameter_constraints`; enforced by `boptim`'s own acquisition layer via BoTorch's `nonlinear_inequality_constraints` instead. See `NonlinearConstraint`, section 4.6, section 5.2, ADR-0006. |

---

## 3. Technology stack

| Layer | Technology | Why |
|---|---|---|
| Language / runtime | Python 3.11+ | Given. |
| Optimization core | Ax (`ax-platform`, the `ax.api.client.Client` API and `ax.api.configs` parameter configs) | Search space definition (including conditional parameters), trial/experiment bookkeeping, a strong default generation strategy, built-in sensitivity analysis, prediction, Pareto-frontier/best-trial lookup, and JSON persistence. Covers most of FR1-FR4, FR6, FR7, FR9, FR11, FR14, FR15 largely "for free". |
| Custom acquisition | BoTorch | Ax's automatic strategy selection is objective-driven (EI/NEI/UCB-family); it does not expose a pure-exploration, objective-independent acquisition function. BoTorch does (`qNegIntegratedPosteriorVariance` and friends), so FR5 is built as a custom acquisition layer on top of BoTorch models, orchestrated around Ax's data. See ADR-0001. |
| Tensor / GP backend | PyTorch, GPyTorch | Transitive dependencies of Ax/BoTorch, not called directly except where the custom acquisition layer needs raw posterior access. |
| Domain / config models | Pydantic v2 | Typed, validated, JSON-serializable-by-construction models for `boptim`'s own public types (parameters, objectives, constraints, trials), shaped to mirror `ax.api.configs` field-for-field where one exists. Keeps the public API decoupled from Ax's internal types (design philosophy #2). |
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
  |                    default generation strategy, sensitivity, predictions,   --+
  |                    Pareto frontier, JSON save/load                          |
  |                                                                              +-- Ax / BoTorch /
  +-- acquisition/     custom BoTorch acquisition layer: the alpha dial,        |   GPyTorch
  |                    batch generation, multi-objective variant                |
  |                                                                              |
  +-- analysis/        sensitivity + prediction wrappers (Ax-backed, with      --+
  |                    a non-Ax fallback for the BoTorch-only code path)
  |
  +-- persistence/     wraps Ax's own JSON save/load, adds reproducibility
                       metadata boptim's domain layer needs on top
```

- **`domain`** has zero ML dependencies. It is what makes the library's public surface stable
  even if the backend changes, and mirrors `ax.api.configs`' own parameter shapes field for
  field wherever one exists (design philosophy #5).
- **`backends/ax`** is the only place that imports Ax. It translates `domain` objects to and
  from Ax's `Client`, and runs the *default* (non-alpha-controlled) generation strategy,
  predictions, sensitivity analysis, Pareto-frontier/best-trial lookup, and persistence.
- **`acquisition`** is the only place that imports BoTorch acquisition machinery directly. It
  implements FR5 (the alpha dial) and its batch and multi-objective variants, working off the
  fitted model that the Ax backend exposes.
- **`analysis`** wraps whichever of Ax's built-in analyses we use, plus a fallback
  implementation that does not depend on Ax's internals, so the library remains usable even in
  a code path that only calls BoTorch directly.
- **`persistence`** wraps Ax's own `Client.save_to_json_file`/`load_from_json_file` for the
  Ax-backed state, and adds only what Ax's own snapshot does not carry: boptim's parameter
  defaults/dependency metadata and `ReproducibilityMetadata` (FR11, FR12).

### 4.2 Data flow for a typical call

1. Caller constructs a `BayesianOptimizer`, either from a plain list of `Parameter`s (the
   common case) or from an explicit `SearchSpace` + `Objective` (the case that needs
   constraints or multi-objective weighting spelled out in full). See section 5.7.
2. Caller calls `bo.tell(x, y)` zero or more times to inject prior data (FR3).
3. Caller calls `bo.ask(n_points=..., alpha=...)`.
   - If `alpha` is left at its default (see 4.3), `BayesianOptimizer` delegates to the Ax
     backend's default generation strategy.
   - Otherwise, `BayesianOptimizer` fits a surrogate model on its own trial history (in the
     unit-cube encoding of `models/SearchSpaceEncoder`, ADR-0007), then hands that model to
     the `acquisition` layer, which builds and optimizes the custom alpha-blended acquisition
     function to produce `n_points` candidates. With fewer than two completed trials there is
     nothing to fit, so space-filling points are suggested instead.
   - The same path is taken with `alpha` left at its default when the search space holds a
     `NonlinearConstraint` (section 4.6, ADR-0006).
4. Caller evaluates the suggested point(s) (outside `boptim`, this is where a real experiment
   or a training run happens) and calls `bo.tell(x, y)` again with the result(s).
5. At any point, caller can call `bo.predict(x)`, `bo.parameterImportance()`,
   `bo.paretoFront`, `bo.save(path)`, or drop to `bo.axClient`/`bo.fitModel()` for anything not
   covered above (FR16).

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
- **Exploration term:** the posterior variance of the underlying function at `x` (the
  epistemic uncertainty, without observation noise: the same quantity
  `qNegIntegratedPosteriorVariance` integrates over the search space), normalized the same
  way.

The blended score is `score(x) = (1 - alpha) * exploitation(x) + alpha * exploration(x)`,
maximized by BoTorch's `optimize_acqf`. For a batch of `n_points > 1`, points are chosen
sequentially: each already-chosen point is passed back to the acquisition function as pending
before the next is optimized. Pending points enter the two terms differently, because a GP
treats them differently:

- The exploration term is computed on a fantasy model conditioned on the pending points
  (BoTorch's `fantasize`). A GP's posterior variance does not depend on the observed values, so
  one fantasy suffices and the variance around a pending point collapses as if observed.
- A mean-only exploitation term cannot be diversified that way: conditioning a GP on its own
  predicted mean leaves the mean unchanged, so `alpha = 0` would return the same point
  `n_points` times (breaking FR10). With pending points the exploitation term therefore becomes
  the expected best outcome over the candidate and the pending points, estimated from joint
  posterior samples (the `qSimpleRegret` construction). With no pending point it is exactly the
  posterior mean, so a single candidate, and the first point of every batch, is still the pure
  mean optimum. The consequence is that from the second point of a batch on, `alpha = 0` means
  "best expected batch outcome", which credits uncertainty, rather than strictly "best
  predicted mean".

For multi-objective studies with no `Objective.weights`, the same blend applies, but the
exploitation term becomes the expected hypervolume improvement over the model-predicted Pareto
front, and the exploration term becomes the mean of the per-objective normalized posterior
variances. `Objective.weights` controls *preference between objectives*; `alpha` controls
*exploration vs. exploitation given that preference*. The two are orthogonal by design.

Explicit weights scalarize the objectives (exactly as `toAxOptimizationConfig` hands Ax a
weighted-sum objective), so a weighted multi-metric `Objective` is served by the
single-objective function on the weighted sum of the direction-signed metrics, not by the
hypervolume variant: scaling the axes of a hypervolume (with its reference point) leaves the
best candidate unchanged, so it could not express a preference.

### 4.4 Handling the full budget range (FR: implicit, from the "4-5 to thousands" requirement)

- The number of purely space-filling initial points before model-based suggestions start maps
  directly to Ax's own `Client.configure_generation_strategy(initialization_budget=...)`
  (part of `AxBackend`'s setup), and can be set to `0` when the caller intends to seed the
  study entirely through `tell()` and cannot afford to "spend" evaluations on a random design.
  `configure_generation_strategy`'s other native knobs
  (`method: Literal["quality", "fast", "random_search"]`, `initialize_with_center`,
  `use_existing_trials_for_initialization`, `min_observed_initialization_trials`,
  `allow_exceeding_initialization_budget`, `torch_device`) are exposed on `AxBackend`'s
  constructor rather than reinvented; see section 5.3.
- Below a configurable minimum number of observed points (a handful), a full marginal
  likelihood fit of the GP hyperparameters is not reliable; the surrogate model factory falls
  back to fixed/weakly-informative priors on lengthscales and noise rather than free
  optimization, and `BayesianOptimizer` surfaces this state so a caller building a UI on top
  can show an explicit "low-confidence model" indicator rather than silently reporting an
  overconfident uncertainty estimate.
- Scaling to very large budgets (thousands of points, where a plain GP's O(n^3) fitting cost
  becomes a real bottleneck) is out of scope for V1, but the `models/` layer is a single,
  swappable factory function specifically so that a sparse/scalable GP can be plugged in later
  without touching the acquisition or API layers. See section 6.

### 4.5 Conditional (hierarchical) parameters

Prompted by Keras Tuner's `conditional_scope`/`parent_name` pattern: a parameter that only
makes sense given another parameter's value (e.g. `num_filters` only matters when
`model_type == "cnn"`). This is not boptim-invented: Ax's own
`ax.api.configs.ChoiceParameterConfig` has a `dependent_parameters` field for exactly this,
mapping a chosen value to the names of other parameters that become part of the active search
space when that value is picked. `boptim`'s `Choice.dependent_parameters` (section 5.2) has
the same name and the same shape, and `backends/ax/toAxSearchSpace.py` passes it straight
through to Ax's `HierarchicalSearchSpace` rather than boptim maintaining its own graph of
conditions. Keeping it declarative (a field on the config object) rather than callback-based
(like Keras Tuner's `hp` object discovering the space by actually running a build function) is
what keeps the search space fully static and inspectable, which is what `persistence/` needs
to save and reload a study exactly, and what a future GUI would need to render a form that
shows or hides fields as the user picks values.

Honesty check on what this buys, kept from the previous draft because it is still true: Ax's
own team describes the current default optimization behavior over a hierarchical search space
as flattening it under the hood for model fitting rather than doing fully tree-aware
optimization, noting that this "works shockingly well" empirically but is still an active area
of their research. So: (1) the definition, persistence, and introspection of a conditional
search space is now essentially free, a straight pass-through of an Ax field, buildable in
Phase 1; (2) genuinely exploiting the structure during optimization (rather than just not
crashing on it) is weaker today than a mature, flat search space, both in Ax's default
strategy and in `boptim`'s own custom acquisition layer (section 4.3), which does not yet
reason about `dependent_parameters` at all. Treat (1) as a Phase 1 item and (2) as a distinct,
later roadmap item (section 6).

### 4.6 Nonlinear constraints and the automatic acquisition-layer switch

Two things can look like "a conditional" at first glance; this project treats them as
genuinely different features rather than stretching one mechanism to cover both.

- **Discrete, Choice-triggered activation**, i.e. Keras Tuner's own `conditional_scope`
  example: `model_type == "mlp"` activates `hidden_units`, `model_type == "cnn"` activates
  `num_filters`. This is section 4.5's `Choice.dependent_parameters`, already a direct
  pass-through of an Ax field, and it is the minimum bar this project committed to. It is
  already covered end to end; nothing below changes it or puts it at risk.
- **A feasibility constraint across parameters expressed as an arbitrary expression**, e.g.
  `var * x ** z <= n`. Here nothing stops existing: `var`, `x`, and `z` are all still active,
  searched dimensions, the constraint just rules out some combinations of their values. Ax's
  own `parameter_constraints` cannot express this: its constraint parser explicitly accepts
  linear expressions only and rejects anything else. BoTorch's `optimize_acqf` can, through
  its `nonlinear_inequality_constraints` argument, but only inside `boptim`'s own custom
  acquisition layer (section 4.3), not Ax's default generation strategy.

`NonlinearConstraint` (section 5.2) is this second thing, kept as a string expression rather
than a raw Python callable specifically so a study that uses one still round-trips through
`JsonStudyRepository` (FR11, FR12) like every other domain object; it is parsed and evaluated
by a small, restricted expression evaluator (arithmetic and a short allowlist of math
functions), never Python's own `eval`.

Its presence changes `BayesianOptimizer.ask()`'s behavior (ADR-0006): calling `ask()` with
`alpha=None` would normally delegate to Ax's no-manual-tuning default strategy (FR4), but Ax
cannot enforce a `NonlinearConstraint`, and neither can it enforce an equality
`LinearConstraint` (`comparator="="`: its `parameter_constraints` accept inequalities only,
ADR-0008). Every `Constraint` reports this through `requires_custom_acquisition_layer`. When any
constraint does, in that case `boptim` instead forces its own custom
acquisition layer with a fixed default `alpha` and logs a warning explaining why, rather than
silently returning candidates that might violate a constraint the caller declared. Calling
`ask()` with `alpha` passed explicitly already uses the custom layer, so nothing changes and no
warning fires; the switch only ever affects the implicit, default-strategy path.

### 4.7 Ax capabilities this draft knowingly does not wrap yet

Read against Ax's own API reference (https://ax.readthedocs.io/en/stable/api.html) while
revising this document. These are real, existing Ax features that are deliberately left out of
the V1 signatures in section 5, not features that were missed:

| Ax capability | What it would give `boptim` | Why deferred |
|---|---|---|
| `IRunner` / `IMetric` + `Client.run_trials()` | A closed-loop mode: hand `boptim` an objective function once, let it run the whole ask/tell/run loop internally, instead of the caller driving `ask()`/`tell()` by hand. Natural fit for the ML side of the audience. | `ask`/`tell` covers both audiences (lab and ML) uniformly; closed-loop is an additive convenience on top, not something either audience is blocked without. Candidate for a `BayesianOptimizer.optimize(objective_fn, n_trials)` helper post-V1. |
| `StorageConfig` (SQL database) | An alternative to `JsonStudyRepository` for teams that want a shared database instead of files. | `StudyRepository` is already an interface for exactly this reason (section 5.6); a DB-backed implementation is additive, not a redesign. |
| `Client.attach_baseline` + relative `OutcomeConstraint` | Expressing an outcome constraint as a multiple of a baseline/status-quo trial (e.g. "qps >= 0.95 * baseline") instead of an absolute bound. | `OutcomeConstraint.relative` (section 5.2) already reserves the field; wiring it to `attach_baseline` is a small, later addition, not a design change. |
| `Client.attach_data` (partial/intermediate data) | Recording an in-progress result before a trial fully completes, useful for early-stopping. | Not needed by the ask/tell pattern this project is built around; `tell()` assumes a trial is complete. |
| `mark_trial_failed` / `mark_trial_abandoned` / `mark_trial_early_stopped` | Fuller trial lifecycle management. | `boptim`'s trials are recorded once they are known; failure/abandonment handling matters more for the closed-loop mode above than for manual `tell()`. |
| `simplify_parameter_changes` / `pruning_target_parameterization` (BONSAI) | Minimizing how much changes between consecutive suggestions, which matters when a human has to physically reconfigure equipment between trials. | Directly relevant to the lab use case; a strong candidate for Phase 3, listed here so it is visibly not forgotten rather than silently absent. |

### 4.8 Key decisions (ADRs)

Full ADRs live under `docs/adr/`, one file per decision, following the format below. Three are
written out in full here as the decisions with the most day-to-day impact; the rest are listed
with the context that should seed them when they are written, following the same format.

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
| Effort for FR1-FR4, FR6, FR7, FR9, FR11, FR14, FR15 | Low (native) | High (all hand-rolled) | Low (native) |
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

1. [ ] Implement `backends/ax/AxBackend.py` against the pinned Ax version.
2. [ ] Implement `acquisition/ExplorationExploitationAcquisition.py` against the pinned
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
(`uv run`, `uv sync`). Use a simple build backend (`uv_build`, or `hatchling` if an Ax/BoTorch
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
heavy dependency tree and an explicit reproducibility requirement. If `setuptools` is preferred
for familiarity, it can still be used as just the build backend under `uv`
(`uv init --build-backend setuptools`) without giving up any of `uv`'s other benefits.

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

#### ADR-0004 (to write): JSON as the persistence format, wrapping Ax's own save/load

Context to seed it: FR11/FR12, and that `Client.save_to_json_file`/`load_from_json_file`
already serialize the Ax-backed state correctly; `JsonStudyRepository` (section 5.6) wraps
that rather than reinventing it, and only adds what Ax's own snapshot does not carry. Follow
the ADR-0001 format above.

#### ADR-0005: Escape hatches to Ax and BoTorch

**Status:** Accepted
**Date:** 2026-09-23
**Deciders:** [project owner]

**Context**

`BayesianOptimizer` is a deliberately high-level facade (design philosophy #2): it does not,
and should not, expose every capability of Ax or BoTorch individually. Some callers will need
something the facade does not cover (section 4.7 lists several concrete, real examples). A
facade with no way out forces those callers to either fork `boptim` or drop it entirely for a
whole project just to reach one feature underneath it.

**Decision**

`BayesianOptimizer` exposes two escape hatches directly: `axClient` (the live
`ax.api.client.Client` instance backing this optimizer, when the backend is `AxBackend`) and
`fitModel()` (the current fitted BoTorch model). Both are first-class, documented parts of the
public API, not private/internal attributes a caller has to know to reach into.

**Options considered**

| Option | Description | Trade-off |
|---|---|---|
| No escape hatch | Only what `BayesianOptimizer` explicitly wraps is reachable | Simplest surface, but strands any caller who needs one more thing Ax or BoTorch already has |
| Escape hatch via private attribute | e.g. `bo._backend._client`, undocumented | Technically possible in Python, but an undocumented private attribute is not a supported contract; it can change without notice |
| Escape hatch as a public, documented property (chosen) | `bo.axClient`, `bo.fitModel()` | A little more public surface to keep stable, in exchange for never trapping a caller behind the facade |

**Trade-off analysis**

The cost is that `axClient`'s and `fitModel()`'s own stability is now, transitively, Ax's and
BoTorch's stability, not `boptim`'s. That is an explicit, accepted trade: it is scoped to
callers who opt into using it, and it does not weaken any guarantee the rest of the public API
makes. The alternative (no escape hatch) trades a cleaner-looking API for a real risk of
callers hitting a wall and abandoning the library for their whole project over one missing
capability, which is a worse outcome for a library meant to be "accessible to all" (section
1.1).

**Consequences**

- `axClient` raises `TypeError` (not `None`) when `backend` is not an `AxBackend`, since
  silently returning `None` would push the "is this available" check onto every caller instead
  of failing where the mismatch actually is.
- `fitModel()` is available regardless of backend, since it is defined at the
  `OptimizationBackend` interface level (section 5.3), not Ax-specific.
- Anything reached through `axClient` is, by definition, outside what `boptim` validates or
  keeps in sync with its own domain objects; this is documented on the property itself
  (section 5.7), not just here.

**Action items**

1. [ ] Add `axClient` and `fitModel()` to `BayesianOptimizer` in the same change that
   implements `AxBackend` (Phase 1), not as an afterthought.
2. [ ] Document at least one worked example of each in `examples/` (section 5.1).

#### ADR-0006: `NonlinearConstraint` forces the custom acquisition layer, with a warning

**Status:** Accepted
**Date:** 2026-09-23
**Deciders:** [project owner]

**Context**

FR17 (a feasibility constraint across parameters expressed as an arbitrary expression, e.g.
`var * x ** z <= n`) cannot be expressed through Ax's own `parameter_constraints`, confirmed
linear-only. BoTorch's `optimize_acqf` supports it directly via
`nonlinear_inequality_constraints`, but only within `boptim`'s own custom acquisition layer
(section 4.3), not Ax's default generation strategy (FR4). `BayesianOptimizer.ask()` with
`alpha=None` normally means "use the default strategy".

**Decision**

When `SearchSpace` contains one or more `NonlinearConstraint` and `ask()` is called with
`alpha=None`, `boptim` automatically switches to its own custom acquisition layer instead of
Ax's default strategy, forcing `alpha` to a fixed default (`DEFAULT_ALPHA_WHEN_FORCED = 0.0`),
and logs a warning stating that the switch happened and why. `alpha` passed explicitly already
uses the custom layer, so no switch or warning is needed in that case.

**Options considered**

| Option | Description | Trade-off |
|---|---|---|
| Silently ignore the constraint on the default path | `ask(alpha=None)` behaves exactly as it would without the `NonlinearConstraint` present | Simplest, but silently returns candidates that may violate a constraint the caller explicitly declared: a correctness bug wearing a "no manual tuning needed" feature's clothes |
| Raise an error instead of switching | `ask(alpha=None)` raises if a `NonlinearConstraint` is present, forcing the caller to pass `alpha` explicitly | Never silently wrong, but breaks FR4's promise for a caller who has no opinion on `alpha` and just wants the default path to work |
| Automatically switch layer and warn (chosen) | `alpha` defaults to a fixed value, a warning is logged, `ask()` still returns | Never silently violates the constraint, never blocks the caller who wants the simple path; costs a small amount of "spooky action": the effective strategy for a given call depends on the search space's contents, not only on that call's own arguments |

**Trade-off analysis**

A raised error is honest but actively defeats FR4's own goal (a caller with no opinion on
tuning should still get a good, working default); silently ignoring the constraint is worse in
every way, since it produces wrong answers without saying so. The switch-and-warn is the only
option that keeps FR4's promise (`ask()` still works with no arguments) without breaking the
promise a `NonlinearConstraint` itself makes (returned candidates satisfy it).

**Consequences**

- `ask()`'s behavior for a given `(n_points, alpha)` pair is not fully determined without also
  knowing the search space's constraints; this is documented on `ask()` itself (section 5.7),
  not left implicit.
- `DEFAULT_ALPHA_WHEN_FORCED` is a named constant, not a number buried in the switch logic, so
  it is easy to find and reconsider later.
- Every switch is logged at `WARNING` level through the standard `logging` module (section
  10's own rule), never printed, never silent.

**Action items**

1. [ ] Implement the search-space check and the forced switch in `BayesianOptimizer.ask()`,
   Phase 2, alongside the custom acquisition layer it depends on.
2. [ ] Add a test asserting the warning fires exactly when `alpha=None` and a
   `NonlinearConstraint` is present, and never otherwise.
3. [ ] Add `examples/nonlinear_constraint.py` demonstrating both the constraint and the switch.

---

## 5. Detailed design

This section names the main files, classes, methods and functions with typed signatures.
Docstrings are shortened to one line here for space; the actual code requires full
Google-style docstrings per section 10. Signatures are Python 3.11+ (`X | None`, built-in
generics). File names follow section 10's naming rule: a file holding exactly one public
symbol is named exactly like that symbol (`BayesianOptimizer.py`, `toAxSearchSpace.py`), not a
lower-cased version of it.

### 5.1 Package layout

```
boptim/
├── pyproject.toml
├── README.md
├── docs/
│   ├── index.md
│   ├── contributing.md              # incl. the naming-convention rationale from section 10
│   ├── changelogs/
│   │   ├── CHANGELOG.md             # running index: latest entries + links to each version below
│   │   ├── changelog-v0.1.0.md
│   │   └── changelog-v0.2.0.md      # one immutable file per released version, never edited after
│   │                                 # release, same "new file, don't edit history" spirit as the
│   │                                 # ADRs (section 10)
│   └── adr/
│       ├── 0001-hybrid-ax-botorch-architecture.md
│       ├── 0002-uv-over-setuptools.md
│       ├── 0003-pydantic-domain-models.md
│       ├── 0004-json-persistence-format.md
│       ├── 0005-escape-hatches-to-ax-and-botorch.md
│       └── 0006-nonlinear-constraint-forces-custom-layer.md
├── examples/
│   ├── lab_experiment.py            # a physical-experiment campaign, small n_trials, alpha swept
│   ├── ml_hyperparameter_search.py  # a training-loop objective, larger n_trials
│   ├── conditional_search_space.py  # Choice.dependent_parameters end to end
│   ├── escape_hatch.py              # axClient and fitModel() used directly, ADR-0005
│   └── nonlinear_constraint.py      # NonlinearConstraint + the ask() switch/warning, ADR-0006
├── scripts/
│   ├── new_adr.py                   # scaffolds docs/adr/NNNN-title.md from the ADR-0001 template
│   ├── cut_release.py               # creates docs/changelogs/changelog-vX.Y.Z.md, updates
│   │                                 # CHANGELOG.md's index, bumps the version in pyproject.toml
│   └── check_naming_convention.py   # CI helper: flags a file whose name does not match its
│                                     # single public symbol, catching what ruff's N-rules don't
├── src/boptim/
│   ├── __init__.py                  # re-exports the public surface (BayesianOptimizer, parameter
│   │                                 # classes, Objective, Metric, constraints, ...); exempt from
│   │                                 # the file-naming rule, like every Python dunder file
│   ├── domain/
│   │   ├── parameters/
│   │   │   ├── Parameter.py         # Parameter (ABC)
│   │   │   ├── Range.py             # Range: mirrors ax.api.configs.RangeParameterConfig
│   │   │   ├── Real.py              # Real(Range): sugar, parameter_type="float"
│   │   │   ├── Integer.py           # Integer(Range): sugar, parameter_type="int"
│   │   │   ├── Choice.py            # Choice: mirrors ax.api.configs.ChoiceParameterConfig,
│   │   │   │                        # dependent_parameters included
│   │   │   ├── Categorical.py       # Categorical(Choice): sugar, parameter_type="str"
│   │   │   ├── Boolean.py           # Boolean(Choice): sugar, values=[True, False]
│   │   │   ├── Fixed.py             # Fixed(Choice): sugar, a single-value Choice
│   │   │   └── Derived.py           # Derived: mirrors ax.api.configs.DerivedParameterConfig
│   │   ├── constraints/
│   │   │   ├── Constraint.py        # Constraint (ABC), parameter-level
│   │   │   ├── LinearConstraint.py  # LinearConstraint
│   │   │   ├── NonlinearConstraint.py  # NonlinearConstraint, FR17
│   │   │   └── validateExpression.py   # the restricted expression grammar
│   │   ├── SearchSpace.py
│   │   ├── Metric.py
│   │   ├── Objective.py             # handles N >= 1 metrics, weights, outcome_constraints uniformly
│   │   ├── OutcomeConstraint.py     # metric-level constraint, mirrors Ax's outcome_constraints
│   │   ├── Trial.py
│   │   └── StudySnapshot.py         # full persisted state
│   ├── backends/
│   │   ├── OptimizationBackend.py   # ABC
│   │   ├── PredictionUnavailableError.py
│   │   └── ax/
│   │       ├── AxBackend.py
│   │       ├── toAxSearchSpace.py
│   │       ├── fromAxSearchSpace.py
│   │       └── toAxOptimizationConfig.py
│   ├── models/
│   │   ├── buildSurrogateModel.py
│   │   ├── encodeTrials.py                      # trials -> training tensors
│   │   ├── predictWithModel.py                  # (mean, sem) of a fitted model at a point
│   │   ├── SearchSpaceEncoder.py                # SearchSpace <-> unit cube, ADR-0007
│   │   ├── ParameterEncoding.py                 # one parameter's column layout
│   │   └── compileExpression.py                 # restricted expression -> torch function
│   ├── acquisition/
│   │   ├── AcquisitionStrategy.py               # ABC
│   │   ├── AlphaAcquisitionStrategy.py          # the implementation behind ask(alpha=...)
│   │   ├── ExplorationExploitationAcquisition.py
│   │   ├── MultiObjectiveExplorationExploitationAcquisition.py
│   │   ├── EncodedConstraints.py                # constraints on the encoded tensor
│   │   ├── sampleFeasibleEncoded.py             # constraint-satisfying space-filling draws
│   │   ├── modelParetoFront.py
│   │   └── toBotorchNonlinearConstraints.py     # FR17, ADR-0006
│   ├── analysis/
│   │   ├── SensitivityAnalyzer.py               # ABC
│   │   ├── SobolSensitivityAnalyzer.py
│   │   └── PredictionResult.py
│   ├── persistence/
│   │   ├── StudyRepository.py                   # ABC
│   │   ├── JsonStudyRepository.py               # wraps Client.save_to_json_file/load_from_json_file
│   │   └── ReproducibilityMetadata.py
│   ├── api/
│   │   └── BayesianOptimizer.py                 # the public facade
│   └── logging_config.py                        # module, not a single symbol: kept snake_case
│                                                  # (see section 10's exception for this case)
└── tests/
    ├── unit/                             # mirrors src/boptim/*
    └── integration/
        └── test_end_to_end.py            # dummy objective, full tell/ask/save/load loop
```

### 5.2 Domain layer

```python
# domain/parameters/Parameter.py
class Parameter(ABC):
    """Base class for every kind of search space parameter."""

    name: str
    default: float | int | str | bool | None


# domain/parameters/Range.py
class Range(Parameter):
    def __init__(
        self,
        name: str,
        bounds: tuple[float, float],
        parameter_type: Literal["float", "int"] = "float",
        step_size: float | None = None,
        scaling: Literal["linear", "log"] | None = None,
        default: float | int | None = None,
    ) -> None:
        """A continuous or integer-stepped dimension.

        Field names (bounds, parameter_type, step_size, scaling) match
        ax.api.configs.RangeParameterConfig directly: this class is a typed
        pass-through, not a reinvention. See backends/ax/toAxSearchSpace.py.
        """


# domain/parameters/Real.py
class Real(Range):
    def __init__(
        self,
        name: str,
        min_value: float,
        max_value: float,
        step_size: float | None = None,
        scaling: Literal["linear", "log"] | None = None,
        default: float | None = None,
    ) -> None:
        """Real(name, min_value, max_value) is sugar for
        Range(name, (min_value, max_value), parameter_type="float").
        A class, not a function, on purpose: it is meant to be called like
        a type constructor (Real("x", 0.0, 1.0)), so it follows the
        CamelCase class-naming rule, not the camelCase function one.
        """


# domain/parameters/Integer.py
class Integer(Range):
    def __init__(
        self,
        name: str,
        min_value: int,
        max_value: int,
        step_size: int | None = None,
        scaling: Literal["linear", "log"] | None = None,
        default: int | None = None,
    ) -> None:
        """Integer(name, min_value, max_value) is sugar for
        Range(name, (min_value, max_value), parameter_type="int")."""


# domain/parameters/Choice.py
class Choice(Parameter):
    def __init__(
        self,
        name: str,
        values: list[float] | list[int] | list[str] | list[bool],
        parameter_type: Literal["float", "int", "str", "bool"],
        is_ordered: bool | None = None,
        dependent_parameters: Mapping[float | int | str | bool, Sequence[str]] | None = None,
        default: float | int | str | bool | None = None,
    ) -> None:
        """A discrete dimension: ordinal or categorical, controlled by is_ordered.

        Field names and shape mirror ax.api.configs.ChoiceParameterConfig
        exactly, dependent_parameters included: a chosen value mapped to
        the names of other Parameters that only become part of the active
        search space when that value is picked. This is Ax's own mechanism
        for conditional/hierarchical search spaces (FR14); boptim exposes
        it as-is rather than inventing a parallel one. See section 4.5.
        """


# domain/parameters/Categorical.py
class Categorical(Choice):
    def __init__(
        self,
        name: str,
        categories: list[str],
        dependent_parameters: Mapping[str, Sequence[str]] | None = None,
        default: str | None = None,
    ) -> None:
        """Categorical(name, categories) is sugar for
        Choice(name, categories, parameter_type="str", is_ordered=False)."""


# domain/parameters/Boolean.py
class Boolean(Choice):
    def __init__(self, name: str, default: bool | None = None) -> None:
        """Boolean(name) is sugar for
        Choice(name, [True, False], parameter_type="bool", is_ordered=False)."""


# domain/parameters/Fixed.py
class Fixed(Choice):
    def __init__(self, name: str, value: float | int | str | bool) -> None:
        """A constant, non-optimized value that stays part of the declared
        parameterization (e.g. a setting recorded on every trial but never
        varied), matching Keras Tuner's Fixed.

        ax.api.configs has no separate FixedParameterConfig in the current
        Client API (only Range, Choice, Derived), though Ax's lower-level
        parameter_from_config() utility still names FixedParameter as a
        concept internally. The closest native equivalent, and what this
        maps to in backends/ax, is a single-value Choice; that mapping,
        not a reimplemented "fixed" concept, is what this class is.
        """


# domain/parameters/Derived.py
class Derived(Parameter):
    def __init__(
        self,
        name: str,
        expression: str,
        parameter_type: Literal["float", "int", "str", "bool"],
    ) -> None:
        """A read-only quantity computed from other parameters through an
        expression string (e.g. "a + b"), not itself searched over.

        Mirrors ax.api.configs.DerivedParameterConfig directly. Useful for
        referencing a computed quantity in a LinearConstraint or reading it
        back from a Trial without recomputing it outside boptim.
        """


# domain/constraints/Constraint.py
class Constraint(ABC):
    """Base class for parameter-level constraints (see OutcomeConstraint
    for metric-level ones, which are not a Constraint subclass since they
    apply to a different thing entirely: an observed value, not a
    decision variable)."""


# domain/constraints/LinearConstraint.py
class LinearConstraint(Constraint):
    def __init__(
        self,
        coefficients: dict[str, float],
        bound: float,
        comparator: Literal["<=", ">=", "="],
    ) -> None:
        """sum(coefficients[name] * value[name]) <comparator> bound.

        Rendered to the string expression Client.configure_experiment's
        parameter_constraints expects (e.g. "a + b + c = 1.0") by
        backends/ax/toAxSearchSpace.py; kept as a typed, validated object
        here rather than asking the caller to hand-write Ax's string
        mini-language directly.

        Example: a 3-component mixture summing to 1 is
        LinearConstraint({"a": 1.0, "b": 1.0, "c": 1.0}, bound=1.0, comparator="=").

        Ax's parameter_constraints accept inequalities only, so a "=" constraint is
        not given to Ax: it is enforced by boptim's own acquisition layer, and
        requires_custom_acquisition_layer is True for it (ADR-0008).
        """


# domain/constraints/NonlinearConstraint.py
class NonlinearConstraint(Constraint):
    def __init__(
        self,
        expression: str,
        comparator: Literal["<=", ">="],
        bound: float,
    ) -> None:
        """A feasibility constraint across parameters expressed as an
        arbitrary expression, e.g. NonlinearConstraint("var * x ** z", "<=", n).
        FR17.

        A string, not a Python callable, specifically so a study using one
        still round-trips through JsonStudyRepository (FR11, FR12) like
        every other domain object. Parsed and evaluated by a small,
        restricted expression evaluator (arithmetic and a short allowlist
        of math functions), never Python's own eval.

        No "=" comparator: an exact nonlinear equality is a measure-zero
        constraint a numerical optimizer cannot meaningfully target, unlike
        LinearConstraint's "=" which BoTorch handles as a true linear
        equality constraint.

        Not expressible through Ax's own parameter_constraints (confirmed
        linear-only); rendered instead to a BoTorch
        nonlinear_inequality_constraints callable by
        acquisition/toBotorchNonlinearConstraints.py, and enforced only
        when boptim's own acquisition layer runs. See section 4.6, ADR-0006.
        """


# domain/SearchSpace.py
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


# domain/Metric.py
class Metric:
    def __init__(self, name: str, minimize: bool) -> None:
        """One measured quantity and its optimization direction."""


# domain/OutcomeConstraint.py
class OutcomeConstraint:
    def __init__(
        self,
        metric_name: str,
        bound: float,
        comparator: Literal["<=", ">="],
        relative: bool = False,
    ) -> None:
        """A constraint on an observed metric rather than on a parameter
        (e.g. "qps >= 100"), matching what Client.configure_optimization's
        own outcome_constraints expects (FR15). Lives alongside Objective,
        not inside SearchSpace, because it constrains an observed outcome,
        not a decision variable.

        relative=True expresses the bound as a multiple of a baseline
        trial rather than an absolute value; wiring this to
        Client.attach_baseline is listed as deferred in section 4.7, this
        field just reserves the shape for it.
        """


# domain/Objective.py
class Objective:
    def __init__(
        self,
        metrics: Sequence[Metric],
        weights: Sequence[float] | None = None,
        outcome_constraints: Sequence[OutcomeConstraint] | None = None,
    ) -> None:
        """One or more metrics with optional relative importance weights
        and optional constraints on the metrics themselves.

        `weights=None` with a single metric is the plain single-objective case.
        `weights=None` with several metrics means no preference between them
        (a standard Pareto multi-objective problem). Explicit weights scalarize
        the preference between objectives; they do not control exploration
        versus exploitation, which is `BayesianOptimizer.ask`'s `alpha` (section 4.3).
        """

    @property
    def is_multi_objective(self) -> bool:
        """Cheap (len check): snake_case."""


# domain/Trial.py
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


# domain/StudySnapshot.py
class StudySnapshot:
    """Everything needed to fully reconstruct a BayesianOptimizer: its
    search space, objective, constraints, trial history, and
    reproducibility metadata. The unit that persistence/ reads and
    writes. Wraps, rather than duplicates, whatever Ax's own
    Client.save_to_json_file already captures; see section 5.6."""
```

### 5.3 Backend layer

```python
# backends/OptimizationBackend.py
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
        """Fits and returns the current surrogate model. Used by the
        acquisition layer, and directly exposed to callers as
        BayesianOptimizer.fitModel() (FR16, ADR-0005)."""

    @abstractmethod
    def predict(
        self, x: dict[str, float | int | str | bool]
    ) -> dict[str, tuple[float, float]]:
        """Returns {metric_name: (mean, sem)}. Matches Client.predict's own
        return shape (predicted mean and standard error of the mean, not
        variance) exactly, rather than converting to a different
        uncertainty representation. Raises PredictionUnavailableError if the
        backend has no model to predict with yet (AxBackend: while Ax is still in
        its initial space-filling phase); BayesianOptimizer.predict then uses
        boptim's own surrogate."""

    @abstractmethod
    def computeSensitivity(self) -> dict[str, dict[str, float]]:
        """Returns {metric_name: {parameter_name: importance}}."""

    @abstractmethod
    def getParetoFrontier(self) -> list[Trial]:
        """Delegates to Client.get_pareto_frontier(use_model_predictions=True),
        not a hand-rolled non-domination scan over raw trial data."""

    @abstractmethod
    def getBestTrial(self) -> Trial | None:
        """Delegates to Client.get_best_parameterization() for the
        single-objective case."""


# backends/ax/AxBackend.py
class AxBackend(OptimizationBackend):
    def __init__(
        self,
        random_seed: int | None = None,
        method: Literal["quality", "fast", "random_search"] = "fast",
        initialization_budget: int | None = None,
        torch_device: str | None = None,
    ) -> None:
        """random_seed, method, initialization_budget, torch_device are
        passed straight through to
        Client.configure_generation_strategy (section 4.4); not
        reinvented, just exposed at construction time."""

    # implements every OptimizationBackend method by delegating to a private
    # ax.api.client.Client instance; exact accessor names confirmed against
    # https://ax.readthedocs.io/en/stable/api.html for this draft, re-check
    # against whatever Ax version pyproject.toml ends up pinning.

    @property
    def client(self) -> Any:
        """The live ax.api.client.Client. Returned as Any here to avoid
        leaking an Ax import into this file's own public signature;
        exposed to end users, typed as Ax's own Client, via
        BayesianOptimizer.axClient (section 5.7, ADR-0005)."""


# backends/ax/toAxSearchSpace.py
def toAxSearchSpace(search_space: SearchSpace) -> list[Any]:
    """Maps boptim Parameters to Ax's RangeParameterConfig /
    ChoiceParameterConfig / DerivedParameterConfig list, and
    LinearConstraint to the parameter_constraints string expressions
    Client.configure_experiment expects. Return type kept as Any to avoid
    leaking an Ax import into this file's own public signature."""


# backends/ax/fromAxSearchSpace.py
def fromAxSearchSpace(ax_parameters: list[Any]) -> SearchSpace:
    """The inverse of toAxSearchSpace, used when reconstructing a
    SearchSpace from a study loaded through Client.load_from_json_file
    (section 5.6)."""


# backends/ax/toAxOptimizationConfig.py
def toAxOptimizationConfig(objective: Objective) -> tuple[str, list[str]]:
    """Maps a boptim Objective to the (objective, outcome_constraints)
    strings/string-list Client.configure_optimization expects, weighted or
    unweighted as appropriate."""
```

### 5.4 Surrogate model and acquisition layer

```python
# models/SearchSpaceEncoder.py (ADR-0007)
class SearchSpaceEncoder:
    """Maps a SearchSpace onto the unit cube [0, 1]^d and back: Range -> one column (linear or
    log scaled, rounded onto its grid for int/step ranges); ordered Choice -> one rank column;
    unordered Choice -> a 0/1 column or a one-hot block; Fixed and Derived take no column.
    decode() returns a parameterization Ax accepts: fixed values included, derived parameters
    computed, parameters switched off by dependent_parameters dropped."""

    def __init__(self, search_space: SearchSpace) -> None: ...
    def encodeParameters(self, parameters: Mapping[str, LevelValue]) -> Tensor: ...
    def decode(self, x: Tensor) -> dict[str, LevelValue]: ...
    def snapColumns(self, x: Tensor) -> Tensor: ...
    def sampleEncoded(self, n: int, seed: int | None = None, snap: bool = True) -> Tensor: ...
    def rawRangeValues(self, x: Tensor) -> Tensor: ...  # natural values, for constraints
    def roundingNeighbors(self, x: Tensor, max_ordinal: int = 8) -> Tensor: ...
    def categoricalFixedFeatures(
        self, max_combinations: int, seed: int | None = None
    ) -> list[dict[int, float]]: ...


# models/encodeTrials.py
def encodeTrials(
    encoder: SearchSpaceEncoder, objective: Objective, trials: Sequence[Trial]
) -> tuple[Tensor, Tensor, Tensor | None]:
    """(train_x, train_y, train_yvar). result_std becomes a fixed variance only when every
    trial gives one for every metric; a partial set is ignored with a warning."""


# models/buildSurrogateModel.py
def buildSurrogateModel(
    train_x: Tensor,
    train_y: Tensor,
    train_yvar: Tensor | None = None,
    minimum_points_for_free_fit: int = 5,
) -> Model:
    """Builds and fits a SingleTaskGP with one independent, internally standardized output per
    column of train_y (so also the multi-objective case). train_x is already in the unit cube,
    so no input transform is applied. Infers observation noise when train_yvar is None.

    Fitting is a MAP fit under BoTorch's default priors. minimum_points_for_free_fit is
    accepted now and, in Phase 2, only logs a low-confidence message below it; the distinct
    weakly-informative-prior fallback and explicit low-confidence signalling are Phase 3
    (section 4.4). The single swap point for a scalable/sparse GP in a future version."""


# acquisition/AcquisitionStrategy.py
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
        seed: int | None = None,
    ) -> list[dict[str, float | int | str | bool]]:
        """Any NonlinearConstraint found in search_space.constraints (FR17)
        is converted by toBotorchNonlinearConstraints and passed to
        optimize_acqf as nonlinear_inequality_constraints; no separate
        parameter for it, since SearchSpace already carries it. Every returned
        point satisfies the search space's parameter constraints."""

    @abstractmethod
    def suggestSpaceFilling(
        self,
        search_space: SearchSpace,
        n_points: int,
        seed: int | None = None,
    ) -> list[dict[str, float | int | str | bool]]:
        """The cold start, used while there are too few trials to fit a model:
        constraint-satisfying scrambled-Sobol points."""


# acquisition/toBotorchNonlinearConstraints.py
def toBotorchNonlinearConstraints(
    constraints: Sequence[NonlinearConstraint],
    parameter_order: Sequence[str],
    constants: Mapping[str, float] | None = None,
) -> list[tuple[Callable[[Tensor], Tensor], bool]]:
    """Compiles each NonlinearConstraint's string expression into the
    Tensor-taking callable BoTorch's optimize_acqf expects for its
    nonlinear_inequality_constraints argument (a value >= 0 meaning
    feasible; ">=" constraints pass through as-is, "<=" constraints are
    negated to fit that convention). parameter_order fixes which tensor
    column is which parameter, since the callable only sees a Tensor, not
    named values. constants supplies fixed numeric values an expression may
    also reference. Each callable is intra-point (the bool is True) and returns
    a Tensor, not a float: BoTorch differentiates it."""


# acquisition/ExplorationExploitationAcquisition.py
class ExplorationExploitationAcquisition(MCAcquisitionFunction):
    """Blends a posterior-mean exploitation term and a posterior-variance
    exploration term: score(x) = (1 - alpha) * exploitation(x) + alpha *
    exploration(x), both min-max normalized over reference_points. See
    section 4.3 for how pending points (batches) enter each term. Evaluates
    single candidates (q = 1); batches are built sequentially via set_X_pending."""

    def __init__(
        self,
        model: Model,
        alpha: float,
        minimize: bool,
        reference_points: Tensor,
        sampler: MCSampler | None = None,
        posterior_transform: PosteriorTransform
        | None = None,  # scalarizes a weighted objective
        X_pending: Tensor | None = None,
    ) -> None: ...

    def forward(self, X: Tensor) -> Tensor: ...


# acquisition/MultiObjectiveExplorationExploitationAcquisition.py
class MultiObjectiveExplorationExploitationAcquisition(MultiObjectiveMCAcquisitionFunction):
    """Multi-objective generalization for the unweighted case: exploitation is the
    expected hypervolume improvement over the model-predicted Pareto front (via
    qExpectedHypervolumeImprovement), exploration is the mean of the per-objective
    normalized posterior variances. objective_weights multiplies each outcome before
    comparison: negative for a minimized objective, 1 for a maximized one."""

    def __init__(
        self,
        model: Model,
        alpha: float,
        objective_weights: Tensor | None,
        ref_point: Tensor,
        reference_points: Tensor,
        sampler: MCSampler | None = None,
        X_pending: Tensor | None = None,
    ) -> None: ...

    def forward(self, X: Tensor) -> Tensor: ...


# acquisition/AlphaAcquisitionStrategy.py
class AlphaAcquisitionStrategy(AcquisitionStrategy):
    """The AcquisitionStrategy behind ask(alpha=...). Picks the single-objective function for one
    metric or a weighted objective (on the weighted sum), the multi-objective one otherwise;
    enumerates unordered choices with optimize_acqf_mixed; optimizes integer/grid/ordered
    parameters as continuous relaxations and rounds them, repairing a rounding that breaks a
    constraint; falls back to the best feasible random point if the optimizer returns an
    infeasible one."""

    def __init__(
        self,
        num_restarts: int = 10,
        raw_samples: int = 512,
        n_reference_points: int = 512,
        n_mc_samples: int = 256,
        max_categorical_combinations: int = 32,
        max_iterations: int = 200,
        polytope_burn_in: int = 200,
        polytope_thinning: int = 10,
    ) -> None:
        """polytope_burn_in and polytope_thinning configure the hit-and-run sampler that draws
        starting points inside linear constraints. BoTorch's own defaults (10000 and 32) are
        much more expensive; 200 and 10 were chosen from measurements (burn-in made no
        measurable difference to the sampled distribution, thinning is the expensive knob and
        costs independence of the samples in high dimension), documented where the constants
        are defined, `acquisition/sampleFeasibleEncoded.py`. Raise `polytope_thinning` for a
        high-dimensional polytope."""
```

### 5.5 Analysis layer

```python
# analysis/SensitivityAnalyzer.py
class SensitivityAnalyzer(ABC):
    @abstractmethod
    def computeSensitivity(
        self, model: Model, search_space: SearchSpace, metric_names: Sequence[str]
    ) -> dict[str, dict[str, float]]: ...


# analysis/SobolSensitivityAnalyzer.py
class SobolSensitivityAnalyzer(SensitivityAnalyzer):
    def __init__(self, num_mc_samples: int = 1024) -> None: ...
    def computeSensitivity(
        self, model: Model, search_space: SearchSpace, metric_names: Sequence[str]
    ) -> dict[str, dict[str, float]]:
        """Fallback used when not delegating to Ax's built-in analysis, so the
        acquisition-only code path (no AxBackend involved) still has parameter
        importance available."""


# analysis/PredictionResult.py
@dataclass(frozen=True)
class PredictionResult:
    mean: dict[str, float]
    sem: dict[str, float]
    """Standard error of the mean: matches Client.predict's own (mean, sem)
    return shape directly rather than converting to variance and
    introducing a field Ax itself does not use."""

    @property
    def variance(self) -> dict[str, float]:
        """sem squared over a handful of metrics: negligible cost,
        snake_case even though it "computes" something, per section 10."""
```

### 5.6 Persistence layer

```python
# persistence/StudyRepository.py
class StudyRepository(ABC):
    @abstractmethod
    def save(self, snapshot: StudySnapshot, path: str | Path) -> None: ...

    @abstractmethod
    def load(self, path: str | Path) -> StudySnapshot: ...


# persistence/JsonStudyRepository.py
class JsonStudyRepository(StudyRepository):
    def save(self, snapshot: StudySnapshot, path: str | Path) -> None:
        """Delegates the Ax-backed portion of the state to
        Client.save_to_json_file(path) directly, and writes boptim's own
        extras (parameter default/dependent_parameters metadata Ax's own
        snapshot does not carry, plus ReproducibilityMetadata) alongside
        it. Does not reimplement what Ax already serializes correctly."""

    def load(self, path: str | Path) -> StudySnapshot:
        """The inverse, built on Client.load_from_json_file(path)."""


# persistence/ReproducibilityMetadata.py
@dataclass(frozen=True)
class ReproducibilityMetadata:
    random_seed: int
    library_versions: dict[str, str]
    created_at: datetime
    boptim_version: str
```

### 5.7 Public API facade

`ask`/`tell` is the naming used by scikit-optimize and Optuna for exactly this pattern, so it
was adopted here over the earlier `suggest`/`observe` names, for the sake of matching an
established convention that both target audiences (ML practitioners and, increasingly, the
BO-literate lab) are likely to already recognize. The constructor accepts either a plain list
of `Parameter`s (the common case) or a fully-built `SearchSpace`/`Objective` pair (the case
that needs constraints or multi-objective weighting spelled out), so simple use stays terse
without losing access to the advanced path. `axClient` and `fitModel()` are the escape hatches
from design philosophy #2 and ADR-0005.

```python
# api/BayesianOptimizer.py
class BayesianOptimizer:
    def __init__(
        self,
        parameters: Sequence[Parameter] | SearchSpace,
        objective: Objective | Literal["minimize", "maximize"] = "minimize",
        constraints: Sequence[Constraint] | None = None,
        outcome_constraints: Sequence[OutcomeConstraint] | None = None,
        name: str = "study",
        random_seed: int | None = None,
        backend: OptimizationBackend | None = None,
        acquisition_strategy: AcquisitionStrategy | None = None,
    ) -> None:
        """The main entry point. Two ways to call it:

        Common case: BayesianOptimizer(parameters=[Real(...), Integer(...)]).
            A single, unnamed metric is assumed; objective="minimize" or
            "maximize" picks its direction.
        Advanced case: pass an already-built SearchSpace (carrying its own
            constraints) and an already-built Objective (named metrics,
            optional weights, optional outcome_constraints, single or
            multi-objective). `constraints`/`outcome_constraints` must be
            left None in that case (a ValueError is raised otherwise: the
            SearchSpace/Objective already own their own constraints, so
            passing both is an ambiguous request, not a merge).

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
                custom acquisition layer, UNLESS the search space has one or
                more NonlinearConstraint (FR17): Ax's default strategy cannot
                enforce one, so alpha is instead set to
                DEFAULT_ALPHA_WHEN_FORCED (0.0) and a warning is logged
                explaining the switch, rather than silently returning
                candidates that might violate a constraint the caller
                declared. Passing alpha explicitly always uses the custom
                layer already, so nothing changes and no warning fires in
                that case. See section 4.6, ADR-0006.
        """

    def predict(self, x: dict[str, float | int | str | bool]) -> PredictionResult:
        """FR7, FR8. Uses the backend's own model when it has one; otherwise
        (early in a study) boptim's own surrogate, fit on the trial history and
        cached until the next tell() (ADR-0007). Raises PredictionUnavailableError
        below two completed trials."""

    def parameterImportance(self) -> dict[str, dict[str, float]]:
        """Refits/queries the surrogate model: has a cost, camelCase. FR6."""

    @property
    def n_trials(self) -> int:
        """len() over an already-held list: cheap, snake_case."""

    @property
    def paretoFront(self) -> list[Trial]:
        """Delegates to the backend's getParetoFrontier() (Ax's own
        Client.get_pareto_frontier under AxBackend, section 5.3), not a
        hand-rolled non-domination scan. Has a cost, camelCase. Collapses
        to a single-element list for a single-objective optimizer, via the
        backend's getBestTrial()."""

    def save(self, path: str | Path) -> None:
        """FR11, FR12. See JsonStudyRepository, section 5.6."""

    @classmethod
    def load(cls, path: str | Path) -> BayesianOptimizer:
        """FR11, FR12."""

    @property
    def axClient(self) -> Any:
        """Escape hatch (FR16, ADR-0005): the live ax.api.client.Client
        instance backing this optimizer. Raises TypeError if backend is
        not an AxBackend. Typed Any here to avoid forcing an Ax import on
        every caller of this file; the real return type is
        ax.api.client.Client. Anything reached through this property is,
        by definition, outside what boptim validates or keeps in sync with
        its own domain objects: for example, a trial attached directly via
        axClient.attach_trial(...) will not appear as a boptim Trial until
        the caller also updates the boptim side, since boptim only learns
        about it through this same escape hatch, not automatically."""

    def fitModel(self) -> Model:
        """Escape hatch (FR16, ADR-0005): the BoTorch surrogate model Ax
        currently holds, so a caller can write and optimize their own
        acquisition function with plain BoTorch and feed the result back
        through tell(), without forking boptim to get an acquisition behavior
        the alpha dial does not cover. This is Ax's model, in Ax's transformed
        input space, and only exists once Ax has left its initial
        space-filling phase; it is not the model ask(alpha=...) uses (that one
        is fit by boptim itself, ADR-0007)."""
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
y = run_my_experiment(**x)  # your own code: a real experiment or a training run
bo.tell(x, {"objective": y})

# ask for a batch of 3, balanced exploration/exploitation (FR10)
batch = bo.ask(n_points=3, alpha=0.5)

# what does the model currently believe, with uncertainty? (FR7, FR8)
prediction = bo.predict(
    {"temperature": 100.0, "num_layers": 4, "solvent": "ethanol", "use_catalyst": False}
)
print(prediction.mean, prediction.sem)

bo.parameterImportance()  # FR6
bo.save("study.json")  # FR11, FR12
bo_reloaded = BayesianOptimizer.load("study.json")
```

A conditional search space (FR14, section 4.5), using Ax's own `dependent_parameters`
mechanism directly rather than a boptim-invented one. This is the direct equivalent of the
Keras Tuner example that set the minimum bar for this feature (`hp.conditional_scope("model_type", ["mlp"])`
activating `hidden_units`, `hp.conditional_scope("model_type", ["cnn"])` activating
`num_filters`):

```python
from boptim import BayesianOptimizer, Categorical, Integer

bo = BayesianOptimizer(
    parameters=[
        Categorical(
            "model_type",
            ["mlp", "cnn"],
            dependent_parameters={"mlp": ["hidden_units"], "cnn": ["num_filters"]},
        ),
        Integer("hidden_units", 8, 512),
        Integer("num_filters", 8, 256),
    ]
)
```

A nonlinear feasibility constraint (FR17, section 4.6), and the automatic switch it triggers
on the default path (ADR-0006):

```python
from boptim import BayesianOptimizer, Real, NonlinearConstraint

bo = BayesianOptimizer(
    parameters=[Real("var", 0.0, 10.0), Real("x", 0.0, 10.0), Real("z", 0.0, 3.0)],
    constraints=[NonlinearConstraint("var * x ** z", "<=", 50.0)],
    objective="maximize",
)

bo.ask()  # alpha left at its default: NOT Ax's usual no-tuning strategy here, since
# Ax cannot enforce the constraint above. boptim switches to its own
# acquisition layer with alpha=DEFAULT_ALPHA_WHEN_FORCED and logs a
# warning saying so. bo.ask(alpha=0.3) would use that layer directly,
# with no warning, since nothing is happening implicitly in that case.
```

The escape hatch (FR16, ADR-0005), for a feature `boptim` does not wrap, here Ax's closed-loop
`run_trials` (section 4.7):

```python
bo.axClient.run_trials(max_trials=20)  # Ax drives the loop itself; see ADR-0005's caveat
# about this bypassing boptim's own bookkeeping
```

---

## 6. Roadmap

The phases are sequenced by technical risk, not by feature importance: every functional
requirement from section 2 is in scope, but the genuinely novel part (the alpha dial) is
built once the foundation under it is solid, rather than the other way around.

**Phase 1: Foundation (domain model + Ax-backed default flow)**
Full domain model, including `Range`/`Real`/`Integer`, `Choice`/`Categorical`/`Boolean`/
`Fixed`/`Derived` with `dependent_parameters` (FR14's definition side), linear parameter
constraints (FR9, `LinearConstraint`, Ax-native), outcome constraints (FR15), and
multi-objective (`Objective` with N >= 1 metrics and optional weights), from the start: Ax's
`Client` natively supports all of this, so this is realistically a mapping-layer effort, not a
re-design later. `AxBackend` (including `getParetoFrontier`/`getBestTrial` delegating to Ax,
and `predict` returning Ax's own (mean, sem) shape), manual injection, `ask()` via Ax's own
default strategy, Ax's built-in sensitivity analysis, `JsonStudyRepository` wrapping Ax's own
save/load, and the `axClient`/`fitModel()` escape hatches (FR16). Deliverable: a usable, if
not yet alpha-controllable, end-to-end loop, plus `examples/lab_experiment.py` and
`examples/escape_hatch.py`.

**Phase 2: The exploration/exploitation acquisition layer**
`ExplorationExploitationAcquisition` and its multi-objective variant, with batch support
(fantasization-based sequential selection) built in from the start rather than retrofitted.
This is FR5, the requirement Ax does not cover, and the reason this project exists rather
than "just use Ax". `NonlinearConstraint` and `toBotorchNonlinearConstraints` (FR17) land in
the same phase, since they are only enforceable once this layer exists; `ask()`'s automatic
switch and warning (ADR-0006) is built and tested alongside them, not after. `examples/
ml_hyperparameter_search.py` and `examples/nonlinear_constraint.py` follow once `alpha` is
usable.

*Status: implemented. See ADR-0007 (the layer fits its own surrogate), the corrected batch
mechanism in section 4.3, and the evidence on the forced default in ADR-0006. Not yet covered,
by design or by deferral: `OutcomeConstraint`s are not enforced by the custom layer (a warning
is logged); `dependent_parameters` are optimized as if all were active and pruned when
decoding (Phase 3 makes the layer reason about them); `LinearConstraint`s on log-scaled
parameters are rejected by the custom layer.*

**Phase 3: Robustness across the full budget range, and deferred hard problems**
Weakly-informative-prior fallback for very small trial counts (down to the 4-5-evaluation
case), explicit low-confidence signaling; the scalable/sparse-GP extension point in
`buildSurrogateModel` documented (not necessarily implemented) for very large trial counts;
FR14's harder half, making the custom acquisition layer actually reason about
`dependent_parameters` instead of just not breaking on it; and, revisiting section 4.7,
`simplify_parameter_changes`/BONSAI, given its direct relevance to a human reconfiguring lab
equipment between trials.

**Phase 4: Polish, docs, packaging**
Full `mkdocs` site (including the `docs/changelogs/` structure and `scripts/cut_release.py`),
ADR-0003 and ADR-0004 written, CI green on lint/type-check/tests/`scripts/
check_naming_convention.py`, `uv`-based packaging finalized, integration test covering the
full tell/ask/save/load loop on a dummy objective, and `examples/conditional_search_space.py`
alongside the two campaign-style examples, so the "general-purpose, not lab-specific" intent
(section 1.1) is demonstrated, not just stated.

---

## 10. Coding standards

*(Section numbers 7-9 and 11 are intentionally left open, consistent with the numbering style
of the source template this document extends.)*

- **Language/runtime:** Python 3.11+
- **Formatting/linting:** `ruff` (lint + format), consistent import ordering.
- **Naming convention (custom, overrides PEP8 default for callables and for file names):**
  - Classes -> `CamelCase` (e.g. `SearchSpace`, `Real`, `ExplorationExploitationAcquisition`).
  - Variables (including function/method parameters) -> `snake_case` (e.g. `n_points`, `random_seed`).
  - Functions and methods -> same rule as classes but starting lowercase, i.e. `camelCase`
    (e.g. `suggestDefault`, `computeSensitivity`, `buildAcquisitionFunction`), not PEP8's usual
    `snake_case` for callables. Python's required dunder methods (`__init__`, `__repr__`, ...)
    are exempt: they keep their mandatory spelling.
  - **File names match the file's single public symbol exactly**, including its case: a file
    defining `class BayesianOptimizer` is `BayesianOptimizer.py`, not `bayesian_optimizer.py`;
    a file defining `def buildSurrogateModel(...)` is `buildSurrogateModel.py`. This is the
    "one class per file" rule (already required by this section) taken one step further: if a
    file's name and its one symbol's name can drift apart, they will, over refactors. A file
    with more than one closely related symbol and no single obvious name (rare, given "one
    responsibility per file") keeps a descriptive `snake_case` name instead; `logging_config.py`
    (section 5.1) is that case, not an exception to avoid. Python's own required module names
    (`__init__.py`, `conftest.py`, ...) are exempt, the same way dunder methods are.
    `scripts/check_naming_convention.py` (section 5.1) enforces this in CI, since `ruff`'s
    naming rules do not cover file names against class/function names.
  - Properties (`@property`) are named by **cost, not by whether they compute anything**:
    - If accessing the property does real work (recomputes something from the full trial
      history, calls into a model, is not O(1)-ish) -> `camelCase`, exactly like a method,
      even though it is called without parentheses. Example: `BayesianOptimizer.paretoFront`.
    - If it just returns an already-stored or negligible-cost value -> `snake_case`, like a
      variable. Example: `BayesianOptimizer.n_trials`, `PredictionResult.variance`.
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
- **Version control:** Conventional Commits, semantic versioning. Changelog entries are
  versioned files, not one growing document: each release gets its own
  `docs/changelogs/changelog-vX.Y.Z.md`, written once and never edited after release (the same
  "new file records what changed, old ones stay put" spirit as the ADRs below), and
  `docs/changelogs/CHANGELOG.md` is a short running index linking to each of them.
  `scripts/cut_release.py` scaffolds this at release time. Architectural decisions get a new
  ADR when they change, rather than an old ADR being edited in place (see `docs/adr/0002-*.md`
  for an example of a decision recorded this way).
- **CI:** GitHub Actions running lint, type-check, tests, and `scripts/
  check_naming_convention.py` on every push.
- **Documentation:** `mkdocs` + `mkdocstrings` built from docstrings; major architectural
  choices (e.g. "why a hybrid Ax + BoTorch architecture", "why Pydantic domain models
  decoupled from Ax's own types", "why escape hatches to Ax and BoTorch") recorded as short
  ADRs (`docs/adr/NNNN-title.md`).
- **Flexibility:** the code should be as flexible and generalist as possible so it can be
  adapted to any user, lab or otherwise. Avoid hard-coded values that may be significant to a
  particular user; define sensible defaults instead (see section 1.4 for the defaults chosen
  in this draft, and section 4.4 for how this plays out for very small vs. very large trial
  budgets).
- **Don't recode what's already coded.** Before adding a new abstraction, check whether
  Ax (https://ax.readthedocs.io/en/stable/api.html) or BoTorch already has the shape needed;
  section 4.5 and 4.6 are the running record of that check for this project, kept up to date
  rather than done once and forgotten.

---

## 12. How future Claude conversations (and contributors) should use this document

- Treat sections 0, 2, 3, 5 and 10 as the contract: search-space types, functional
  requirements, the technology choices and their rationale (section 4.8's ADRs), the package
  layout, and the naming/style rules. Do not silently deviate from them; propose a change and,
  if accepted, update the relevant ADR or section rather than drifting from it in code.
- Section 1.4 ("Defaults chosen for this draft") lists the handful of decisions made to keep
  this document concrete rather than because they were architecturally required. These are
  safe to revisit without re-opening the rest of the design.
- Section 5's signatures are the target shape of the code, not yet the code. When
  implementing, keep the signatures; if a signature turns out to be wrong once real code is
  written against it, update this document in the same change, not after.
- Before adding any new domain abstraction, check it against Ax's and BoTorch's own API
  reference first (section 4.7 is the running list of what was deliberately deferred, not
  missed); this document was revised more than once specifically because an earlier draft
  invented a mechanism (conditional parameters via a custom `Condition` class) that Ax already
  provides (`dependent_parameters`). Don't repeat that.
- If a request does not have a clear answer in this document (a new feature, a changed
  requirement, an ambiguous naming case not covered by section 10), do not guess: ask, the way
  this document itself was produced through a round of clarifying questions before being
  written. Once answered, fold the answer back into this document so the next conversation
  does not have to ask again.
