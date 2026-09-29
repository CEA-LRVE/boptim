# ADR-0006: `NonlinearConstraint` forces the custom acquisition layer, with a warning

**Status:** Accepted (implementation deferred to Phase 2)
**Date:** 2026-09-23
**Deciders:** [project owner]

## Context

FR17 (a feasibility constraint across parameters expressed as an arbitrary expression, e.g.
`var * x ** z <= n`) cannot be expressed through Ax's own `parameter_constraints`, confirmed
linear-only. BoTorch's `optimize_acqf` supports it directly via
`nonlinear_inequality_constraints`, but only within `boptim`'s own custom acquisition layer
(`PROJECT_SPECIFICATION.md` section 4.3), not Ax's default generation strategy (FR4).
`BayesianOptimizer.ask()` with `alpha=None` normally means "use the default strategy".

## Decision

When `SearchSpace` contains one or more `NonlinearConstraint` and `ask()` is called with
`alpha=None`, `boptim` automatically switches to its own custom acquisition layer instead of
Ax's default strategy, forcing `alpha` to a fixed default (`DEFAULT_ALPHA_WHEN_FORCED = 0.0`),
and logs a warning stating that the switch happened and why. `alpha` passed explicitly already
uses the custom layer, so no switch or warning is needed in that case.

## Options considered

| Option | Description | Trade-off |
|---|---|---|
| Silently ignore the constraint on the default path | `ask(alpha=None)` behaves exactly as it would without the `NonlinearConstraint` present | Simplest, but silently returns candidates that may violate a constraint the caller explicitly declared: a correctness bug wearing a "no manual tuning needed" feature's clothes |
| Raise an error instead of switching | `ask(alpha=None)` raises if a `NonlinearConstraint` is present, forcing the caller to pass `alpha` explicitly | Never silently wrong, but breaks FR4's promise for a caller who has no opinion on `alpha` and just wants the default path to work |
| Automatically switch layer and warn (chosen) | `alpha` defaults to a fixed value, a warning is logged, `ask()` still returns | Never silently violates the constraint, never blocks the caller who wants the simple path; costs a small amount of "spooky action": the effective strategy for a given call depends on the search space's contents, not only on that call's own arguments |

## Trade-off analysis

A raised error is honest but actively defeats FR4's own goal (a caller with no opinion on
tuning should still get a good, working default); silently ignoring the constraint is worse in
every way, since it produces wrong answers without saying so. The switch-and-warn is the only
option that keeps FR4's promise (`ask()` still works with no arguments) without breaking the
promise a `NonlinearConstraint` itself makes (returned candidates satisfy it).

## Consequences

- `ask()`'s behavior for a given `(n_points, alpha)` pair is not fully determined without also
  knowing the search space's constraints; this is documented on `ask()` itself (section 5.7),
  not left implicit.
- `DEFAULT_ALPHA_WHEN_FORCED` is a named constant, not a number buried in the switch logic, so
  it is easy to find and reconsider later.
- Every switch is logged at `WARNING` level through the standard `logging` module (section
  10's own rule), never printed, never silent.

## Action items

1. [ ] Implement the search-space check and the forced switch in `BayesianOptimizer.ask()`,
   Phase 2, alongside the custom acquisition layer it depends on.
2. [ ] Add a test asserting the warning fires exactly when `alpha=None` and a
   `NonlinearConstraint` is present, and never otherwise.
3. [ ] Add `examples/nonlinear_constraint.py` demonstrating both the constraint and the switch.

## Phase 1 status

`NonlinearConstraint` does not exist yet (it is deferred to Phase 2 alongside the acquisition
layer that is the only thing able to enforce it — see `domain/constraints/__init__.py`'s own
docstring), so this ADR's switch has nothing to trigger on yet. In this Phase 1 drop,
`BayesianOptimizer.ask(alpha=...)` raises `NotImplementedError` for *any* non-`None` `alpha`,
unconditionally: the distinction this ADR draws (explicit `alpha` vs. an implicit,
constraint-triggered switch) only becomes meaningful once both `alpha` and
`NonlinearConstraint` are real, in Phase 2.
