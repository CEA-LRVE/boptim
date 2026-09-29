# ADR-0002: `uv` for project and dependency management

**Status:** Accepted
**Date:** 2026-09-23
**Deciders:** [project owner]

## Context

The project has a heavy, sometimes fragile dependency tree (PyTorch, BoTorch, Ax, GPyTorch),
several of which have platform-specific wheels. FR12 explicitly requires reproducibility. The
project owner's default on other projects is `setuptools`.

## Decision

Use `uv` for environment creation, dependency resolution, locking, and running dev commands
(`uv run`, `uv sync`). Use a simple build backend (`uv_build`, or `hatchling` if an Ax/BoTorch
edge case needs it) rather than `setuptools`, since this is a pure-Python package with no C
extensions to compile.

## Options considered

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

## Trade-off analysis

For a pure-Python package, `setuptools`' specific strength (compiled extensions) is not
relevant here, so it offers no advantage that matters for this project. `uv`'s advantages
(install speed, native lockfile) map directly onto two things this project actually has: a
heavy dependency tree and an explicit reproducibility requirement. If `setuptools` is preferred
for familiarity, it can still be used as just the build backend under `uv`
(`uv init --build-backend setuptools`) without giving up any of `uv`'s other benefits.

## Consequences

- Contributors run `uv sync` once instead of manually managing a virtualenv.
- `uv.lock` is committed to version control.
- CI uses `uv`'s official GitHub Action instead of a manual `pip install` step.

## Action items

1. [x] `uv init`-equivalent `pyproject.toml` written (Phase 1). Confirming the pinned
   dependency set resolves cleanly requires running `uv sync` against a real package index,
   which the environment this ADR and the surrounding code were authored in does not have
   network access to do; this is the first thing to run once this drop reaches an environment
   that does.
2. [ ] Commit `uv.lock`.
3. [ ] Update the CI workflow to use `astral-sh/setup-uv`.

## Correction after first real `uv sync`

The `[build-system].requires` pin (`uv_build>=0.4,<0.5`) was wrong: no version in that range
exists on the index, so `uv sync` failed immediately on `boptim` itself, before ever reaching
`ax-platform`/`botorch`. `uv_build` moves in lockstep with `uv`'s own fast release cadence, so
the fix is a wide, open-ended range (`uv_build>=0.5,<1.0`) rather than a narrower guess that
would just go stale again. Also fixed in the same pass: `[tool.uv] dev-dependencies` is
deprecated by current `uv` in favor of PEP 735 `[dependency-groups]`; moved there, with
`[project.optional-dependencies].dev` kept alongside it so `uv sync --extra dev` (as the
README documents) still works unchanged. Both were caught by a person actually running
`uv sync --extra dev --extra docs` locally, not by anything in this sandbox: a concrete
example of why the README's verification-status note (and this project's own habit of
recording things this way, in a new section rather than editing the numbers above) matters.
