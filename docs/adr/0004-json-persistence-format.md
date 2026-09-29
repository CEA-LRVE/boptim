# ADR-0004: JSON as the persistence format, wrapping Ax's own save/load

**Status:** Proposed (to write in full)
**Date:** 2026-09-23
**Deciders:** [project owner]

## Context to seed it

FR11/FR12, and that `Client.save_to_json_file`/`load_from_json_file` already serialize the
Ax-backed state correctly; `JsonStudyRepository` (`PROJECT_SPECIFICATION.md` section 5.6)
wraps that rather than reinventing it, and only adds what Ax's own snapshot does not carry.

## Phase 1 implementation note (to fold into this ADR when written)

`Client` only offers a *file-based* save/load API (`save_to_json_file(filepath)`/
`load_from_json_file(filepath)`), not an in-memory JSON string. `AxBackend.exportState`/
`AxBackend.importState` bridge this by writing to (and reading from) a `tempfile.
TemporaryDirectory()`-scoped file, so that `StudySnapshot.backend_state` can hold the
resulting `dict` directly and `JsonStudyRepository` can still present the caller with exactly
**one** JSON file per study (`BayesianOptimizer.save(path)`/`.load(path)` both take a single
`path`), rather than a pair of sibling files. This single-file design was not explicit in
`PROJECT_SPECIFICATION.md` section 5.3/5.6's signatures (`OptimizationBackend.exportState`/
`.importState` do not appear in section 5.3 at all: they were added during Phase 1
implementation to give `JsonStudyRepository` a backend-agnostic hook for the Ax-backed portion
of the state, without `persistence/` importing `ax.*` directly, which would break ADR-0001's
"only `backends/ax` imports Ax" rule). Flagged here, and in `backends/OptimizationBackend.py`'s
own docstrings for those two methods, for the project owner to review per section 12.

## Action items

1. [ ] Write this ADR in full, following the ADR-0001 format.
2. [ ] Review and, if accepted, formally add `exportState`/`importState` to
   `PROJECT_SPECIFICATION.md` section 5.3's `OptimizationBackend` listing (see the note above
   and in `backends/OptimizationBackend.py`).
