# ADR-0003: Pydantic domain models decoupled from Ax's internal types

**Status:** Proposed (to write in full)
**Date:** 2026-09-23
**Deciders:** [project owner]

## Context to seed it

Design philosophy #2 (`PROJECT_SPECIFICATION.md` section 1.3): "`boptim`'s public API is its
own, but it is not a cage." Ax and BoTorch are implementation details behind an adapter for
the common case, not something a caller has to understand to get started, and Ax's own public
API surface has already changed once (the move to `ax.api.client.Client`/`ax.api.configs`
this specification itself was revised to track — see the note at the top of
`PROJECT_SPECIFICATION.md`). A domain layer with zero ML dependencies (section 4.1) is what
lets `boptim`'s public surface stay stable even if the backend underneath it changes.

Phase 1 built this decision into the code (`domain/` has no `ax.*` or `botorch.*` imports
anywhere; `boptim.domain.parameters.Range.Range.toAxKwargs()` and its siblings are the one
narrow, explicit seam where a domain object exposes what it needs to become an Ax config,
called only from `backends/ax/`) without this ADR having been written out in full yet. Writing
it should mostly be a matter of recording the decision already reflected in the package layout
(section 4.1, section 5.1), following the ADR-0001 format above, rather than making a new one.

## Phase 1 implementation note (to fold into this ADR when written)

Pydantic models with a positional, spec-shaped `__init__` are (re)built by Pydantic through
that same `__init__`, called with *field names*. That works for `Range`, `Choice`, `Derived`
and every concrete model, but not for the sugar subclasses (`Real` takes `lower_bound`, not
`bounds`). And lists of the abstract `Parameter`/`Constraint` cannot be serialized or rebuilt
by declared type. Decision: `toDict()` on each base kind plus `parameterFromDict` /
`constraintFromDict` dispatchers, wired into `SearchSpace` with `field_validator(mode="before")`
and `field_serializer`. Adding a new `Parameter`/`Constraint` kind means adding it to its
dispatcher in the same change.

## Action items

1. [ ] Write this ADR in full, following the ADR-0001 format.
2. [ ] Cross-reference `domain/parameters/Range.py`'s `toAxKwargs()` method (and its
   siblings on `Choice`/`Derived`) as the concrete seam this ADR's decision produces.
