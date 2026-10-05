#!/usr/bin/env python3
"""Scaffolds `docs/adr/NNNN-title.md` from the ADR-0001 template.

Usage:
    python scripts/new_adr.py "Some short decision title"

Picks the next unused 4-digit ADR number by scanning `docs/adr/` for
existing `NNNN-*.md` files, slugifies the title for the file name, and
writes a template with today's date and every section ADR-0001 has
(Status, Date, Deciders, Context, Decision, Options considered, Trade-off
analysis, Consequences, Action items) left for the author to fill in.
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

ADR_DIRECTORY = Path("docs/adr")

TEMPLATE = """\
# ADR-{number}: {title}

**Status:** Proposed
**Date:** {date}
**Deciders:** [project owner]

## Context

<!-- What problem does this decision address? What constraints apply? -->

## Decision

<!-- The decision itself, stated plainly. -->

## Options considered

| Option | Description | Trade-off |
|---|---|---|
| | | |

## Trade-off analysis

<!-- Why the chosen option wins, and what it explicitly gives up. -->

## Consequences

<!-- What this decision commits future code/contributors to. -->

## Action items

1. [ ]
"""


def slugify(title: str) -> str:
    """Turns a title into a lowercase, hyphen-separated file name fragment.

    Args:
        title: The decision's title.

    Returns:
        The slug, for example `"use ax"` gives `"use-ax"`.
    """
    lowered = title.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug


def nextAdrNumber(adr_directory: Path) -> int:
    """Finds the next free ADR number.

    Args:
        adr_directory: The directory holding the `NNNN-*.md` records.

    Returns:
        One more than the highest existing number, or `1` if there are none.
    """
    highest = 0
    for path in adr_directory.glob("[0-9][0-9][0-9][0-9]-*.md"):
        highest = max(highest, int(path.name[:4]))
    return highest + 1


def main(argv: list[str]) -> int:
    """Creates the next numbered ADR file from the template.

    Args:
        argv: The command line: the decision's title.

    Returns:
        `0` on success, `1` if the file already exists, `2` on a usage error.
    """
    if len(argv) != 2:
        print('Usage: python scripts/new_adr.py "Some short decision title"', file=sys.stderr)
        return 2

    title = argv[1]
    ADR_DIRECTORY.mkdir(parents=True, exist_ok=True)
    number = nextAdrNumber(ADR_DIRECTORY)
    slug = slugify(title)
    path = ADR_DIRECTORY / f"{number:04d}-{slug}.md"
    if path.exists():
        print(f"new_adr: {path} already exists; not overwriting.", file=sys.stderr)
        return 1

    path.write_text(
        TEMPLATE.format(number=f"{number:04d}", title=title, date=date.today().isoformat()),
        encoding="utf-8",
    )
    print(f"new_adr: wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
