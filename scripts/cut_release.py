#!/usr/bin/env python3
"""Cuts a release: scaffolds `docs/changelogs/changelog-vX.Y.Z.md`, updates
`CHANGELOG.md`'s index, and bumps the version in `pyproject.toml`.

Usage:
    python scripts/cut_release.py 0.2.0 "Phase 2: exploration/exploitation acquisition layer"

Does not touch git (no commit, no tag): that stays a manual, deliberate step
after reviewing what this script generated, matching the "new file records
what changed, old ones stay put" spirit of section 10's versioning rule (the
same spirit `docs/adr/`'s own files follow).
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

PYPROJECT_PATH = Path("pyproject.toml")
CHANGELOG_INDEX_PATH = Path("docs/changelogs/CHANGELOG.md")
CHANGELOG_DIRECTORY = Path("docs/changelogs")

ENTRY_TEMPLATE = """\
# v{version} ({date})

{summary}

## Added

-

## Changed

-

## Fixed

-
"""

_VERSION_LINE = re.compile(r'^version = "[^"]*"$', re.MULTILINE)


def bumpPyprojectVersion(version: str) -> None:
    """Sets the `version = "..."` line of `pyproject.toml`.

    Args:
        version: The new version.

    Raises:
        ValueError: if `pyproject.toml` has no version line.
    """
    text = PYPROJECT_PATH.read_text(encoding="utf-8")
    new_text, count = _VERSION_LINE.subn(f'version = "{version}"', text, count=1)
    if count == 0:
        raise ValueError(f'No `version = "..."` line found in {PYPROJECT_PATH}.')
    PYPROJECT_PATH.write_text(new_text, encoding="utf-8")


def writeChangelogEntry(version: str, summary: str) -> Path:
    """Writes the immutable changelog entry of a release from the template.

    Args:
        version: The released version.
        summary: A one-line summary.

    Returns:
        The path of the new entry.

    Raises:
        FileExistsError: if the entry already exists. A released entry is never overwritten.
    """
    CHANGELOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    entry_path = CHANGELOG_DIRECTORY / f"changelog-v{version}.md"
    if entry_path.exists():
        raise FileExistsError(
            f"{entry_path} already exists; not overwriting a released entry."
        )
    entry_path.write_text(
        ENTRY_TEMPLATE.format(version=version, date=date.today().isoformat(), summary=summary),
        encoding="utf-8",
    )
    return entry_path


def prependToChangelogIndex(version: str, summary: str) -> None:
    """Adds a release to the top of the running changelog index, creating the index if needed.

    Args:
        version: The released version.
        summary: A one-line summary.
    """
    today = date.today().isoformat()
    new_line = f"- [v{version}](./changelog-v{version}.md) ({today}): {summary}.\n"
    if not CHANGELOG_INDEX_PATH.exists():
        CHANGELOG_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        CHANGELOG_INDEX_PATH.write_text(
            "# Changelog\n\nA short running index. Each release has its own immutable entry, "
            "written once and never edited after release.\n\n" + new_line,
            encoding="utf-8",
        )
        return
    lines = CHANGELOG_INDEX_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    insert_at = next(
        (i for i, line in enumerate(lines) if line.startswith("- [v")), len(lines)
    )
    lines.insert(insert_at, new_line)
    CHANGELOG_INDEX_PATH.write_text("".join(lines), encoding="utf-8")


def main(argv: list[str]) -> int:
    """Cuts a release: writes its changelog entry, updates the index, bumps the version.

    Args:
        argv: The command line: the version, then an optional one-line summary.

    Returns:
        `0` on success, `2` if no version was given.
    """
    if len(argv) < 2:
        print(
            'Usage: python scripts/cut_release.py X.Y.Z ["one-line summary"]', file=sys.stderr
        )
        return 2

    version = argv[1]
    summary = argv[2] if len(argv) > 2 else f"Release {version}."

    entry_path = writeChangelogEntry(version, summary)
    prependToChangelogIndex(version, summary)
    bumpPyprojectVersion(version)

    print(f"cut_release: wrote {entry_path}")
    print(f"cut_release: updated {CHANGELOG_INDEX_PATH}")
    print(f"cut_release: bumped {PYPROJECT_PATH} to version {version!r}")
    print("cut_release: review the generated files, then commit and tag manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
