#!/usr/bin/env python3
"""CI helper: flags a file whose name does not match its single public symbol.

`ruff`'s naming rules (`N801`, etc.) check the *spelling* of a class or function
name, but not whether the *file* it lives in is named after it. This script
does the second check, enforcing section 10's rule: "File names match the
file's single public symbol exactly."

A file with zero or more than one top-level public (non-underscore-prefixed)
class/function is left alone: section 10 already carves out that case
("a file with more than one closely related symbol and no single obvious
name ... keeps a descriptive snake_case name instead"), and this script has
no reliable way to guess whether such a name is "descriptive" enough.

Usage:
    python scripts/check_naming_convention.py [root]

Exits non-zero (and prints one line per violation) if any file whose single
public symbol does not match its file name is found.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

#: File (base) names exempt from the rule, matching section 10's own carve-outs
#: for Python's required module names and this project's one deliberate
#: snake_case module (`logging_config.py`, explained in its own docstring).
EXEMPT_FILE_NAMES = frozenset({"__init__.py", "conftest.py", "logging_config.py"})

#: Directories never scanned: virtual envs, caches, build artifacts.
EXCLUDED_DIR_NAMES = frozenset({
    ".venv", "venv", "__pycache__", ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "build", "dist", "site",
})


def findTopLevelPublicSymbolNames(source: str) -> list[str]:
    """Returns the names of every top-level `class`/`def`/`async def` in
    `source` whose name does not start with `_`, in source order.
    """
    tree = ast.parse(source)
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                names.append(node.name)
    return names


def checkFile(path: Path) -> str | None:
    """Returns a human-readable violation message for `path`, or `None` if it
    complies (or is exempt, or has zero/multiple public symbols).
    """
    if path.name in EXEMPT_FILE_NAMES:
        return None
    public_names = findTopLevelPublicSymbolNames(path.read_text(encoding="utf-8"))
    if len(public_names) != 1:
        return None
    (symbol_name,) = public_names
    if path.stem != symbol_name:
        return (
            f"{path}: file defines a single public symbol {symbol_name!r} but "
            f"is not named {symbol_name}.py"
        )
    return None


def findPythonFiles(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*.py")
        if not any(part in EXCLUDED_DIR_NAMES for part in p.parts)
    )


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path("src")
    if not root.exists():
        print(f"check_naming_convention: root {root} does not exist", file=sys.stderr)
        return 2

    violations: list[str] = []
    files_checked = 0
    for path in findPythonFiles(root):
        files_checked += 1
        violation = checkFile(path)
        if violation is not None:
            violations.append(violation)

    if violations:
        print(f"check_naming_convention: {len(violations)} violation(s):")
        for violation in violations:
            print(f"  - {violation}")
        return 1

    print(f"check_naming_convention: OK ({files_checked} files checked, 0 violations).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
