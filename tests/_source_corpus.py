"""ONE shared, session-scoped parse of the torchlens package source.

Dozens of lint/census tests each walked ``torchlens/`` and ``ast.parse``'d
all ~520 files independently — ~5 s of CPU and ~270 MB of AST allocation
churn PER TEST FILE, duplicated across the session, with the gen-2 gc cost
of every parse landing on whichever test happened to be allocating.

This module is the one shared corpus. ``conftest.pytest_collection_finish``
calls :func:`prewarm` (only when a consumer module was imported during
collection) BEFORE the import-time ``gc.freeze()``, so the corpus is built
exactly once and lands in the frozen generation: gen-2 collections never
scan it, and per-test gc cost stays proportional to session-created objects.

Contract for consumers:

* ``package_files()`` is the canonical walk (sorted, ``__pycache__``
  excluded); iterate it instead of re-globbing.
* ``package_source(path)`` / ``package_ast(path)`` return SHARED objects.
  Never mutate a returned tree (no ``node.parent = ...`` annotation passes);
  a consumer that must annotate copies first. Reads are unlimited.
* The corpus reflects the source tree at first access in the process — the
  same snapshot semantics every per-test ``rglob`` walk already had.
"""

from __future__ import annotations

import ast
from functools import cache, lru_cache
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
PACKAGE_ROOT = REPO_ROOT / "torchlens"


@lru_cache(maxsize=1)
def package_files() -> tuple[Path, ...]:
    """Every ``.py`` file under the torchlens package, sorted, one walk."""

    return tuple(
        sorted(path for path in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in path.parts)
    )


@cache
def package_source(path: Path) -> str:
    """The file's source text, read once per process."""

    return path.read_text(encoding="utf-8")


@cache
def package_ast(path: Path) -> ast.Module:
    """The file's parsed module tree, parsed once per process. SHARED: never mutate."""

    return ast.parse(package_source(path), filename=str(path))


def module_source(path: Path) -> str:
    """Corpus-cached source for package files; uncached direct read elsewhere."""

    resolved = path.resolve()
    if PACKAGE_ROOT in resolved.parents:
        return package_source(resolved)
    return resolved.read_text(encoding="utf-8")


def module_ast(path: Path) -> ast.Module:
    """Corpus-cached parse for package files; uncached direct parse elsewhere.

    The uncached branch keeps non-package parses OUT of the long-lived
    corpus (they would accumulate post-``gc.freeze`` and re-create the
    per-test gen-2 gc drag the freeze exists to kill).
    """

    resolved = path.resolve()
    if PACKAGE_ROOT in resolved.parents:
        return package_ast(resolved)
    return ast.parse(resolved.read_text(encoding="utf-8"), filename=str(resolved))


def prewarm() -> None:
    """Build the whole corpus now (called pre-``gc.freeze`` from conftest)."""

    for path in package_files():
        package_ast(path)
