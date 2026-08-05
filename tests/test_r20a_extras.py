"""Packaging guards for ``pyproject.toml`` optional-dependency extras (r20a).

These tests assert the metadata-level contract of the advertised extras, catching
the two defects the r20a hardening pass fixed:

* ``all`` was an empty extra (``all = []``), so ``pip install torchlens[all]``
  silently installed nothing while a populated ``all-stretch`` carried the real
  union -- a name-vs-behaviour lie.
* ``io`` was a vestigial, unpinned duplicate of ``tabular``'s ``pyarrow>=14`` that
  no code or docs referenced (the parquet exporter directs users to ``[tabular]``).

They parse ``pyproject.toml`` directly (not installed metadata, which can be stale
under an editable install) so they track the source of truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from packaging.requirements import Requirement

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on the py3.10 CI leg when tomli is present
    tomllib = pytest.importorskip(
        "tomli", reason="need tomllib (py>=3.11) or tomli (py<3.11) to parse pyproject.toml"
    )

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _optional_dependencies() -> dict[str, list[str]]:
    with PYPROJECT.open("rb") as handle:
        data = tomllib.load(handle)
    return data["project"]["optional-dependencies"]


def _extras_referenced_by(specs: list[str]) -> set[str]:
    """Return the torchlens extras a self-referential extra pulls in."""

    referenced: set[str] = set()
    for spec in specs:
        req = Requirement(spec)
        assert req.name == "torchlens", (
            f"meta-rollup member {spec!r} must self-reference torchlens (e.g. "
            "'torchlens[all-stretch]'), not name a third-party package directly"
        )
        assert req.extras, f"meta-rollup member {spec!r} must reference at least one extra"
        referenced |= set(req.extras)
    return referenced


def test_all_extra_is_non_empty() -> None:
    extras = _optional_dependencies()
    assert "all" in extras, "the `all` extra must exist"
    assert extras["all"], (
        "`all` must not be empty: `pip install torchlens[all]` must install something"
    )


def test_all_extra_members_reference_valid_non_empty_extras() -> None:
    extras = _optional_dependencies()
    defined = set(extras)
    referenced = _extras_referenced_by(extras["all"])
    assert referenced, "`all` must aggregate at least one extra"
    unknown = referenced - defined
    assert not unknown, f"`all` references extras that are not defined: {sorted(unknown)}"
    empty = sorted(name for name in referenced if not extras[name])
    assert not empty, f"`all` aggregates empty extras (installs nothing): {empty}"


def test_all_extra_resolves_to_real_packages() -> None:
    """The transitive closure of `all` must resolve to at least one real dependency."""

    extras = _optional_dependencies()
    packages: set[str] = set()
    for name in _extras_referenced_by(extras["all"]):
        for spec in extras[name]:
            packages.add(Requirement(spec).name.lower())
    assert packages, "`all` must transitively install real third-party packages"
