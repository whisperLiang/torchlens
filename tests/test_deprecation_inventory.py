"""The deprecation census and its expiry metadata (grind r2 row 15 / matrix R48).

The debt this closes: ``torchlens/_deprecations.py`` carries no removal-version
metadata, so shims accumulate with no expiry and nobody can answer "which of
these is past its window?". Round 0 counted ~202 "deprecated" mentions with no
inventory behind them.

This module IS the inventory. Every deprecation the package can emit belongs to
exactly one registered family, each family carries removal metadata, and the
membership is DERIVED from the shipped tables (runtime imports for the alias
maps, an AST scan for the emission sites) rather than transcribed -- so a new
deprecation cannot be added without registering it, and a retired one cannot
linger as a phantom row.

Removal is an API decision, not a test's decision: ``remove_in`` is a closed
vocabulary of honest states, and the report surfaces which families are waiting
on a maintainer call. Nothing here removes a shim.

The inventory lives in the tests, next to the exemption ledger
(``tests/test_validation_exemption_ledger.py``), so auditing the deprecation
surface adds no package code.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pytest

import torchlens as tl
from torchlens import options as tl_options

pytestmark = pytest.mark.smoke

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _REPO_ROOT / "torchlens"

#: Warning categories that make a ``warnings.warn`` call a DEPRECATION emission.
#: ``TorchLensDeprecationWarning`` (grind b4, R48-2/R48-4) is a
#: ``DeprecationWarning`` subclass introduced so the pytest gate can select
#: TorchLens's own deprecations by CATEGORY -- correct ``stacklevel`` now blames
#: the caller, so the old module-keyed filter no longer sees them. The scanner
#: must recognize both spellings or converting a site to the subclass would make
#: it vanish from the census, which is exactly the invisibility this file exists
#: to prevent.
_DEPRECATION_CATEGORY_NAMES = frozenset({"DeprecationWarning", "TorchLensDeprecationWarning"})

#: How the deprecated spelling reaches the user.
_KINDS = frozenset(
    {
        "moved_name",  # top-level name that now lives in a submodule
        "api_shim",  # paper-era public function kept as a shim
        "kwarg_alias",  # old keyword spelling of a still-supported option
        "kwarg_value",  # still-accepted kwarg, deprecated VALUE
        "noop_kwarg",  # accepted and ignored; the feature is gone
        "noop_function",  # callable retained as an inert stub
        "attr_alias",  # renamed attribute/method on a public class
        "artifact_advisory",  # not an API deprecation at all (see the finding)
    }
)

#: Honest removal states. Neither invents a version number: TorchLens has never
#: recorded one for these shims, and picking one is a maintainer call.
_REMOVE_IN_VOCABULARY = frozenset(
    {
        # The package already advertises a vague window in code
        # (``torchlens/__init__.py::_REMOVED_IN`` == "a future 2.x release").
        # Registered as-is; tightening it to a real version is the open ask.
        "unspecified_future_2x",
        # No window advertised anywhere. Removal is a public-API call and is
        # forked to the maintainer, never taken by a grind lane.
        "pending_maintainer_signoff",
        # Retained deliberately with no removal intent recorded yet.
        "retained_indefinitely",
    }
)


@dataclass(frozen=True)
class DeprecationFamily:
    """One registered family of deprecated spellings.

    Parameters
    ----------
    name:
        Stable family id.
    kind:
        Member of :data:`_KINDS`.
    replacement:
        Canonical spelling users should move to.
    remove_in:
        Member of :data:`_REMOVE_IN_VOCABULARY`.
    deprecated_in:
        TorchLens version that started warning, or ``"unrecorded"`` for families
        that predate this registry. Deliberately not back-filled by guesswork.
    sites:
        ``file::function`` emission sites owned by this family. Closed against an
        AST scan of the package.
    members:
        Explicitly enumerated deprecated spellings, for families whose members
        are not derivable from a shipped table.
    note:
        Anything an auditor needs that the fields above do not carry.
    """

    name: str
    kind: str
    replacement: str
    remove_in: str
    deprecated_in: str
    sites: tuple[str, ...]
    members: tuple[str, ...] = ()
    note: str = ""


DEPRECATION_FAMILIES: tuple[DeprecationFamily, ...] = (
    DeprecationFamily(
        name="moved_top_level_names",
        kind="moved_name",
        replacement="the owning submodule (torchlens.types / .errors / .io / ...)",
        remove_in="unspecified_future_2x",
        deprecated_in="unrecorded",
        sites=("torchlens/__init__.py::_warn_moved_name",),
        note=(
            "Membership is torchlens.__init__._MOVED_OBJECTS. The warning already "
            "says 'Removed in <_REMOVED_IN>', whose value is the prose 'a future "
            "2.x release' -- an advertised but unactionable window."
        ),
    ),
    DeprecationFamily(
        name="paper_era_api_shims",
        kind="api_shim",
        replacement="the 2.x spelling named in torchlens.__init__._LEGACY_API_SHIMS",
        remove_in="pending_maintainer_signoff",
        deprecated_in="unrecorded",
        sites=("torchlens/__init__.py::_warn_legacy_api_name",),
        note=(
            "log_forward_pass / render_graph / ModelHistory and friends. The "
            "warning text calls them a compatibility shim with no window at all."
        ),
    ),
    DeprecationFamily(
        name="flat_option_kwargs",
        kind="kwarg_alias",
        replacement="the grouped options object (capture=/visualization=/...)",
        remove_in="pending_maintainer_signoff",
        deprecated_in="unrecorded",
        sites=("torchlens/_deprecations.py::warn_deprecated_alias",),
        members=("mode", "node_mode", "max_module_depth", "layout_engine"),
        note=(
            "The largest family by far. Membership is derived from the shipped "
            "_*_FLAT_TO_GROUP tables, minus the visualization names that are NOT "
            "in _VISUALIZATION_DEPRECATED_FLAT and minus save.grad_transform, "
            "which the resolver excludes from warning. The four explicitly listed "
            "members are the pre-2.x draw() spellings resolved outside those "
            "tables (options.py _normalize_visualization_kwargs)."
        ),
    ),
    DeprecationFamily(
        name="renamed_public_callables",
        kind="attr_alias",
        replacement="the canonical name passed as warn_deprecated_alias's second argument",
        remove_in="pending_maintainer_signoff",
        deprecated_in="unrecorded",
        sites=("torchlens/_deprecations.py::warn_deprecated_alias",),
        members=(
            "peek",
            "batched_extract",
            "record_span",
            "capture_output_structure",
            "get_model_metadata",
            "validate_saved_outs",
            "vis_node_mode",
            "replay",
            "replay_from",
            "rerun",
            "intervening",
            "param",
            "Trace.replay",
            "Trace.replay_from",
            "Trace.rerun",
            "Trace.validate_saved_outs",
            "Bundle.replay",
            "Bundle.rerun",
            "conditional_then_entry_edges",
            "conditional_elif_entry_edges",
            "conditional_else_entry_edges",
        ),
    ),
    DeprecationFamily(
        name="crawler_era_noop_functions",
        kind="noop_function",
        replacement="nothing: the rescue re-run + mechanical belt replaced the crawler",
        remove_in="pending_maintainer_signoff",
        deprecated_in="unrecorded",
        sites=(
            "torchlens/backends/torch/wrappers.py::patch_detached_references",
            "torchlens/backends/torch/wrappers.py::clear_patch_detached_references_cache",
        ),
        members=("patch_detached_references", "clear_patch_detached_references_cache"),
        note=(
            "Inert stubs kept so callers that inspected the old PatchReport "
            "counters keep importing. Documented in "
            "docs/migration/scoped_detached_patching.md."
        ),
    ),
    DeprecationFamily(
        name="crawler_era_noop_kwargs",
        kind="noop_kwarg",
        replacement="nothing: accepted and ignored",
        remove_in="pending_maintainer_signoff",
        deprecated_in="unrecorded",
        sites=("torchlens/backends/torch/wrappers.py::wrap_torch",),
        members=("wrap_torch(patch_policy=)", "wrap_torch(patch_modules=)"),
        note="CLAUDE.md already records these as deprecated no-ops.",
    ),
    DeprecationFamily(
        name="domain_node_styles",
        kind="kwarg_value",
        replacement="examples/recipes/<style>.py, or the future torchlens.<style> plugin",
        remove_in="pending_maintainer_signoff",
        deprecated_in="unrecorded",
        sites=(
            "torchlens/options.py::_validate_node_style",
            "torchlens/visualization/_render_dot.py::_validate_draw_options",
        ),
        members=("node_style='vision'", "node_style='attention'"),
        note=(
            "A deprecated VALUE, not a deprecated name: the kwarg stays. Two "
            "independent validators emit it, so both sites are registered."
        ),
    ),
    DeprecationFamily(
        name="inert_backward_perturbation_flag",
        kind="noop_kwarg",
        replacement="nothing: the flag drove an inert check, never a real comparison",
        remove_in="pending_maintainer_signoff",
        deprecated_in="unrecorded",
        sites=("torchlens/validation/backward.py::validate_backward_pass",),
        members=("validate_backward_pass(perturb_saved_grads=True)",),
        note=(
            "The warning itself states the old implementation was not a "
            "captured-gradient comparison, so there is nothing to preserve."
        ),
    ),
    # REMOVED in grind r3 (R15-F1): the "artifact_schema_age_advisory" family
    # was never an API deprecation -- it advises that a loaded bundle predates
    # the runtime schema. It now raises the visible
    # `torchlens._io.ArtifactSchemaAgeWarning` (a UserWarning subclass), so it
    # is out of this inventory's scope by construction. Its behavior is pinned
    # by tests/test_rehydration_floor.py::
    # test_between_floor_advisory_is_a_visible_user_warning.
)

_FAMILIES_BY_NAME = {family.name: family for family in DEPRECATION_FAMILIES}


# ---------------------------------------------------------------------------
# Derived side: emission sites and alias membership, read from the package.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PackageScan:
    """One pass of AST facts about a package tree.

    Parameters
    ----------
    sites:
        ``path::function`` keys of raw ``warnings.warn(..., DeprecationWarning)``
        calls.
    alias_names:
        Literal old names passed to ``warn_deprecated_alias``.
    """

    sites: frozenset[str]
    alias_names: frozenset[str]


@lru_cache(maxsize=8)
def scan_package(package_root: Path, base: Path) -> _PackageScan:
    """Return both deprecation AST facts in ONE cached pass over the tree.

    Parsing the package twice (once per closure test) cost ~30s, which is too
    slow for a smoke-tier gate; one cached recursive pass with no parent map runs
    in about a second.

    Parameters
    ----------
    package_root:
        Root of the package tree to scan.
    base:
        Directory reported paths are relative to.

    Returns
    -------
    _PackageScan
        Emission sites and literal alias names.
    """

    sites: set[str] = set()
    alias_names: set[str] = set()
    for path in sorted(package_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        # Cheap prefilter: an emission is impossible unless one of the two tokens
        # appears literally in the source, so only those files are parsed. This
        # is what keeps a whole-package AST audit inside the smoke budget.
        if "DeprecationWarning" not in text and "warn_deprecated_alias" not in text:
            continue
        relative = path.relative_to(base).as_posix()
        _visit(ast.parse(text), relative, "<module>", sites, alias_names)
    return _PackageScan(frozenset(sites), frozenset(alias_names))


def _visit(
    node: ast.AST,
    relative: str,
    enclosing: str,
    sites: set[str],
    alias_names: set[str],
) -> None:
    """Collect deprecation facts under ``node``, tracking the enclosing function.

    Parameters
    ----------
    node:
        Node to descend from.
    relative:
        Repo-relative path of the module being scanned.
    enclosing:
        Name of the innermost enclosing function.
    sites:
        Accumulator for ``path::function`` emission sites.
    alias_names:
        Accumulator for literal deprecated spellings.
    """

    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "warn":
            categories = [*node.args, *(keyword.value for keyword in node.keywords)]
            if any(
                isinstance(value, ast.Name) and value.id in _DEPRECATION_CATEGORY_NAMES
                for value in categories
            ):
                sites.add(f"{relative}::{enclosing}")
        elif (
            isinstance(func, ast.Name)
            and func.id == "warn_deprecated_alias"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            alias_names.add(node.args[0].value)
    for child in ast.iter_child_nodes(node):
        child_enclosing = (
            child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else enclosing
        )
        _visit(child, relative, child_enclosing, sites, alias_names)


def deprecation_emission_sites(package_root: Path, base: Path | None = None) -> set[str]:
    """Return ``file::function`` for every RAW DeprecationWarning emission.

    Only direct ``warnings.warn(..., DeprecationWarning)`` calls count. Families
    that emit through the shared ``warn_deprecated_alias`` helper are closed by
    NAME instead (:func:`literal_alias_names`), which is the stronger check for
    them: the helper has exactly one emission site but dozens of callers.

    Parameters
    ----------
    package_root:
        Root of the shipped package.
    base:
        Base directory the reported paths are relative to. Defaults to the repo
        root; tests planting a synthetic package pass their own.

    Returns
    -------
    set[str]
        ``path::function`` keys relative to ``base``.
    """

    return set(scan_package(package_root, base if base is not None else _REPO_ROOT).sites)


def literal_alias_names(package_root: Path, base: Path | None = None) -> set[str]:
    """Return every literal old name passed to ``warn_deprecated_alias``.

    Parameters
    ----------
    package_root:
        Root of the shipped package.
    base:
        Base directory used for path reporting (irrelevant to the result; kept so
        both accessors share one cached scan).

    Returns
    -------
    set[str]
        Deprecated spellings named as string literals at the call site.
    """

    return set(scan_package(package_root, base if base is not None else _REPO_ROOT).alias_names)


def warning_flat_kwarg_names() -> set[str]:
    """Return flat option kwargs whose use emits a deprecation warning.

    Mirrors ``options._resolve_option_group``: every flat name in a
    ``_*_FLAT_TO_GROUP`` table warns, except that the visualization group warns
    only for ``_VISUALIZATION_DEPRECATED_FLAT`` and the save group excludes
    ``grad_transform``.

    Returns
    -------
    set[str]
        Deprecated flat kwarg spellings.
    """

    names: set[str] = set()
    names |= set(tl_options._CAPTURE_FLAT_TO_GROUP)
    names |= set(tl_options._SAVE_FLAT_TO_GROUP) - {"grad_transform"}
    names |= set(tl_options._VISUALIZATION_DEPRECATED_FLAT)
    names |= set(tl_options._REPLAY_FLAT_TO_GROUP)
    names |= set(tl_options._INTERVENTION_FLAT_TO_GROUP)
    names |= set(tl_options._STREAMING_FLAT_TO_GROUP)
    return names


def registered_alias_members() -> set[str]:
    """Return every deprecated spelling covered by a registered family.

    Returns
    -------
    set[str]
        Union of explicit family members, the derived flat-kwarg family, the
        moved-name table, and the paper-era shim table.
    """

    members: set[str] = set()
    for family in DEPRECATION_FAMILIES:
        members |= set(family.members)
    members |= warning_flat_kwarg_names()
    members |= set(tl._MOVED_OBJECTS)
    members |= set(tl._LEGACY_API_SHIMS)
    return members


def inventory_gaps(derived: set[str], registered: set[str]) -> tuple[set[str], set[str]]:
    """Return unregistered and phantom entries.

    Parameters
    ----------
    derived:
        Entries found in the package.
    registered:
        Entries covered by :data:`DEPRECATION_FAMILIES`.

    Returns
    -------
    tuple[set[str], set[str]]
        ``(unregistered, phantom)``.
    """

    return derived - registered, registered - derived


# ---------------------------------------------------------------------------
# The R48 gate.
# ---------------------------------------------------------------------------


def test_every_deprecation_has_removal_metadata() -> None:
    """Every registered family carries an honest removal state and a replacement."""

    assert DEPRECATION_FAMILIES
    for family in DEPRECATION_FAMILIES:
        assert family.kind in _KINDS, f"{family.name}: unknown kind {family.kind!r}"
        assert family.remove_in in _REMOVE_IN_VOCABULARY, (
            f"{family.name}: remove_in {family.remove_in!r} is outside the closed "
            "vocabulary; a new state needs a documented meaning, not a free-text value"
        )
        assert family.deprecated_in, f"{family.name}: deprecated_in must be set"
        assert family.replacement.strip(), f"{family.name}: needs a replacement spelling"
        assert family.sites, f"{family.name}: needs at least one emission site"


def test_family_names_are_unique() -> None:
    """Family ids are unique, so the registry is a real index."""

    names = [family.name for family in DEPRECATION_FAMILIES]
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# Closure: the census is exhaustive against the package.
# ---------------------------------------------------------------------------


def test_every_deprecation_emission_site_is_registered() -> None:
    """A DeprecationWarning cannot be emitted from an unregistered site."""

    derived = deprecation_emission_sites(_PACKAGE_ROOT)
    registered = {site for family in DEPRECATION_FAMILIES for site in family.sites}
    unregistered, phantom = inventory_gaps(derived, registered)
    assert not unregistered, (
        "DeprecationWarning emitted from a site with no inventory entry (register "
        f"it in DEPRECATION_FAMILIES with removal metadata): {sorted(unregistered)}"
    )
    assert not phantom, f"registered sites that no longer emit: {sorted(phantom)}"


def test_every_literal_alias_name_is_registered() -> None:
    """Every literally-named deprecated spelling belongs to a family."""

    derived = literal_alias_names(_PACKAGE_ROOT)
    unregistered, _ = inventory_gaps(derived, registered_alias_members())
    assert not unregistered, f"deprecated spellings with no inventory entry: {sorted(unregistered)}"


def test_flat_kwarg_family_membership_is_derived_not_transcribed() -> None:
    """The flat-kwarg family reads the shipped tables, so it cannot go stale."""

    derived = warning_flat_kwarg_names()
    assert "save_grads" in derived, "capture flat kwargs warn and must be inventoried"
    assert "vis_node_mode" in derived, "deprecated visualization flat names are inventoried"
    assert "grad_transform" not in derived, (
        "options._resolve_option_group excludes save.grad_transform from warning; "
        "the inventory must mirror that, not over-claim"
    )
    assert "view" not in derived, (
        "canonical grouped-field spellings that merely also work flat are NOT "
        "deprecated and must not be counted as debt"
    )


def test_moved_name_and_shim_tables_are_nonempty_and_disjoint() -> None:
    """The two top-level tables stay distinct surfaces."""

    moved = set(tl._MOVED_OBJECTS)
    shims = set(tl._LEGACY_API_SHIMS)
    assert moved and shims
    assert not moved & shims, f"a name cannot be both moved and a legacy shim: {moved & shims}"


# ---------------------------------------------------------------------------
# The census, reported as numbers so the debt is measurable round over round.
# ---------------------------------------------------------------------------


def deprecated_spelling_census() -> dict[str, int]:
    """Return the per-surface count of deprecated public spellings.

    Returns
    -------
    dict[str, int]
        Surface name -> number of deprecated spellings.
    """

    census = {
        "moved_top_level_names": len(tl._MOVED_OBJECTS),
        "paper_era_api_shims": len(tl._LEGACY_API_SHIMS),
        "flat_option_kwargs": len(
            warning_flat_kwarg_names() | set(_FAMILIES_BY_NAME["flat_option_kwargs"].members)
        ),
    }
    for family in DEPRECATION_FAMILIES:
        if family.members and family.name not in census:
            census[family.name] = len(family.members)
    return census


def test_census_matches_the_recorded_baseline() -> None:
    """The census is pinned, so growth in the deprecation surface is visible.

    A new shim is not forbidden -- the house rule is "no NEW shims", and this is
    how that rule becomes checkable instead of aspirational.
    """

    assert deprecated_spelling_census() == {
        "moved_top_level_names": 50,
        "paper_era_api_shims": 9,
        "flat_option_kwargs": 80,
        "renamed_public_callables": 21,
        "crawler_era_noop_functions": 2,
        "crawler_era_noop_kwargs": 2,
        "domain_node_styles": 2,
        "inert_backward_perturbation_flag": 1,
    }


def test_no_family_claims_a_concrete_removal_version_yet() -> None:
    """No shim is scheduled for removal without a maintainer decision.

    Deliberately asserted rather than assumed: a grind lane must not quietly
    schedule a public-API removal, and this test is where such a change becomes
    visible. When a real version is chosen it is added to
    ``_REMOVE_IN_VOCABULARY`` and this expectation is updated in the same diff.
    """

    scheduled = [
        family.name
        for family in DEPRECATION_FAMILIES
        if family.remove_in
        not in {"unspecified_future_2x", "pending_maintainer_signoff", "retained_indefinitely"}
    ]
    assert not scheduled, f"removal scheduled without sign-off: {scheduled}"


# ---------------------------------------------------------------------------
# The inventory mechanism must be able to go RED.
# ---------------------------------------------------------------------------


class TestInventoryMechanismIsRedCapable:
    """Plant an unregistered deprecation and prove the closure reports it."""

    def test_site_scanner_finds_a_planted_emission(self, tmp_path: Path) -> None:
        """A new DeprecationWarning site is discovered by the scan."""

        package = tmp_path / "torchlens"
        package.mkdir()
        (package / "mod.py").write_text(
            "import warnings\ndef f():\n    warnings.warn('x', DeprecationWarning, stacklevel=2)\n",
            encoding="utf-8",
        )
        found = deprecation_emission_sites(package, base=tmp_path)
        assert found == {"torchlens/mod.py::f"}, found

    def test_site_scanner_ignores_other_warning_categories(self, tmp_path: Path) -> None:
        """A UserWarning is not a deprecation and is not demanded."""

        package = tmp_path / "torchlens"
        package.mkdir()
        (package / "mod.py").write_text(
            "import warnings\ndef f():\n    warnings.warn('x', UserWarning)\n",
            encoding="utf-8",
        )
        assert deprecation_emission_sites(package, base=tmp_path) == set()

    def test_alias_scanner_finds_a_planted_literal(self, tmp_path: Path) -> None:
        """A new literal alias name is discovered by the scan."""

        package = tmp_path / "torchlens"
        package.mkdir()
        (package / "mod.py").write_text(
            "def f():\n    warn_deprecated_alias('planted_old', 'planted_new')\n",
            encoding="utf-8",
        )
        assert literal_alias_names(package, base=tmp_path) == {"planted_old"}

    def test_inventory_gap_checker_reports_both_directions(self) -> None:
        """Unregistered and phantom entries are both surfaced."""

        unregistered, phantom = inventory_gaps({"new"}, {"old"})
        assert unregistered == {"new"}
        assert phantom == {"old"}

    def test_removal_vocabulary_rejects_free_text(self) -> None:
        """``remove_in`` is closed, so 'soon' cannot masquerade as metadata."""

        assert "soon" not in _REMOVE_IN_VOCABULARY
        assert "3.0" not in _REMOVE_IN_VOCABULARY
