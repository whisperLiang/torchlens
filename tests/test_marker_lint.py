"""Tier-marker lint: keep the smoke tier honest.

Two enforcement layers back the documented test tiers (CLAUDE.md "Testing Tiers",
tests/AGENTS.md "Markers"):

1. **Marker disjointness** (collection-level). pytest markers are ADDITIVE: a test
   carrying ``smoke`` together with ``heavy`` or ``slow`` still runs under
   ``-m smoke`` -- the heavier marker does NOT remove it from the fast gate. That
   combination is always a partition mistake; the fix is dropping ``smoke`` from
   the test (per-parametrize-case marks that give DIFFERENT cases different tiers
   are fine and pass this check, because ``get_closest_marker`` sees the resolved
   per-item marks).

2. **Duration budget** (runtime tripwire). The conftest hook accounts for fixture
   setup, call, and teardown time, then checks both each item and each resolved
   parametrized family against the documented 5s partition boundary.

3. **State-isolation census** (static). Every warn-once module global must appear
   in the root autouse reset inventory, and every module-scoped fixture that
   creates a Trace must yield so it can release that Trace at module teardown.

Both checks read their data from the surrounding session, so they enforce on every
smoke/full-tier run and pass vacuously when this file is run alone.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke


def test_no_smoke_test_carries_a_heavier_tier_marker(request: pytest.FixtureRequest) -> None:
    """No collected item may combine ``smoke`` with ``heavy`` or ``slow``."""

    conflicted = []
    for item in request.session.items:
        if item.get_closest_marker("smoke") is None:
            continue
        for heavier in ("heavy", "slow"):
            if item.get_closest_marker(heavier) is not None:
                conflicted.append(f"{item.nodeid} [smoke + {heavier}]")
    assert not conflicted, (
        "Tests carry `smoke` together with a heavier tier marker, so `-m smoke` "
        "still runs them despite the heavier mark. Drop `smoke` from each "
        "(markers are additive):\n  " + "\n  ".join(conflicted)
    )


def test_smoke_tests_stay_within_duration_budget(request: pytest.FixtureRequest) -> None:
    """Every smoke-marked test must finish within the tier's duration budget.

    The budget value lives in ``tests/conftest.py`` (``SMOKE_DURATION_BUDGET_SECONDS``)
    and rides along on each recorded offender -- a bare ``conftest`` import here would
    be ambiguous during full-suite collection (nested conftests share the module name).
    """

    offenders = getattr(request.session, "_tl_smoke_budget_offenders", [])
    lines = [
        f"{nodeid}: {duration:.1f}s (budget {budget:.0f}s)"
        for nodeid, duration, budget in offenders
    ]
    assert not offenders, (
        "Smoke-marked tests exceeded the smoke-tier duration budget this session. "
        "Re-tier them (move to `heavy` for 5-20s, `slow` for >20s) or make them "
        "faster:\n  " + "\n  ".join(lines)
    )


def test_smoke_parametrized_families_stay_within_duration_budget(
    request: pytest.FixtureRequest,
) -> None:
    """Resolved smoke parameter families must stay within the aggregate budget.

    A family of N parameters legitimately costs ~N single-test durations (the
    selector matrix is 278 cells; the surface oracle is 6 goldens), so the
    aggregate budget is the enforcement budget (not the 5s partition
    threshold), load-scaled like the per-test hook. Tightening both to 5s is
    the tracked follow-up that lands with the >5s re-tier sweep.
    """

    budget = getattr(request.session, "_tl_smoke_family_budget_value", 30.0)
    family_totals = getattr(request.session, "_tl_smoke_family_durations", {})
    offenders = [
        (family, duration) for family, duration in family_totals.items() if duration > budget
    ]
    lines = [f"{family}: {duration:.1f}s (budget {budget:.0f}s)" for family, duration in offenders]
    assert not offenders, (
        "Smoke parametrized families exceeded the aggregate smoke-tier budget. "
        "Split or re-tier the family:\n  " + "\n  ".join(lines)
    )


def test_smoke_module_imports_stay_within_duration_budget(
    request: pytest.FixtureRequest,
) -> None:
    """Smoke-bearing modules must import and collect within the 5s boundary."""

    budget = 5.0
    durations = getattr(request.session, "_tl_module_collection_durations", {})
    smoke_paths = {
        str(item.path)
        for item in request.session.items
        if item.get_closest_marker("smoke") is not None
    }
    offenders = [
        (path, durations[path]) for path in sorted(smoke_paths) if durations.get(path, 0.0) > budget
    ]
    lines = [f"{path}: {duration:.1f}s (budget {budget:.0f}s)" for path, duration in offenders]
    assert not offenders, (
        "Smoke-bearing modules exceeded the import/collection budget. Move expensive setup "
        "behind fixtures or re-tier the module:\n  " + "\n  ".join(lines)
    )


def _assigned_module_names(statement: ast.stmt) -> set[str]:
    """Return module names assigned by one top-level statement.

    Parameters
    ----------
    statement:
        Top-level syntax-tree statement.

    Returns
    -------
    set[str]
        Directly assigned names.
    """

    if isinstance(statement, ast.Assign):
        targets = statement.targets
    elif isinstance(statement, ast.AnnAssign):
        targets = [statement.target]
    else:
        return set()
    return {target.id for target in targets if isinstance(target, ast.Name)}


def _warn_once_declarations(package_root: Path) -> set[tuple[str, str]]:
    """Collect warn-once module-global declarations from TorchLens sources.

    Parameters
    ----------
    package_root:
        Root of the ``torchlens`` package.

    Returns
    -------
    set[tuple[str, str]]
        ``(module_name, attribute_name)`` declarations.
    """

    declarations: set[tuple[str, str]] = set()
    for path in package_root.rglob("*.py"):
        module_parts = path.relative_to(package_root.parent).with_suffix("").parts
        if module_parts[-1] == "__init__":
            module_parts = module_parts[:-1]
        module_name = ".".join(module_parts)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for statement in tree.body:
            for name in _assigned_module_names(statement):
                normalized = name.lower()
                if (
                    "warned" in normalized
                    or "warning_emitted" in normalized
                    or normalized.endswith("_warning_types")
                ):
                    declarations.add((module_name, name))
    return declarations


def test_warn_once_sentinel_census_matches_autouse_reset(
    request: pytest.FixtureRequest,
) -> None:
    """Every declared warn-once global must be isolated by the autouse fixture."""

    package_root = Path(__file__).resolve().parents[1] / "torchlens"
    discovered = _warn_once_declarations(package_root)
    configured_specs: tuple[tuple[str, str, object], ...] = (
        request.config._tl_warn_once_sentinel_specs
    )
    configured = {(module_name, name) for module_name, name, _default in configured_specs}
    runtime_only = {("torchlens.visualization._render_dot", "_SIBLING_ORDER_WARNING_EMITTED")}
    assert configured == discovered | runtime_only, (
        "Warn-once sentinel reset inventory drifted. Add/remove entries in "
        "tests/conftest.py::_WARN_ONCE_SENTINELS. "
        f"Missing resets: {sorted(discovered - configured)}; "
        f"stale resets: {sorted(configured - discovered - runtime_only)}"
    )


def _is_module_scoped_fixture(decorator: ast.expr) -> bool:
    """Return whether a decorator declares a long-lived pytest fixture.

    Any scope wider than the default per-test function scope ("class",
    "module", "package", "session") retains its Trace across tests, so all
    of them need the teardown gate — matching only the literal "module"
    left the wider scopes unguarded (b2p2 opus-B2P2-10 / sol-R77-2).

    Parameters
    ----------
    decorator:
        Function decorator syntax node.

    Returns
    -------
    bool
        Whether the decorator is ``pytest.fixture(scope=<non-function>)``.
    """

    if not isinstance(decorator, ast.Call):
        return False
    function = decorator.func
    is_fixture = (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id == "pytest"
        and function.attr == "fixture"
    ) or (isinstance(function, ast.Name) and function.id == "fixture")
    return is_fixture and any(
        keyword.arg == "scope"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value in {"class", "module", "package", "session"}
        for keyword in decorator.keywords
    )


def _calls_trace(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a fixture body directly calls a function named ``trace``.

    Parameters
    ----------
    function:
        Fixture function syntax node.

    Returns
    -------
    bool
        Whether a direct ``trace(...)`` or ``*.trace(...)`` call exists.
    """

    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "trace":
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr == "trace":
            return True
    return False


def _module_trace_fixtures_without_yield(tests_root: Path) -> list[str]:
    """Find module-scoped Trace fixtures that cannot perform teardown.

    Parameters
    ----------
    tests_root:
        Root of the test suite.

    Returns
    -------
    list[str]
        Stable ``path::fixture`` violations.
    """

    violations: list[str] = []
    for path in tests_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        functions = (
            node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        for function in functions:
            if not any(_is_module_scoped_fixture(item) for item in function.decorator_list):
                continue
            if not _calls_trace(function):
                continue
            if any(isinstance(node, (ast.Yield, ast.YieldFrom)) for node in ast.walk(function)):
                continue
            violations.append(f"{path.relative_to(tests_root)}::{function.name}")
    return violations


def test_module_scoped_trace_fixtures_have_teardown() -> None:
    """Module-scoped fixtures retaining Traces must yield for explicit cleanup."""

    tests_root = Path(__file__).resolve().parent
    violations = _module_trace_fixtures_without_yield(tests_root)
    assert not violations, (
        "Module-scoped fixtures create live Traces without a teardown path. Yield the Trace "
        "and call cleanup() in finally:\n  " + "\n  ".join(violations)
    )


def test_root_conftest_does_not_inject_repo_into_sys_path() -> None:
    """The suite must not hide an unusable editable install via path mutation."""

    conftest_path = Path(__file__).with_name("conftest.py")
    tree = ast.parse(conftest_path.read_text(encoding="utf-8"), filename=str(conftest_path))
    violations: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        is_sys_path = (
            isinstance(owner, ast.Attribute)
            and isinstance(owner.value, ast.Name)
            and owner.value.id == "sys"
            and owner.attr == "path"
        )
        if is_sys_path and node.func.attr in {"append", "extend", "insert"}:
            violations.append(node.lineno)
    assert not violations, (
        "tests/conftest.py mutates sys.path and can mask a broken installed distribution; "
        f"offending lines: {violations}"
    )


def test_root_tests_do_not_import_ambiguous_conftest_module() -> None:
    """Root tests must consume shared state without bare ``conftest`` imports."""

    tests_root = Path(__file__).resolve().parent
    violations: list[str] = []
    for path in tests_root.glob("test*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "conftest":
                violations.append(f"{path.name}:{node.lineno}")
    assert not violations, (
        "Root tests import the ambiguous bare `conftest` module; use the session output "
        f"environment or a real helper module instead: {violations}"
    )
