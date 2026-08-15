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


def test_bounded_tier_tests_stay_within_duration_budget(
    request: pytest.FixtureRequest,
) -> None:
    """Every bounded-tier test must finish within its tier's duration budget.

    The budget is TWO-directional (R41): ``smoke`` AND unmarked tests are held
    to the 5s partition boundary, ``heavy`` to its 20s ceiling (all
    load-scaled); ``slow``/``rare``/``serial`` are exempt by contract. Budget
    values live in ``tests/conftest.py`` and ride along on each recorded
    offender -- a bare ``conftest`` import here would be ambiguous during
    full-suite collection (nested conftests share the module name).

    This test also asserts its own LAST-position ordering: the offender
    ledger only covers tests that already ran, so a reordering regression
    (e.g. a plugin shuffling after the conftest reorder) must go red here
    rather than silently truncating coverage.
    """

    items = request.session.items
    own_index = next(
        index for index, item in enumerate(items) if item.nodeid == request.node.nodeid
    )
    stragglers = [
        item.nodeid for item in items[own_index + 1 :] if "test_marker_lint" not in item.nodeid
    ]
    assert not stragglers, (
        "the duration-budget lint no longer runs last -- its offender ledger "
        f"would miss these later tests: {stragglers[:5]}"
    )

    guidance = {
        "smoke": "re-tier to `heavy` (5-20s) or `slow` (>20s), or make it faster",
        "unmarked": "unmarked tests run in the mid backstop: add `heavy`/`slow` "
        "consciously, or make it faster",
        "heavy": "re-tier to `slow` (>20s) or make it faster",
    }
    offenders = getattr(request.session, "_tl_duration_budget_offenders", [])
    lines = [
        f"{nodeid} [{tier}]: {duration:.1f}s (budget {budget:.0f}s) -- {guidance[tier]}"
        for nodeid, tier, duration, budget in offenders
    ]
    assert not offenders, (
        "Tests exceeded their tier duration budget this session:\n  " + "\n  ".join(lines)
    )


def test_smoke_parametrized_families_stay_within_duration_budget(
    request: pytest.FixtureRequest,
) -> None:
    """Resolved smoke parameter families must stay within the aggregate budget.

    A family of N parameters legitimately costs ~N single-test durations (the
    selector matrix is 278 cells), so each family's budget scales with its
    resolved cell count: load_factor * max(2x the per-test budget, the
    per-cell allowance x n_cells). Genuine per-cell ballooning still trips.
    """

    family_stats = getattr(request.session, "_tl_smoke_family_stats", {})
    family_budgets = getattr(request.session, "_tl_smoke_family_budgets", {})
    offenders = [
        (family, total, count, family_budgets.get(family, 0.0))
        for family, (total, count) in family_stats.items()
        if total > family_budgets.get(family, float("inf"))
    ]
    lines = [
        f"{family}: {total:.1f}s over {count} cells (budget {budget:.0f}s)"
        for family, total, count, budget in offenders
    ]
    assert not offenders, (
        "Smoke parametrized families exceeded their aggregate cell-scaled "
        "budget. Split or re-tier the family:\n  " + "\n  ".join(lines)
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
    # Behavioral fidelity latches the NAME heuristic cannot discover (nothing
    # "warned"-shaped in the identifier), declared here explicitly so the two
    # ledgers (this census and the conftest reset list) can no longer disagree
    # silently (grind p5, B2P3-16 / sol R76-2). A test that trips one of these
    # degrades every later test in the session, so the reset is REQUIRED.
    sticky_latches = {
        ("torchlens.utils.rng", "_cuda_rng_unusable"),
        # Last-degradation record for lazy auto-arm (fix/distributed-r4): not
        # "warned"-shaped, but a test that degrades arming would otherwise
        # leak its reason into every later auto_arm_degradation() read.
        ("torchlens.distributed._lifecycle", "_AUTO_ARM_DEGRADATION"),
    }
    expected = discovered | runtime_only | sticky_latches
    assert configured == expected, (
        "Warn-once sentinel reset inventory drifted. Add/remove entries in "
        "tests/conftest.py::_WARN_ONCE_SENTINELS. "
        f"Missing resets: {sorted(expected - configured)}; "
        f"stale resets: {sorted(configured - expected)}"
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


_REGISTRY_MUTATOR_NAMES = frozenset({"register_container", "unregister_container"})
"""Public registry mutators whose import-time call is an order-dependence bug."""


def _import_time_nodes(tree: ast.Module) -> list[ast.AST]:
    """Return every node that executes when the module is IMPORTED.

    Function/lambda bodies run only when called, so they are skipped -- but
    their decorators DO run at import and stay included. Class bodies execute
    at import and are walked.
    """

    nodes: list[ast.AST] = []
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            stack.extend(getattr(node, "decorator_list", []))
            continue
        nodes.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return nodes


def _facet_register_aliases(tree: ast.Module) -> set[str]:
    """Names under which the facet ``register`` decorator is imported bare."""

    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("facets"):
            for imported in node.names:
                if imported.name == "register":
                    aliases.add(imported.asname or imported.name)
    return aliases


def test_no_module_level_registry_mutation_in_tests() -> None:
    """Test modules must not mutate public registries at IMPORT time.

    A module-level ``@tl.facets.register`` or ``tl.register_container(...)``
    fires at pytest COLLECTION -- before any fixture can isolate it -- so a
    full collection left extra recipes/containers in the process-global
    registry for the whole session while a targeted run did not: the same
    trace hashed different recipe/provenance state depending on how pytest
    was invoked (hunt-b2-sol R76/R77). Register inside a restoring fixture.

    Covers all three call spellings (grind p5 §3.9: the original check saw
    only the ``x.facets.register`` decorator form): attribute decorators,
    BARE-NAME decorators (``from ...facets import register``), and
    import-time ``register_container``/``unregister_container`` calls whether
    attribute-qualified or bare.
    """

    tests_root = Path(__file__).resolve().parent
    violations: list[str] = []
    for path in tests_root.rglob("test*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        facet_aliases = _facet_register_aliases(tree)
        for node in _import_time_nodes(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_violation = False
            if isinstance(func, ast.Attribute):
                if func.attr in _REGISTRY_MUTATOR_NAMES:
                    is_violation = True
                elif func.attr == "register":
                    base = func.value
                    if isinstance(base, ast.Attribute) and base.attr == "facets":
                        is_violation = True
            elif isinstance(func, ast.Name):
                if func.id in _REGISTRY_MUTATOR_NAMES or func.id in facet_aliases:
                    is_violation = True
            if is_violation:
                violations.append(f"{path.relative_to(tests_root)}:{node.lineno}")
    assert not violations, (
        "import-time registry mutation (facet register / register_container) runs at "
        f"pytest collection; register inside a restoring fixture: {sorted(violations)}"
    )


# ---------------------------------------------------------------------------
# R77 round-3: strengthened teardown lint for long-lived Trace fixtures.
#
# The original scanner (`_module_trace_fixtures_without_yield`) has three
# proven evasion shapes: (a) fixtures nested inside test classes are invisible
# (it only reads `tree.body`); (b) a fixture that builds its Trace through a
# module-local helper (`make_trace()` -> `tl.trace(...)`) is invisible (only
# literal `trace(...)` calls are matched); (c) a fixture with a BARE trailing
# `yield` -- no statement after it and no try/finally -- passes despite having
# no teardown code at all. The scanner below closes all three. Indirection
# through helpers is resolved transitively but only within the SAME module;
# cross-module helper indirection is documented out of scope.
# ---------------------------------------------------------------------------


def _iter_scoped_fixture_functions(
    tree: ast.Module,
) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Collect widely-scoped fixtures at module level and inside classes.

    Parameters
    ----------
    tree:
        Parsed test module.

    Returns
    -------
    list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]
        ``(qualified_name, function)`` pairs for every fixture whose scope is
        wider than per-test function scope, including fixtures nested inside
        (arbitrarily nested) test classes.
    """

    found: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []

    def visit(body: list[ast.stmt], prefix: str) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(_is_module_scoped_fixture(item) for item in node.decorator_list):
                    found.append((f"{prefix}{node.name}", node))
            elif isinstance(node, ast.ClassDef):
                visit(node.body, f"{prefix}{node.name}.")

    visit(tree.body, "")
    return found


def _module_local_trace_helper_names(tree: ast.Module) -> set[str]:
    """Resolve module-level helper functions that (transitively) call trace.

    Parameters
    ----------
    tree:
        Parsed test module.

    Returns
    -------
    set[str]
        Names of module-level functions whose bodies reach a ``trace(...)`` /
        ``*.trace(...)`` call, directly or through other module-level helpers
        (fixed point within the module; cross-module helpers are out of scope).
    """

    module_functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    trace_callers = {name for name, node in module_functions.items() if _calls_trace(node)}
    changed = True
    while changed:
        changed = False
        for name, node in module_functions.items():
            if name in trace_callers:
                continue
            for call in ast.walk(node):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id in trace_callers
                ):
                    trace_callers.add(name)
                    changed = True
                    break
    return trace_callers


def _calls_trace_or_local_helper(
    function: ast.FunctionDef | ast.AsyncFunctionDef, helper_names: set[str]
) -> bool:
    """Return whether a fixture reaches a trace call directly or via helpers.

    Parameters
    ----------
    function:
        Fixture function syntax node.
    helper_names:
        Module-level helper functions known to (transitively) call trace.

    Returns
    -------
    bool
        Whether the fixture creates a Trace through any in-module path.
    """

    if _calls_trace(function):
        return True
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in helper_names
        ):
            return True
    return False


def _block_has_unguarded_yield(statements: list[ast.stmt], guarded: bool) -> bool:
    """Return whether any yield in a statement block lacks a teardown path.

    A yield is *guarded* when some enclosing block (within the fixture) has at
    least one statement after the statement containing it, or when it sits
    inside the body/handlers/orelse of a ``try`` with a ``finally`` clause.
    Yields inside nested function/class definitions belong to those objects,
    not to the fixture, and are skipped.

    Parameters
    ----------
    statements:
        Statement block to scan.
    guarded:
        Whether an enclosing construct already guarantees teardown.

    Returns
    -------
    bool
        Whether an unguarded (teardown-free) yield exists in the block.
    """

    for index, statement in enumerate(statements):
        followed = guarded or index + 1 < len(statements)
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(statement, ast.Try):
            inner = followed or bool(statement.finalbody)
            blocks = [statement.body, statement.orelse]
            blocks.extend(handler.body for handler in statement.handlers)
            if any(_block_has_unguarded_yield(block, inner) for block in blocks):
                return True
            if _block_has_unguarded_yield(statement.finalbody, followed):
                return True
            continue
        nested_blocks = [
            value
            for _field, value in ast.iter_fields(statement)
            if isinstance(value, list) and value and isinstance(value[0], ast.stmt)
        ]
        if nested_blocks:
            if any(_block_has_unguarded_yield(block, followed) for block in nested_blocks):
                return True
            continue
        if not followed and any(
            isinstance(node, (ast.Yield, ast.YieldFrom)) for node in ast.walk(statement)
        ):
            return True
    return False


def _fixture_has_real_teardown(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a fixture owns an actual teardown path, not a bare yield.

    Parameters
    ----------
    function:
        Fixture function syntax node.

    Returns
    -------
    bool
        ``True`` when the fixture yields with at least one statement after the
        yield (or a try/finally around it), or registers a finalizer through
        ``request.addfinalizer(...)``.
    """

    for node in ast.walk(function):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "addfinalizer"
        ):
            return True
    has_yield = _block_contains_yield(function.body)
    return has_yield and not _block_has_unguarded_yield(function.body, False)


def _block_contains_yield(statements: list[ast.stmt]) -> bool:
    """Return whether a block yields, ignoring nested function/class bodies.

    Parameters
    ----------
    statements:
        Statement block to scan.

    Returns
    -------
    bool
        Whether the block contains a yield belonging to the enclosing function.
    """

    for statement in statements:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        nested_blocks = [
            value
            for _field, value in ast.iter_fields(statement)
            if isinstance(value, list) and value and isinstance(value[0], ast.stmt)
        ]
        if isinstance(statement, ast.Try):
            nested_blocks.extend(handler.body for handler in statement.handlers)
        if nested_blocks:
            if any(_block_contains_yield(block) for block in nested_blocks):
                return True
            continue
        if any(isinstance(node, (ast.Yield, ast.YieldFrom)) for node in ast.walk(statement)):
            return True
    return False


def _strict_trace_fixture_violations_in_source(source: str, label: str) -> list[str]:
    """Scan one module's source for teardown-free long-lived Trace fixtures.

    Parameters
    ----------
    source:
        Python source text of a test module.
    label:
        Stable label (relative path) used in violation strings.

    Returns
    -------
    list[str]
        ``label::qualified_fixture_name`` violations.
    """

    tree = ast.parse(source, filename=label)
    helper_names = _module_local_trace_helper_names(tree)
    violations: list[str] = []
    for qualified_name, function in _iter_scoped_fixture_functions(tree):
        if not _calls_trace_or_local_helper(function, helper_names):
            continue
        if not _fixture_has_real_teardown(function):
            violations.append(f"{label}::{qualified_name}")
    return violations


def _strict_trace_fixture_violations(tests_root: Path) -> list[str]:
    """Scan the whole test suite for teardown-free long-lived Trace fixtures.

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
    for path in sorted(tests_root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        violations.extend(
            _strict_trace_fixture_violations_in_source(source, str(path.relative_to(tests_root)))
        )
    return violations


def test_scoped_trace_fixtures_have_real_teardown() -> None:
    """Widely-scoped Trace fixtures need actual teardown code, however nested."""

    tests_root = Path(__file__).resolve().parent
    violations = _strict_trace_fixture_violations(tests_root)
    assert not violations, (
        "Widely-scoped fixtures create live Traces without a real teardown path "
        "(class-nested fixtures, helper-built traces, and bare trailing yields "
        "all count). Yield the Trace and clean up after the yield (or in a "
        "try/finally):\n  " + "\n  ".join(violations)
    )


_EVASION_CLASS_NESTED = """
import pytest
import torchlens as tl

class TestGroup:
    @pytest.fixture(scope="module")
    def cached_trace(self):
        return tl.trace(model, x)
"""

_EVASION_HELPER_INDIRECTION = """
import pytest
import torchlens as tl

def make_trace():
    return tl.trace(model, x)

def build_log():
    return make_trace()

@pytest.fixture(scope="module")
def cached_trace():
    return build_log()
"""

_EVASION_BARE_YIELD = """
import pytest
import torchlens as tl

@pytest.fixture(scope="session")
def cached_trace():
    yield tl.trace(model, x)
"""

_COMPLIANT_STATEMENT_AFTER_YIELD = """
import pytest
import torchlens as tl

@pytest.fixture(scope="module")
def cached_trace():
    log = tl.trace(model, x)
    yield log
    log.cleanup()
"""

_COMPLIANT_TRY_FINALLY = """
import pytest
import torchlens as tl

def make_trace():
    return tl.trace(model, x)

class TestGroup:
    @pytest.fixture(scope="class")
    def cached_trace(self):
        log = make_trace()
        try:
            yield log
        finally:
            log.cleanup()
"""

_COMPLIANT_FUNCTION_SCOPE_BARE_YIELD = """
import pytest
import torchlens as tl

@pytest.fixture()
def per_test_trace():
    yield tl.trace(model, x)
"""


@pytest.mark.parametrize(
    ("snippet", "expected"),
    [
        pytest.param(
            _EVASION_CLASS_NESTED, ["planted.py::TestGroup.cached_trace"], id="class-nested"
        ),
        pytest.param(
            _EVASION_HELPER_INDIRECTION, ["planted.py::cached_trace"], id="helper-indirection"
        ),
        pytest.param(_EVASION_BARE_YIELD, ["planted.py::cached_trace"], id="bare-yield"),
        pytest.param(_COMPLIANT_STATEMENT_AFTER_YIELD, [], id="ok-statement-after-yield"),
        pytest.param(_COMPLIANT_TRY_FINALLY, [], id="ok-try-finally-class-nested-helper"),
        pytest.param(_COMPLIANT_FUNCTION_SCOPE_BARE_YIELD, [], id="ok-function-scope"),
    ],
)
def test_strict_trace_fixture_scanner_is_red_capable(snippet: str, expected: list[str]) -> None:
    """The strengthened scanner catches each proven evasion shape exactly.

    Red-capability proof for the three R77 evasions: (a) class-nested
    fixtures, (b) one-or-more-hop in-module helper indirection to the trace
    call, (c) a bare trailing yield with no teardown statement. The compliant
    shapes prove the scanner does not overfire.
    """

    assert _strict_trace_fixture_violations_in_source(snippet, "planted.py") == expected


def _lru_cached_functions(package_root: Path) -> dict[tuple[str, str], str]:
    """Collect ``lru_cache``/``cache``-decorated module functions and their source.

    Parameters
    ----------
    package_root:
        Root of the ``torchlens`` package.

    Returns
    -------
    dict[tuple[str, str], str]
        ``(module_name, function_name)`` -> function source segment.
    """

    cached: dict[tuple[str, str], str] = {}
    for path in package_root.rglob("*.py"):
        module_parts = path.relative_to(package_root.parent).with_suffix("").parts
        if module_parts[-1] == "__init__":
            module_parts = module_parts[:-1]
        module_name = ".".join(module_parts)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for statement in tree.body:
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorated = ast.unparse(statement.decorator_list) if statement.decorator_list else ""
            if "lru_cache" in decorated or "functools.cache" in decorated:
                cached[(module_name, statement.name)] = (
                    ast.get_source_segment(source, statement) or ""
                )
    return cached


def test_capability_dependent_caches_are_cleared() -> None:
    """Every probe-derived lru_cache must be in the conftest clear list.

    Restoring the lazy ``HAS_*`` capability latches un-poisons the b7fe953e
    incident class at the first layer only: an ``lru_cache`` whose value was
    computed FROM a poisoned probe keeps the poisoned result for the process
    (grind p5 §3.9, the class one layer down). This census flags every
    module-level cached function that references ``_torch_compat`` / a
    ``HAS_*`` flag -- directly or through another flagged cache -- and
    requires it in ``tests/conftest.py::_CAPABILITY_DEPENDENT_CACHES`` so the
    autouse probe-restore clears it. Deliberately process-frozen caches
    (torch-version-fixed inventories) do not reference probes and stay out.
    """

    import re

    from tests.conftest import _CAPABILITY_DEPENDENT_CACHES

    package_root = Path(__file__).resolve().parents[1] / "torchlens"
    cached = _lru_cached_functions(package_root)
    probe_pattern = re.compile(r"_torch_compat|\bHAS_[A-Z_]+\b")
    dependent: set[tuple[str, str]] = {
        key for key, body in cached.items() if probe_pattern.search(body)
    }
    # Fixpoint: a cache calling another dependent cache is dependent too.
    while True:
        names = {name for _, name in dependent}
        grown = dependent | {
            key
            for key, body in cached.items()
            if key not in dependent and any(re.search(rf"\b{name}\s*\(", body) for name in names)
        }
        if grown == dependent:
            break
        dependent = grown
    declared = set(_CAPABILITY_DEPENDENT_CACHES)
    assert dependent <= declared, (
        "lru_cached functions derive from a capability probe but are missing from "
        "tests/conftest.py::_CAPABILITY_DEPENDENT_CACHES (the probe restore cannot "
        f"clear them): {sorted(dependent - declared)}"
    )
    assert declared <= set(cached), (
        "stale _CAPABILITY_DEPENDENT_CACHES rows (no such cached function): "
        f"{sorted(declared - set(cached))}"
    )


def test_capability_dependent_cache_clear_actually_clears() -> None:
    """The conftest clear helper empties every declared probe-derived cache."""

    import importlib

    from tests.conftest import _CAPABILITY_DEPENDENT_CACHES, _clear_capability_dependent_caches

    primed = []
    for module_name, attr in _CAPABILITY_DEPENDENT_CACHES:
        function = getattr(importlib.import_module(module_name), attr)
        function()  # prime
        assert function.cache_info().currsize >= 1
        primed.append(function)
    _clear_capability_dependent_caches()
    for function in primed:
        assert function.cache_info().currsize == 0, f"{function} survived the probe restore"
