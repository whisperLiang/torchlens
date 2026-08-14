import os
import random
import sys
import time
import weakref
from collections.abc import Iterator
from os.path import join as opj
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch

from torchlens import _state

# Menagerie tests exercise the menagerie/ build subsystem, which is not importable
# on Python < 3.11 because it uses datetime.UTC. Skip collecting them on those
# interpreters so the core suite still runs in the documented smoke environment.
collect_ignore_glob = []
if sys.version_info < (3, 11):
    collect_ignore_glob.append("test_menagerie_*.py")
    collect_ignore_glob.append("crawler/*.py")

# Output directories are assigned under pytest's private basetemp in
# ``pytest_configure`` before test modules import these constants.
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_OUTPUTS_DIR = ""
REPORTS_DIR = ""
VIS_OUTPUT_DIR = ""

_MISSING = object()
_WARN_ONCE_SENTINELS: tuple[tuple[str, str, object], ...] = (
    # NOTE: this sentinel is a weakref.WeakSet in the package (type-keyed cache
    # eviction, F3a); resetting it to a plain set() would strong-pin model
    # classes — the reset must preserve the weak container type.
    ("torchlens._capture_state_helpers", "_VALIDATION_DEEPCOPY_WARNING_TYPES", weakref.WeakSet()),
    ("torchlens._capture_state_helpers", "_COMPILED_MODEL_UNWRAP_WARNED", False),
    ("torchlens._capture_state_helpers", "_COMPILED_FORCED_EAGER_WARNED", False),
    ("torchlens._deprecations", "_WARNED_DEPRECATIONS", set()),
    ("torchlens._io", "_LEGACY_THREAD_WARNING_EMITTED", {"flag": False}),
    ("torchlens._io.bundle", "_NONPERSISTENT_DISCLOSURE_WARNED", False),
    ("torchlens._io.bundle", "_UNATTESTABLE_ACTIVATION_DISCLOSURE_WARNED", False),
    ("torchlens._state", "_functorch_warning_emitted", False),
    ("torchlens._state", "_dynamo_warning_emitted", False),
    ("torchlens.backends.tf._tf_compat", "_warned_missing_capabilities", set()),
    ("torchlens.backends.torch.ops", "_UNSUPPORTED_OUTPUT_CONTAINER_WARNED", set()),
    ("torchlens.data_classes.op", "_WARNED_REFERENCE_SAVE_MODE", False),
    ("torchlens.distributed._lifecycle", "_AUTO_ARM_WARNED", False),
    ("torchlens.fastlog._storage_resolver", "_WARNED_REFERENCE_SAVE_MODE", False),
    ("torchlens.utils._torch_compat", "_warned_missing_capabilities", set()),
    ("torchlens.utils.introspection", "_col_offset_cache_warned", False),
    ("torchlens.validation._stock_layer_grads", "_PASS_INDEX_PARSE_WARNED", False),
    ("torchlens.visualization._render_common", "_SIBLING_ORDER_WARNING_EMITTED", False),
    ("torchlens.visualization._render_dot", "_SIBLING_ORDER_WARNING_EMITTED", False),
    ("torchlens.visualization.auto_collapse", "_COUNT_MISMATCH_WARNING_EMITTED", False),
)


# ---------------------------------------------------------------------------
# Coverage: auto-generate text report when pytest --cov is used
# ---------------------------------------------------------------------------


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    """Initialize session-private outputs with shipped usage stats disabled.

    Parameters
    ----------
    config:
        Active pytest configuration.
    """

    global TEST_OUTPUTS_DIR, REPORTS_DIR, VIS_OUTPUT_DIR

    output_root = config._tmp_path_factory.getbasetemp() / "torchlens-generated"
    TEST_OUTPUTS_DIR = str(output_root)
    REPORTS_DIR = str(output_root / "reports")
    VIS_OUTPUT_DIR = str(output_root / "visualizations")
    config._tl_prior_test_outputs_dir = os.environ.get("TORCHLENS_TEST_OUTPUTS_DIR")
    os.environ["TORCHLENS_TEST_OUTPUTS_DIR"] = TEST_OUTPUTS_DIR
    config._tl_warn_once_sentinel_specs = _WARN_ONCE_SENTINELS
    _state._collect_usage_stats = False
    _state._function_call_counts.clear()
    _state._function_call_models.clear()
    # Pay PyTorch's one-time RNG and deterministic-mode initialization during
    # session setup, not against whichever smoke test happens to run first.
    torch.random.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(deterministic, warn_only=deterministic_warn_only)


def pytest_unconfigure(config: pytest.Config) -> None:
    """Restore the caller's test-output environment after pytest exits.

    Parameters
    ----------
    config:
        Active pytest configuration.
    """

    prior = getattr(config, "_tl_prior_test_outputs_dir", None)
    if prior is None:
        os.environ.pop("TORCHLENS_TEST_OUTPUTS_DIR", None)
    else:
        os.environ["TORCHLENS_TEST_OUTPUTS_DIR"] = prior


# Smoke-tier duration budget (see tests/test_marker_lint.py). The PARTITION
# threshold for moving a test out of smoke is 5s measured; the ENFORCEMENT
# budget stays 15s (3x, load-scaled below) until every >5s smoke test is
# re-tiered — several measured 9.5-14.2s quiet on 2026-08-13. Lower to 5.0
# once that re-tier lands (tracked follow-up).
SMOKE_DURATION_BUDGET_SECONDS = 15.0


def _smoke_budget_load_factor() -> float:
    """Scale the wall-clock budget by CPU oversubscription at measurement time.

    Wall-clock durations inflate roughly with run-queue pressure; a fixed
    budget false-trips whenever an orchestrator runs sibling lanes on the same
    box (measured 2026-08-13: the same four tests read 2.9-14.2s quiet but
    15.6-34.9s at loadavg ~5x nproc). Capped so a pathological load reading
    can never disarm the lint entirely.
    """

    try:
        load_per_cpu = os.getloadavg()[0] / max(os.cpu_count() or 1, 1)
    except OSError:  # pragma: no cover - getloadavg unsupported on the platform.
        return 1.0
    return min(max(load_per_cpu, 1.0), 4.0)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[Any]
) -> Iterator[pytest.TestReport]:
    """Record smoke tests whose full setup/call/teardown exceeds the budget.

    A static lint cannot know runtimes, so an UNMARKED slow test landing in the
    smoke tier is only catchable at runtime. Offenders are stashed on the session
    and asserted empty by ``test_marker_lint.py`` (ordered last), which names each
    offender and its measured duration.
    """

    report = yield
    durations = getattr(item, "_tl_phase_durations", None)
    if durations is None:
        durations = {}
        item._tl_phase_durations = durations
    durations[report.when] = report.duration
    if report.when == "teardown" and item.get_closest_marker("smoke") is not None:
        total_duration = sum(durations.values())
        family = getattr(item, "originalname", None) or item.name.split("[")[0]
        family_totals = getattr(item.session, "_tl_smoke_family_durations", None)
        if family_totals is None:
            family_totals = {}
            item.session._tl_smoke_family_durations = family_totals
        family_key = f"{item.path}::{family}"
        family_totals[family_key] = family_totals.get(family_key, 0.0) + total_duration
    else:
        total_duration = 0.0
    budget = SMOKE_DURATION_BUDGET_SECONDS * _smoke_budget_load_factor()
    item.session._tl_smoke_budget_value = budget
    # A parametrized family's aggregate legitimately scales with its parameter
    # count (the 278-cell selector matrix costs ~16s at 57ms/cell on a quiet
    # box) — give aggregates 2x the per-test budget; genuine family ballooning
    # still trips at that bar. Tightens with the 5s re-tier follow-up.
    item.session._tl_smoke_family_budget_value = budget * 2.0
    if total_duration > budget:
        offenders = getattr(item.session, "_tl_smoke_budget_offenders", None)
        if offenders is None:
            offenders = []
            item.session._tl_smoke_budget_offenders = offenders
        offenders.append((item.nodeid, total_duration, budget))
    return report


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(
    collector: pytest.Collector,
) -> Iterator[pytest.CollectReport]:
    """Record test-module import and collection time for smoke-tier enforcement.

    Parameters
    ----------
    collector:
        Collector whose work is about to run.
    """

    started = time.perf_counter()
    report = yield
    if isinstance(collector, pytest.Module):
        durations = getattr(collector.session, "_tl_module_collection_durations", None)
        if durations is None:
            durations = {}
            collector.session._tl_module_collection_durations = durations
        durations[str(collector.path)] = time.perf_counter() - started
    return report


def _is_full_usage_stats_run(config: pytest.Config) -> bool:
    """Return whether pytest selected the complete default non-rare suite.

    Parameters
    ----------
    config:
        Active pytest configuration.

    Returns
    -------
    bool
        ``True`` only for an unfiltered invocation rooted at ``tests/``.
    """

    if config.option.keyword:
        return False
    mark_expression = (config.option.markexpr or "").strip().replace("(", "").replace(")", "")
    if mark_expression not in {"", "not rare"}:
        return False
    requested_paths = [Path(str(arg).split("::", maxsplit=1)[0]).resolve() for arg in config.args]
    return requested_paths == [Path(TESTS_DIR).resolve()]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Order the ArgSpec coverage test last; skip assertion-dependent tests under -O.

    ``python -O`` strips every ``assert``, which disables the postprocess contract
    audit outright -- arming it then raises rather than silently reporting a clean
    audit that verified nothing. Tests that arm the audit therefore cannot run
    under ``-O`` and must SKIP, not fail, so the ``-O`` leg stays a meaningful
    verdict-identity check on everything else.
    """

    if not __debug__:
        skip_no_assertions = pytest.mark.skip(
            reason="requires assertions; the postprocess audit cannot run under python -O"
        )
        for item in items:
            if item.get_closest_marker("requires_assertions") is not None:
                item.add_marker(skip_no_assertions)

    collect_usage_stats = _is_full_usage_stats_run(config)
    _state._collect_usage_stats = collect_usage_stats
    if collect_usage_stats:
        _state._function_call_counts.clear()
        _state._function_call_models.clear()

    skip_partial_usage_stats = pytest.mark.skip(
        reason="ArgSpec usage coverage requires an unfiltered full tests/ run"
    )
    coverage_tests = []
    lint_tests = []
    other_tests = []
    for item in items:
        if "test_arg_positions" in item.nodeid:
            if not collect_usage_stats:
                item.add_marker(skip_partial_usage_stats)
            coverage_tests.append(item)
        elif "test_marker_lint" in item.nodeid:
            # The duration-budget lint reads offenders recorded during the run,
            # so it must execute after every other test in the session.
            lint_tests.append(item)
        else:
            other_tests.append(item)
    items[:] = other_tests + coverage_tests + lint_tests


def _coverage_requested(config: pytest.Config) -> bool:
    """Return whether this session explicitly requested pytest-cov collection.

    Parameters
    ----------
    config:
        Active pytest configuration.

    Returns
    -------
    bool
        ``True`` when pytest-cov is active for this session.
    """

    cov_source = getattr(config.option, "cov_source", None)
    return bool(cov_source)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write coverage artifacts only for real coverage runs.

    Parameters
    ----------
    session:
        Active pytest session.
    exitstatus:
        Final pytest exit status.
    """

    del exitstatus
    _state._collect_usage_stats = False
    _state._function_call_counts.clear()
    _state._function_call_models.clear()
    config = session.config
    if config.option.collectonly or not _coverage_requested(config):
        return
    try:
        from coverage import Coverage
        from coverage.exceptions import NoDataError
    except ImportError:
        return

    try:
        cov = Coverage()
        cov.load()
        report_path = opj(REPORTS_DIR, "coverage_report.txt")
        with open(report_path, "w") as f:
            cov.report(file=f, show_missing=True, skip_empty=True)
        html_dir = opj(REPORTS_DIR, "coverage_html")
        cov.html_report(directory=html_dir, skip_empty=True)
    except (FileNotFoundError, NoDataError):
        return


# Fixtures


@pytest.fixture(autouse=True, scope="session")
def _isolated_torchlens_cache(tmp_path_factory: pytest.TempPathFactory):
    """Point ``TORCHLENS_CACHE_DIR`` at a fresh per-session directory.

    ``tl.trace(..., cache=True)`` without an explicit ``cache_dir=`` falls back to
    ``~/.cache/torchlens`` (read lazily from ``TORCHLENS_CACHE_DIR`` on every call).
    A shared on-disk cache makes cache-hit assertions order- and history-dependent:
    a stale entry left by an earlier session or another worktree turns a
    first-capture cache-miss assertion into a phantom failure. Every test session
    gets its own empty cache root instead; the prior environment is restored on
    teardown so the suite never leaks state into the invoking shell.
    """

    prior = os.environ.get("TORCHLENS_CACHE_DIR")
    os.environ["TORCHLENS_CACHE_DIR"] = str(tmp_path_factory.mktemp("torchlens_cache"))
    yield
    if prior is None:
        os.environ.pop("TORCHLENS_CACHE_DIR", None)
    else:
        os.environ["TORCHLENS_CACHE_DIR"] = prior


def _copy_sentinel_value(value: object) -> object:
    """Return an independent snapshot of a supported warn-once value.

    Parameters
    ----------
    value:
        Boolean, set, or dictionary sentinel value.

    Returns
    -------
    object
        Independent mutable copy, or the original immutable value.
    """

    if isinstance(value, (set, dict, weakref.WeakSet)):
        return value.copy()
    return value


def _set_sentinel_default(module: ModuleType, name: str, default: object) -> None:
    """Reset one imported warn-once sentinel to its cold-process value.

    Parameters
    ----------
    module:
        Imported module that owns the sentinel.
    name:
        Module attribute name.
    default:
        Cold-process sentinel value.
    """

    setattr(module, name, _copy_sentinel_value(default))


@pytest.fixture(autouse=True)
def _reset_warn_once_sentinels() -> Iterator[None]:
    """Give every test fresh warn-once state and restore its incoming state."""

    snapshots: dict[tuple[str, str], object] = {}
    for module_name, name, default in _WARN_ONCE_SENTINELS:
        module = sys.modules.get(module_name)
        if module is None:
            snapshots[(module_name, name)] = _MISSING
            continue
        snapshots[(module_name, name)] = _copy_sentinel_value(
            getattr(module, name, _MISSING)
        )
        _set_sentinel_default(module, name, default)

    try:
        yield
    finally:
        for module_name, name, default in _WARN_ONCE_SENTINELS:
            module = sys.modules.get(module_name)
            if module is None:
                continue
            prior = snapshots[(module_name, name)]
            if prior is _MISSING:
                _set_sentinel_default(module, name, default)
            else:
                setattr(module, name, prior)


@pytest.fixture(autouse=True)
def _reset_rng_state() -> Iterator[None]:
    """Seed each test deterministically and restore all incoming RNG settings."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_initialized = torch.cuda.is_initialized()
    cuda_states = torch.cuda.get_rng_state_all() if cuda_initialized else None
    deterministic = torch.are_deterministic_algorithms_enabled()
    deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if cuda_initialized:
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)

    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.use_deterministic_algorithms(deterministic, warn_only=deterministic_warn_only)


@pytest.fixture
def default_input1():
    return torch.rand(6, 3, 224, 224)


@pytest.fixture
def default_input2():
    return torch.rand(6, 3, 224, 224)


@pytest.fixture
def default_input3():
    return torch.rand(6, 3, 224, 224)


@pytest.fixture
def default_input4():
    return torch.rand(6, 3, 224, 224)


@pytest.fixture
def zeros_input():
    return torch.zeros(6, 3, 224, 224)


@pytest.fixture
def ones_input():
    return torch.ones(6, 3, 224, 224)


@pytest.fixture
def vector_input():
    return torch.rand(5)


@pytest.fixture
def input_2d():
    return torch.rand(5, 5)


@pytest.fixture
def input_complex():
    return (torch.complex(torch.rand(3, 3), torch.rand(3, 3)),)


@pytest.fixture
def small_input():
    return torch.rand(2, 3, 32, 32)


@pytest.fixture
def seq_input():
    """(seq_len, batch, embed_dim) for transformer models."""
    return torch.rand(10, 2, 16)


@pytest.fixture
def token_input():
    """Integer tokens for embedding models."""
    return torch.randint(0, 100, (2, 10))


@pytest.fixture
def input_3d():
    """Volumetric input for Conv3d models."""
    return torch.rand(1, 1, 4, 4, 4)


@pytest.fixture
def input_1d_seq():
    """1D sequence input for Conv1d models."""
    return torch.rand(2, 3, 16)
