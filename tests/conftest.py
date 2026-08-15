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
    ("torchlens.distributed._lifecycle", "_AUTO_ARM_DEGRADATION", None),
    ("torchlens.fastlog._storage_resolver", "_WARNED_REFERENCE_SAVE_MODE", False),
    ("torchlens.postprocess.ast_branches", "_source_drift_warned", set()),
    ("torchlens.utils._torch_compat", "_warned_missing_capabilities", set()),
    # Capture-scoped CUDA RNG retry latch (grind p5, B2P3-16): re-armed per
    # capture in the package, and reset per test here so a test that trips it
    # via a direct log_current_rng_states() call cannot degrade later tests.
    ("torchlens.utils.rng", "_cuda_rng_unusable", False),
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

    # A CI run must never mutate goldens: any armed update/regen/record flag
    # would silently rebaseline instead of verifying (b7 R53-3 / b10 R78-8).
    from _oracle_env import golden_mutation_flags_armed_under_ci

    armed = golden_mutation_flags_armed_under_ci(os.environ)
    if armed:
        raise pytest.UsageError(
            "golden update/regen flags are forbidden under CI: " + ", ".join(armed)
        )

    output_root = config._tmp_path_factory.getbasetemp() / "torchlens-generated"
    TEST_OUTPUTS_DIR = str(output_root)
    REPORTS_DIR = str(output_root / "reports")
    VIS_OUTPUT_DIR = str(output_root / "visualizations")
    config._tl_prior_test_outputs_dir = os.environ.get("TORCHLENS_TEST_OUTPUTS_DIR")
    os.environ["TORCHLENS_TEST_OUTPUTS_DIR"] = TEST_OUTPUTS_DIR
    # Arm the collapse/sibling-order verification tripwire for OUR suite.
    # The library no longer keys strictness on the ambient PYTEST_CURRENT_TEST
    # marker (r3 b7-opus R47-A: a downstream project's pytest rendering a
    # TorchLens graph must never inherit our hard-assert mode), so the
    # TorchLens suite opts in through the torchlens-owned knob explicitly.
    config._tl_prior_collapse_strict = os.environ.get("TORCHLENS_COLLAPSE_STRICT")
    os.environ.setdefault("TORCHLENS_COLLAPSE_STRICT", "1")
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
    prior_strict = getattr(config, "_tl_prior_collapse_strict", None)
    if prior_strict is None:
        os.environ.pop("TORCHLENS_COLLAPSE_STRICT", None)
    else:
        os.environ["TORCHLENS_COLLAPSE_STRICT"] = prior_strict


# Tier duration budgets (see tests/test_marker_lint.py). The r3 re-tier
# landed, so the ENFORCEMENT budget now equals the documented PARTITION
# boundary: smoke and unmarked tests must fit 5s, heavy 20s (each load-scaled
# below; the pre-r3 15s crutch let a 59s test stay smoke under sprint load —
# R41 b2 opus+sol). `slow` is unbounded, `rare` only runs on request, and
# `serial` is exempt by definition (its wall time under parallel load is
# exactly what the marker declares unrepresentative).
SMOKE_DURATION_BUDGET_SECONDS = 5.0
HEAVY_DURATION_BUDGET_SECONDS = 20.0
#: Per-parametrize-cell allowance for a smoke family's aggregate budget: a
#: family's cost legitimately scales with its cell count (278 selector cells
#: at ~57ms/cell), so the aggregate bar is max(2x the per-test budget,
#: this allowance x n_cells) — genuine per-cell ballooning still trips it.
SMOKE_FAMILY_PER_CELL_SECONDS = 0.1


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


def _duration_budget_tier(item: pytest.Item) -> tuple[str, float] | None:
    """Return the duration-budget tier for one collected item.

    The budget is TWO-directional (b2 3-lab: 217-391 unmarked files escaped
    it entirely): an UNMARKED test runs in the mid/phase backstops, so it is
    held to the same 5s partition boundary as smoke — if it needs longer it
    needs a `heavy` or `slow` marker, chosen consciously.

    Parameters
    ----------
    item:
        Collected test item.

    Returns
    -------
    tuple[str, float] | None
        ``(tier_name, base_budget_seconds)``, or ``None`` for exempt tiers
        (``slow`` unbounded, ``rare`` request-only, ``serial`` load-exempt).
    """

    for exempt in ("slow", "rare", "serial"):
        if item.get_closest_marker(exempt) is not None:
            return None
    if item.get_closest_marker("heavy") is not None:
        return ("heavy", HEAVY_DURATION_BUDGET_SECONDS)
    if item.get_closest_marker("smoke") is not None:
        return ("smoke", SMOKE_DURATION_BUDGET_SECONDS)
    return ("unmarked", SMOKE_DURATION_BUDGET_SECONDS)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[Any]
) -> Iterator[pytest.TestReport]:
    """Record tests whose full setup/call/teardown exceeds their tier budget.

    A static lint cannot know runtimes, so a slow test landing in a bounded
    tier is only catchable at runtime. Offenders are stashed on the session
    and asserted empty by ``test_marker_lint.py`` (ordered last), which names
    each offender, its tier, and its measured duration.
    """

    report = yield
    durations = getattr(item, "_tl_phase_durations", None)
    if durations is None:
        durations = {}
        item._tl_phase_durations = durations
    durations[report.when] = report.duration
    if report.when != "teardown":
        return report
    total_duration = sum(durations.values())
    load_factor = _smoke_budget_load_factor()
    if item.get_closest_marker("smoke") is not None:
        family = getattr(item, "originalname", None) or item.name.split("[")[0]
        family_stats = getattr(item.session, "_tl_smoke_family_stats", None)
        if family_stats is None:
            family_stats = {}
            item.session._tl_smoke_family_stats = family_stats
        family_key = f"{item.path}::{family}"
        family_total, family_count = family_stats.get(family_key, (0.0, 0))
        family_total += total_duration
        family_count += 1
        family_stats[family_key] = (family_total, family_count)
        family_budget = load_factor * max(
            2.0 * SMOKE_DURATION_BUDGET_SECONDS,
            SMOKE_FAMILY_PER_CELL_SECONDS * family_count,
        )
        family_budgets = getattr(item.session, "_tl_smoke_family_budgets", None)
        if family_budgets is None:
            family_budgets = {}
            item.session._tl_smoke_family_budgets = family_budgets
        family_budgets[family_key] = family_budget
    tier_budget = _duration_budget_tier(item)
    if tier_budget is None:
        return report
    tier, base_budget = tier_budget
    budget = base_budget * load_factor
    item.session._tl_smoke_budget_value = budget
    if total_duration > budget:
        offenders = getattr(item.session, "_tl_duration_budget_offenders", None)
        if offenders is None:
            offenders = []
            item.session._tl_duration_budget_offenders = offenders
        offenders.append((item.nodeid, tier, total_duration, budget))
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
    # The coverage audit asserts called-functions ⊆ ArgSpec entries, so any
    # broad subset is sound (a smaller run can only check less, never lie).
    # Arm it on the nightly fast tier too: with only {"", "not rare"} accepted
    # no CI invocation ever collected stats and the gate skipped in 100% of CI
    # runs (b10 R79 / opus-R79-1).
    if mark_expression not in {"", "not rare", "not slow and not rare"}:
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
        snapshots[(module_name, name)] = _copy_sentinel_value(getattr(module, name, _MISSING))
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


_CAPABILITY_DEPENDENT_CACHES: tuple[tuple[str, str], ...] = (
    # Second-layer lru_caches whose cached value DERIVES from a lazy HAS_*
    # capability probe (grind p5 §3.9: the b7fe953e class one layer down).
    # Restoring the _torch_compat latches alone leaves a value computed under
    # a stubbed runtime frozen in these caches for the whole process, so the
    # probe restore must clear them too. The census in
    # tests/test_marker_lint.py::test_capability_dependent_caches_are_cleared
    # keeps this list complete.
    ("torchlens._runnable_state_context", "_fake_tensor_mode_class"),
    ("torchlens._runnable_state_context", "_count_bounded_fake_tensor_mode_class"),
)


def _clear_capability_dependent_caches() -> None:
    """Clear every registered probe-derived lru_cache (loaded modules only)."""

    for module_name, attr in _CAPABILITY_DEPENDENT_CACHES:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        getattr(module, attr).cache_clear()


@pytest.fixture(autouse=True)
def _restore_lazy_capability_probes() -> Iterator[None]:
    """Restore lazy ``HAS_*`` capability latches to their pre-test state.

    The lazy ``_torch_compat`` probes latch on first use, so a test that stubs
    ``sys.modules`` (or otherwise shims the runtime) while a probe fires
    poisons the flag for the whole process -- the recorded ``b7fe953e``
    incident class, previously patched per-test rather than systemically.
    Snapshotting before and restoring after every test bounds any mis-latch to
    the test that caused it; an un-latched probe simply re-probes on its next
    use, which is cheap and hits the real runtime. Second-layer caches built
    FROM a probed capability (``_CAPABILITY_DEPENDENT_CACHES``) are cleared in
    the same breath -- restoring the flag while a derived cache keeps the
    poisoned value would just move the incident one layer down.
    """

    from torchlens.utils import _torch_compat

    snapshot = _torch_compat.capability_probe_snapshot()
    try:
        yield
    finally:
        _torch_compat.restore_capability_probes(snapshot)
        _clear_capability_dependent_caches()


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
