import os
import sys
from os.path import join as opj

import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from torchlens import _state  # noqa: E402

# Menagerie tests exercise the menagerie/ build subsystem, which is not importable
# on Python < 3.11 because it uses datetime.UTC. Skip collecting them on those
# interpreters so the core suite still runs in the documented smoke environment.
collect_ignore_glob = []
if sys.version_info < (3, 11):
    collect_ignore_glob.append("test_menagerie_*.py")
    collect_ignore_glob.append("crawler/*.py")

# Deterministic seeding
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.use_deterministic_algorithms(True)

# Output directories. Routine test-generated artifacts go under tests/generated_outputs/,
# which is gitignored. Human-inspectable aesthetic reports keep stable paths there.
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_OUTPUTS_DIR = opj(TESTS_DIR, "generated_outputs")
REPORTS_DIR = opj(TEST_OUTPUTS_DIR, "reports")
VIS_OUTPUT_DIR = opj(TEST_OUTPUTS_DIR, "visualizations")

sub_dirs = [
    "cornet",
    "graph-neural-networks",
    "language-models",
    "multimodal-models",
    "taskonomy",
    "timm",
    "torchaudio",
    "torchvision-main",
    "torchvision-detection",
    "torchvision-opticflow",
    "torchvision-segmentation",
    "torchvision-video",
    "torchvision-quantize",
    "toy-networks",
    "aesthetic_test_models",
    "generative-models",
    "nlp-models",
    "time-series",
    "point-cloud",
    "super-resolution",
    "autoencoders",
    "state-space-models",
    "attention-variants",
    "gating-skip-patterns",
    "exotic-architectures",
    "reinforcement-learning",
    "efficient-transformers",
    "decoder-only-llms",
    "encoder-only",
    "encoder-decoder",
    "perceiver",
    "moe-models",
    "detection-additional",
]

os.makedirs(REPORTS_DIR, exist_ok=True)
for sub_dir in sub_dirs:
    os.makedirs(opj(VIS_OUTPUT_DIR, sub_dir), exist_ok=True)


# ---------------------------------------------------------------------------
# Coverage: auto-generate text report when pytest --cov is used
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Enable usage stats collection for ArgSpec coverage analysis."""
    _state._collect_usage_stats = True
    _state._function_call_counts.clear()
    _state._function_call_models.clear()


# Smoke-tier duration budget (see tests/test_marker_lint.py). The partition
# threshold for moving a test out of smoke is 5s measured; the enforcement
# budget is ~3x that so parallel-box load noise does not false-trip the lint.
SMOKE_DURATION_BUDGET_SECONDS = 15.0


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    """Record smoke-marked tests that blow the tier's duration budget.

    A static lint cannot know runtimes, so an UNMARKED slow test landing in the
    smoke tier is only catchable at runtime. Offenders are stashed on the session
    and asserted empty by ``test_marker_lint.py`` (ordered last), which names each
    offender and its measured duration.
    """

    report = yield
    if (
        report.when == "call"
        and report.duration > SMOKE_DURATION_BUDGET_SECONDS
        and item.get_closest_marker("smoke") is not None
    ):
        offenders = getattr(item.session, "_tl_smoke_budget_offenders", None)
        if offenders is None:
            offenders = []
            item.session._tl_smoke_budget_offenders = offenders
        offenders.append((item.nodeid, report.duration))
    return report


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

    coverage_tests = []
    lint_tests = []
    other_tests = []
    for item in items:
        if "test_arg_positions" in item.nodeid:
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


@pytest.fixture(autouse=True)
def _reset_deprecation_dedup():
    """Give every test a fresh deprecation-warning dedup set.

    ``warn_deprecated_alias`` correctly warns only once per process (so real user
    sessions are not spammed), but that process-global dedup makes any test that asserts
    ``pytest.warns(DeprecationWarning)`` order-dependent: if an earlier test already
    tripped the same alias, the warning is suppressed here and the assertion fails with
    "DID NOT WARN". Clearing the set before each test restores per-test isolation without
    changing the shipped once-per-process behavior.
    """

    from torchlens import _deprecations

    _deprecations._WARNED_DEPRECATIONS.clear()
    yield


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
