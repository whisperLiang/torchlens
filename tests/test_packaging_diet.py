"""Packaging-diet tests for lazy pandas and IPython imports."""

from pathlib import Path
import importlib.util
import re
import subprocess
import sys
from unittest.mock import patch
import zipfile

import pytest
import torch
from torch import nn

import torchlens as tl


class _TinyModel(nn.Module):
    """Small model for packaging smoke tests."""

    def __init__(self) -> None:
        """Initialize the tiny model layers."""

        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a single linear layer.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output tensor.
        """

        return self.linear(x)


def _make_log() -> tl.Trace:
    """Create a completed model log for packaging tests.

    Returns
    -------
    tl.Trace
        Completed log for a tiny model.
    """

    return tl.trace(_TinyModel(), torch.randn(1, 3))


def test_core_import_capture_and_show_without_optional_tabular_or_notebook_use() -> None:
    """Core import, capture, and show path succeed in the current dev environment."""

    log = _make_log()

    assert len(log.layer_list) > 0
    assert log.show(vis_mode="none") is None


def test_plain_capture_never_imports_dynamo_or_fsdp() -> None:
    """A fresh-process plain trace/record must not import torch._dynamo or FSDP.

    W21 cold-start guarantee: the compiled-wrapper and FSDP guards are lazy
    sys.modules probes, so a plain eager capture never pays those imports
    (~2s process time, ~874 modules, ~128MB RSS cold). Once FSDP genuinely is
    imported, the same probe must still detect and reject an FSDP wrapper.
    """

    script = """
import sys
import torch
from torch import nn
import torchlens as tl

model = nn.Sequential(nn.Linear(3, 3), nn.ReLU())
trace = tl.trace(model, torch.randn(2, 3))
assert len(trace.ops) > 0
recording = tl.record(model, torch.randn(2, 3), save=tl.func("relu"))
offenders = [
    name
    for name in sys.modules
    if name == "torch._dynamo"
    or name.startswith("torch._dynamo.")
    or name == "torch.distributed.fsdp"
    or name.startswith("torch.distributed.fsdp.")
]
assert not offenders, f"plain capture imported: {offenders}"

try:
    from torch.distributed.fsdp import FullyShardedDataParallel
except ImportError:
    FullyShardedDataParallel = None
if FullyShardedDataParallel is not None:
    wrapped = FullyShardedDataParallel.__new__(FullyShardedDataParallel)
    nn.Module.__init__(wrapped)
    wrapped.module = nn.Linear(3, 3)
    try:
        tl.trace(wrapped, torch.randn(2, 3))
    except RuntimeError as exc:
        assert "FullyShardedDataParallel" in str(exc)
    else:
        raise AssertionError("FSDP wrapper was not rejected after fsdp import")
print("COLD_IMPORT_OK")
"""
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=repo_root,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "COLD_IMPORT_OK" in result.stdout


def test_to_pandas_succeeds_when_tabular_extra_is_available() -> None:
    """Trace.to_pandas succeeds when pandas is installed."""

    log = _make_log()
    frame = log.to_pandas()

    assert not frame.empty
    assert "layer_label" in frame.columns


def test_to_pandas_missing_pandas_mentions_tabular_extra() -> None:
    """Trace.to_pandas raises a helpful extra-install hint when pandas is missing."""

    log = _make_log()

    with patch.dict("sys.modules", {"pandas": None}):
        try:
            log.to_pandas()
        except ImportError as exc:
            message = str(exc)
        else:
            raise AssertionError("Expected ImportError when pandas is unavailable.")

    assert "pandas is required for this feature" in message
    assert "pip install torchlens[tabular]" in message


def test_repr_html_succeeds_when_notebook_extra_is_available() -> None:
    """Trace._repr_html_ returns the Phase 3 HTML card when IPython is installed."""
    pytest.importorskip("IPython")

    log = _make_log()
    html = log._repr_html_()

    assert html.startswith("<div")
    assert "TorchLens Trace" in html
    assert "NaN/Inf" in html


def test_repr_html_missing_ipython_falls_back_to_text() -> None:
    """Trace._repr_html_ falls back to text when IPython is missing."""

    log = _make_log()

    with patch.dict("sys.modules", {"IPython": None, "IPython.display": None}):
        html = log._repr_html_()

    assert html == repr(log)


def test_packaging_metadata_uses_resolvable_gradcam_and_guarded_tinygrad_extra() -> None:
    """Packaging metadata keeps the corrected grad-cam name and tinygrad marker."""

    pyproject_text = Path(__file__).resolve().parent.parent.joinpath("pyproject.toml").read_text()

    assert 'gradcam = ["grad-cam~=1.5"]' in pyproject_text
    assert '"grad-cam~=1.5",' in pyproject_text
    assert "tinygrad = [\"tinygrad>=0.13,<0.14; python_version >= '3.11'\"]" in pyproject_text


def test_precommit_and_conftest_guard_use_python3_safe_predicates() -> None:
    """The hook entrypoints and menagerie guard stay on the hardened conditions."""

    project_root = Path(__file__).resolve().parent.parent
    precommit_text = project_root.joinpath(".pre-commit-config.yaml").read_text()
    conftest_text = project_root.joinpath("tests", "conftest.py").read_text()

    assert "entry: python scripts/check_no_breaking_markers.py" not in precommit_text
    assert "entry: scripts/check_no_breaking_markers.py --commit-msg" in precommit_text
    assert "entry: scripts/check_no_breaking_markers.py --pre-push" in precommit_text
    assert "if sys.version_info < (3, 11):" in conftest_text
    assert 'collect_ignore_glob.append("test_menagerie_*.py")' in conftest_text
    assert 'collect_ignore_glob.append("crawler/*.py")' in conftest_text


def test_ci_workflows_pin_torch_and_scope_lint_to_owned_paths() -> None:
    """Packaging CI keeps torch pins honest and excludes the external menagerie boundary."""

    project_root = Path(__file__).resolve().parent.parent
    nightly_text = project_root.joinpath(".github", "workflows", "nightly.yml").read_text()
    weekly_text = project_root.joinpath(".github", "workflows", "weekly.yml").read_text()
    lint_text = project_root.joinpath(".github", "workflows", "lint.yml").read_text()

    for workflow_text in (nightly_text, weekly_text):
        assert "torch==2.7.*" in workflow_text
        assert 'uv pip install --system -c "${{ runner.temp }}/torch-2.7-constraints.txt"' in (
            workflow_text
        )
        assert "uv pip check" in workflow_text
        assert 'assert torch.__version__.startswith("2.7.")' in workflow_text

    assert "ruff format --check torchlens tests scripts" in lint_text
    assert "ruff check torchlens tests scripts" in lint_text

    # The excluded set moved from lint.yml CLI flags into pyproject's
    # `[tool.ruff] extend-exclude` so that pre-commit -- which passes explicit
    # staged filenames and therefore ignores CLI --exclude -- reaches the same
    # verdict as this gate. Assert the boundary at its single authority, and that
    # it has NOT drifted back into duplicate CLI flags.
    pyproject_text = project_root.joinpath("pyproject.toml").read_text()
    extend_exclude = re.search(
        r"^\s*extend-exclude\s*=\s*\[(.*?)\]", pyproject_text, re.DOTALL | re.MULTILINE
    )
    assert extend_exclude is not None, "pyproject [tool.ruff] must declare extend-exclude"
    excluded = set(re.findall(r'"([^"]+)"', extend_exclude.group(1)))
    assert excluded == {"menagerie", "tests/crawler", "tests/test_menagerie_*.py"}
    assert "--exclude" not in lint_text


def test_ruff_pin_is_identical_across_declaration_sites() -> None:
    """The ruff that WRITES the code and the ruff that JUDGES it must be one version.

    Three files independently name a ruff version: pyproject's dev extra, the Lint
    workflow's install step, and the ruff-pre-commit ``rev``. When they drift, the
    pre-commit formatter rewrites code to a style CI then rejects -- which is exactly
    how the repo accumulated 147 format-stale files under a v0.9.7 hook while CI
    judged with 0.15.4.
    """

    project_root = Path(__file__).resolve().parent.parent
    pyproject_text = project_root.joinpath("pyproject.toml").read_text()
    lint_text = project_root.joinpath(".github", "workflows", "lint.yml").read_text()
    precommit_text = project_root.joinpath(".pre-commit-config.yaml").read_text()

    dev_pins = set(re.findall(r'"ruff==([0-9]+\.[0-9]+\.[0-9]+)"', pyproject_text))
    ci_pins = set(re.findall(r"ruff==([0-9]+\.[0-9]+\.[0-9]+)", lint_text))
    hook_revs = set(
        re.findall(
            r"repo:\s*https://github\.com/astral-sh/ruff-pre-commit\s*\n"
            r"(?:\s*#.*\n)*"
            r"\s*rev:\s*v([0-9]+\.[0-9]+\.[0-9]+)",
            precommit_text,
        )
    )

    assert len(dev_pins) == 1, f"expected exactly one ruff dev pin, got {dev_pins}"
    assert len(ci_pins) == 1, f"expected exactly one ruff CI pin, got {ci_pins}"
    assert len(hook_revs) == 1, f"expected exactly one ruff-pre-commit rev, got {hook_revs}"
    assert dev_pins == ci_pins == hook_revs, (
        "ruff version drift: pyproject dev extra "
        f"{dev_pins}, lint.yml {ci_pins}, .pre-commit-config.yaml {hook_revs}. "
        "The formatter and the gate must be the same ruff."
    )


@pytest.mark.slow
def test_built_wheel_manifest_is_diet(tmp_path: Path) -> None:
    """Assert the built wheel's manifest: schemas in, py.typed in, menagerie OUT.

    Nothing used to test the wheel manifest, and it had drifted three ways at
    once: ``menagerie*`` was in the distributed package set (2886 of 3316
    members, ~13.5 MB, plus ``menagerie`` squatting as a top-level import name),
    ``torchlens/py.typed`` was missing so downstream mypy ignored every
    annotation in the package (PEP 561), and only the schema files were checked.
    """

    project_root = Path(__file__).resolve().parent.parent
    wheel_dir = tmp_path / "wheelhouse"
    wheel_dir.mkdir()

    if importlib.util.find_spec("build") is not None:
        command = [sys.executable, "-m", "build", "--wheel", "--outdir", str(wheel_dir)]
    elif importlib.util.find_spec("pip") is not None:
        command = [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "-w", str(wheel_dir)]
    else:
        pytest.skip("neither build nor pip is importable for wheel construction")

    subprocess.run(command, cwd=project_root, check=True)
    wheels = sorted(wheel_dir.glob("torchlens-*.whl"))
    assert len(wheels) == 1

    with zipfile.ZipFile(wheels[0]) as wheel_zip:
        members = wheel_zip.namelist()
        top_level_members = [m for m in members if m.endswith("top_level.txt")]
        assert len(top_level_members) == 1
        top_level = wheel_zip.read(top_level_members[0]).decode().split()

    schema_members = [
        m for m in members if m.startswith("torchlens/schemas/") and m.endswith(".json")
    ]
    assert schema_members, "expected at least one torchlens/schemas/*.json wheel member"

    # PEP 561: without this marker file downstream type checkers treat the
    # package as untyped and skip every annotation it ships.
    assert "torchlens/py.typed" in members, "wheel must ship the PEP 561 py.typed marker"

    # The menagerie corpus is repo/sdist-only, never part of the installed library.
    menagerie_members = [m for m in members if m.startswith("menagerie")]
    assert not menagerie_members, (
        f"wheel ships {len(menagerie_members)} menagerie member(s); the corpus is "
        "not part of the distributed library (see [tool.setuptools.packages.find])"
    )
    assert top_level == ["torchlens"], (
        f"wheel installs top-level name(s) {top_level}; torchlens must be the only one"
    )
