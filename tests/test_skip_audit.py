"""Skip-surface audit (R79): make every skip in tests/ honest and auditable.

Three enforcement layers:

1. **importorskip ledger.** Every ``pytest.importorskip`` target in tests/ must
   appear in ``IMPORTORSKIP_LEDGER`` with an availability tier, so a skip is
   always attributable to a *known* class of absence:

   - ``"test-extra"``: the module ships with the declared ``[test]`` extra (or
     is a core torchlens dependency), so a full dev/CI install HAS it and a
     skip indicates install breakage, never a legitimate environment.
   - ``"optional-preview"``: the module belongs to a backend/bridge/appliance
     extra (``jax``, ``mlx``, ``tf``, ``tabular``, ``notebook``, ...) and is
     legitimately absent outside that extra's environment.
   - ``"unavailable-ok"``: no declared extra covers it; the ledger note says
     why the absence is acceptable (torch build probes, unreleased deps,
     research-model deps, py-version backports).

   A new importorskip target must be ledgered consciously (the inventory test
   fails otherwise -- red-capable by construction), and when the environment
   claims the full ``[test]`` extra every ``"test-extra"`` target must resolve.

2. **Unconditional-skip ledger.** An AST scan flags every ``pytest.skip`` call
   with no conditional ancestor, every bare ``pytest.mark.skip`` decorator, and
   every ``pytestmark`` carrying a bare skip. Each hit must be ledgered in
   ``UNCONDITIONAL_SKIP_LEDGER`` with a dated justification: no silent dead
   tests.

3. **``python -O`` leg self-verification.** The ``requires_assertions`` marker
   exists so assert-based audits skip (not silently pass) under ``-O``; no CI
   workflow runs that leg, so this file proves the contract in-suite with one
   targeted subprocess: marked tests SKIP under ``-O`` while a plain sentinel
   test executes.

Both scanners prove red-capability against planted offenders in ``tmp_path``.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from functools import cache
from pathlib import Path

import pytest

# NOTE: no module-level smoke pytestmark -- the -O subprocess test below is
# `heavy`, and the tier markers are additive/disjoint (tests/test_marker_lint.py).
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

TEST_EXTRA = "test-extra"
OPTIONAL_PREVIEW = "optional-preview"
UNAVAILABLE_OK = "unavailable-ok"
VALID_TIERS = frozenset({TEST_EXTRA, OPTIONAL_PREVIEW, UNAVAILABLE_OK})

# Importing this sentinel is the environment's claim to carry the full [test]
# extra; a partial install then can no longer masquerade as full coverage.
FULL_TEST_EXTRA_SENTINEL = "timm"

# Every pytest.importorskip target in tests/ -> (tier, why that tier).
# Keep sorted; the inventory test enforces exact set equality.
IMPORTORSKIP_LEDGER: dict[str, tuple[str, str]] = {
    "IPython": (OPTIONAL_PREVIEW, "notebook extra"),
    "PIL": (TEST_EXTRA, "pillow (also a core torchlens dependency)"),
    "brainscore_core": (OPTIONAL_PREVIEW, "neuro extra (brain-score dist)"),
    "cairosvg": (
        UNAVAILABLE_OK,
        "undeclared SVG-render inspection helper; extras-gap candidate reported 2026-08-15",
    ),
    "captum.attr": (OPTIONAL_PREVIEW, "captum extra"),
    "cornet": (
        UNAVAILABLE_OK,
        "CORnet research package, installed from GitHub, no maintained PyPI dist",
    ),
    "dacite": (
        UNAVAILABLE_OK,
        "model-explorer export-bridge demo dependency, no declared extra",
    ),
    "dagua": (UNAVAILABLE_OK, "unreleased in-development layout engine"),
    "e3nn.o3": (
        UNAVAILABLE_OK,
        "research-model dependency for real-world coverage, deliberately undeclared",
    ),
    "equinox": (OPTIONAL_PREVIEW, "jax extra"),
    "fitz": (
        UNAVAILABLE_OK,
        "PyMuPDF PDF-render inspection helper, undeclared; extras-gap candidate reported 2026-08-15",
    ),
    "flax.nnx": (OPTIONAL_PREVIEW, "jax extra"),
    "graphviz": (TEST_EXTRA, "core torchlens dependency (dist 'graphviz')"),
    "jax": (OPTIONAL_PREVIEW, "jax extra"),
    "jax.experimental.shard_map": (OPTIONAL_PREVIEW, "jax extra"),
    "jax.lax": (OPTIONAL_PREVIEW, "jax extra"),
    "jax.numpy": (OPTIONAL_PREVIEW, "jax extra"),
    "jax.random": (OPTIONAL_PREVIEW, "jax extra"),
    "jupyter_client": (OPTIONAL_PREVIEW, "notebook extra"),
    "keras": (OPTIONAL_PREVIEW, "tf extra (ships with tensorflow>=2.16)"),
    "lightning": (TEST_EXTRA, "lightning"),
    "matplotlib": (
        UNAVAILABLE_OK,
        "undeclared viz-test dependency; extras-gap candidate reported 2026-08-15",
    ),
    "matplotlib.pyplot": (
        UNAVAILABLE_OK,
        "undeclared viz-test dependency; extras-gap candidate reported 2026-08-15",
    ),
    "mlx": (OPTIONAL_PREVIEW, "mlx extra"),
    "mlx.core": (OPTIONAL_PREVIEW, "mlx extra"),
    "mlx.nn": (OPTIONAL_PREVIEW, "mlx extra"),
    "model_explorer": (
        UNAVAILABLE_OK,
        "export-bridge target with no declared extra; extras-gap candidate reported 2026-08-15",
    ),
    "paddle": (OPTIONAL_PREVIEW, "paddle extra (paddlepaddle dist)"),
    "pandas": (OPTIONAL_PREVIEW, "tabular extra"),
    "pandas.api.types": (OPTIONAL_PREVIEW, "tabular extra"),
    "pennylane": (
        UNAVAILABLE_OK,
        "quantum-ML research-model dependency, deliberately undeclared",
    ),
    "psutil": (TEST_EXTRA, "psutil"),
    "pyarrow": (OPTIONAL_PREVIEW, "tabular extra"),
    "pyarrow.parquet": (OPTIONAL_PREVIEW, "tabular extra"),
    "pydantic": (TEST_EXTRA, "pydantic"),
    "pydot": (TEST_EXTRA, "pydot"),
    "pytorch_lightning": (TEST_EXTRA, "ships inside the 'lightning' distribution"),
    "rsatoolbox": (OPTIONAL_PREVIEW, "neuro extra"),
    "sae_lens": (OPTIONAL_PREVIEW, "sae extra"),
    "sentence_transformers": (OPTIONAL_PREVIEW, "compat-shims extra"),
    "tensorboard": (
        UNAVAILABLE_OK,
        "export-bridge target with no declared extra",
    ),
    "tensorflow": (OPTIONAL_PREVIEW, "tf extra"),
    "timm": (TEST_EXTRA, "timm"),
    "tinygrad": (OPTIONAL_PREVIEW, "tinygrad extra (py>=3.11 only)"),
    "tinygrad.nn": (OPTIONAL_PREVIEW, "tinygrad extra (py>=3.11 only)"),
    "tomli": (
        UNAVAILABLE_OK,
        "py<3.11 tomllib backport, only conditionally needed; extras-gap candidate "
        "reported 2026-08-15 (tomli; python_version<'3.11' belongs in [test])",
    ),
    "torch._subclasses.fake_tensor": (
        UNAVAILABLE_OK,
        "torch build/version capability probe (core torch is required)",
    ),
    "torch.ao.quantization": (
        UNAVAILABLE_OK,
        "torch build/version capability probe (core torch is required)",
    ),
    "torch.distributed": (
        UNAVAILABLE_OK,
        "torch build capability probe; absent on some torch builds",
    ),
    "torch.distributed.tensor": (
        UNAVAILABLE_OK,
        "torch build/version capability probe (core torch is required)",
    ),
    "torch.distributed.tensor.parallel": (
        UNAVAILABLE_OK,
        "torch build/version capability probe (core torch is required)",
    ),
    "torch.nn.attention.bias": (
        UNAVAILABLE_OK,
        "torch version capability probe (newer-torch namespace)",
    ),
    "torch_geometric": (TEST_EXTRA, "torch_geometric"),
    "torch_geometric.nn": (TEST_EXTRA, "torch_geometric"),
    "torchaudio": (TEST_EXTRA, "torchaudio"),
    "torchvision": (TEST_EXTRA, "torchvision"),
    "torchvision.models": (TEST_EXTRA, "torchvision"),
    "torchvision.models.resnet": (TEST_EXTRA, "torchvision"),
    "torchvision.models.segmentation": (TEST_EXTRA, "torchvision"),
    "torchvision.ops": (TEST_EXTRA, "torchvision"),
    "torchvision.transforms": (TEST_EXTRA, "torchvision"),
    "transformer_lens": (
        UNAVAILABLE_OK,
        "bridge integration without a declared extra; extras-gap candidate reported 2026-08-15",
    ),
    "transformers": (TEST_EXTRA, "transformers"),
    "transformers.modeling_outputs": (TEST_EXTRA, "transformers"),
    "visualpriors": (TEST_EXTRA, "visualpriors"),
    "wandb": (OPTIONAL_PREVIEW, "wandb extra"),
    "xarray": (
        UNAVAILABLE_OK,
        "export-bridge target with no declared extra; extras-gap candidate reported 2026-08-15",
    ),
}

# importorskip call sites whose target is computed, not a string literal.
# Each must be consciously allowlisted here (relative posix path from tests/).
DYNAMIC_IMPORTORSKIP_SITES = frozenset(
    {
        "test_backend_registry.py",  # importorskips the registry's own dependency map
        "test_menagerie_module_split.py",  # importorskips a parametrized optional dep
    }
)

# Every unconditional skip in tests/ (scanner key -> dated justification).
UNCONDITIONAL_SKIP_LEDGER: dict[str, str] = {
    "test_io_integration.py::test_data_parallel_and_ddp_streaming_case_is_explicitly_skipped": (
        "[2026-08-15] deliberate placeholder: torchlens is single-process by design, "
        "so no DataParallel/DDP streaming capture exists to test; the entry documents "
        "the absent coverage explicitly instead of vanishing from the suite"
    ),
}

_CONDITIONAL_ANCESTORS: tuple[type, ...] = tuple(
    node_type
    for node_type in (
        ast.If,
        ast.IfExp,
        ast.Try,
        ast.ExceptHandler,
        ast.While,
        ast.For,
        getattr(ast, "Match", None),
    )
    if node_type is not None
)


def _iter_test_files(root: Path) -> list[Path]:
    """Every python file under ``root``, excluding bytecode caches."""

    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


@cache
def _read_text(path_str: str) -> str:
    """Cached raw source read (needle pre-filter input)."""

    return Path(path_str).read_text()


@cache
def _parse_with_parents(path_str: str) -> ast.Module:
    """Parse a file and annotate every node with its ``_tl_parent``.

    Only files that pass a raw-text needle check ever reach this parse, which
    keeps the scanners inside the smoke duration budget.
    """

    tree = ast.parse(_read_text(path_str), filename=path_str)
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._tl_parent = node  # type: ignore[attr-defined]
    return tree


def collect_importorskip_targets(root: Path) -> tuple[dict[str, list[str]], list[str]]:
    """Return (literal target -> sites, dynamic-call site files) under ``root``."""

    literal: dict[str, list[str]] = {}
    dynamic: list[str] = []
    for path in _iter_test_files(root):
        if "importorskip" not in _read_text(str(path)):
            continue
        rel = path.relative_to(root).as_posix()
        tree = _parse_with_parents(str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else func.id
                if isinstance(func, ast.Name)
                else None
            )
            if name != "importorskip":
                continue
            first = node.args[0] if node.args else None
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                literal.setdefault(first.value, []).append(f"{rel}:{node.lineno}")
            else:
                dynamic.append(rel)
    return literal, dynamic


def _is_bare_mark_skip(node: ast.expr) -> bool:
    """Whether ``node`` is ``pytest.mark.skip`` or ``pytest.mark.skip(...)``."""

    target = node.func if isinstance(node, ast.Call) else node
    try:
        rendered = ast.unparse(target)
    except Exception:  # pragma: no cover - unparse never fails on parsed source
        return False
    return rendered.endswith("mark.skip")


def collect_unconditional_skips(root: Path) -> dict[str, str]:
    """Scan ``root`` for skips that fire unconditionally.

    Detects three shapes:

    - a ``pytest.skip(...)`` call statement with no conditional ancestor
      (``if``/``try``/``while``/``for``/``match``) inside its enclosing scope;
    - a bare ``pytest.mark.skip`` / ``pytest.mark.skip(...)`` decorator on a
      function or class (``skipif`` never matches);
    - a module-level ``pytestmark`` assignment carrying a bare skip mark.

    Returns scanner keys (``relpath::qualname``) -> a short site description.
    """

    findings: dict[str, str] = {}
    needles = ("pytest.skip", "mark.skip", "pytestmark")
    for path in _iter_test_files(root):
        text = _read_text(str(path))
        if not any(needle in text for needle in needles):
            continue
        rel = path.relative_to(root).as_posix()
        tree = _parse_with_parents(str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for dec in node.decorator_list:
                    if _is_bare_mark_skip(dec):
                        findings[f"{rel}::{node.name}"] = (
                            f"bare mark.skip decorator at line {dec.lineno}"
                        )
            elif isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if "pytestmark" not in targets:
                    continue
                marks = (
                    node.value.elts
                    if isinstance(node.value, (ast.List, ast.Tuple))
                    else [node.value]
                )
                if any(_is_bare_mark_skip(mark) for mark in marks):
                    findings[f"{rel}::<pytestmark>"] = (
                        f"bare mark.skip in pytestmark at line {node.lineno}"
                    )
            elif isinstance(node, ast.Call):
                try:
                    func_name = ast.unparse(node.func)
                except Exception:  # pragma: no cover
                    continue
                if func_name != "pytest.skip":
                    continue
                ancestor = getattr(node, "_tl_parent", None)
                conditional = False
                scope = "<module>"
                while ancestor is not None:
                    if isinstance(ancestor, _CONDITIONAL_ANCESTORS):
                        conditional = True
                    if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        scope = ancestor.name
                        break
                    ancestor = getattr(ancestor, "_tl_parent", None)
                if not conditional:
                    findings[f"{rel}::{scope}"] = (
                        f"unconditional pytest.skip call at line {node.lineno}"
                    )
    return findings


# ---------------------------------------------------------------------------
# 1. importorskip inventory vs ledger
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_ledger_tiers_are_valid() -> None:
    """Every ledger row uses a closed tier vocabulary and a non-empty note."""

    for target, (tier, note) in IMPORTORSKIP_LEDGER.items():
        assert tier in VALID_TIERS, f"{target}: unknown tier {tier!r}"
        assert note.strip(), f"{target}: empty ledger note"
    sentinel_tier, _ = IMPORTORSKIP_LEDGER[FULL_TEST_EXTRA_SENTINEL]
    assert sentinel_tier == TEST_EXTRA, "the full-extra sentinel must itself be test-extra"


@pytest.mark.smoke
def test_importorskip_inventory_matches_ledger() -> None:
    """Every importorskip target is ledgered; every ledger row is still used."""

    literal, _ = collect_importorskip_targets(TESTS_DIR)
    inventory = set(literal)
    ledgered = set(IMPORTORSKIP_LEDGER)
    unledgered = inventory - ledgered
    stale = ledgered - inventory
    assert not unledgered, (
        "New pytest.importorskip targets must be ledgered consciously in "
        "IMPORTORSKIP_LEDGER (tests/test_skip_audit.py) with an availability tier:\n  "
        + "\n  ".join(f"{t} (e.g. {literal[t][0]})" for t in sorted(unledgered))
    )
    assert not stale, (
        "Ledger rows with no remaining importorskip site (delete them):\n  "
        + "\n  ".join(sorted(stale))
    )


@pytest.mark.smoke
def test_dynamic_importorskip_sites_are_allowlisted() -> None:
    """Non-literal importorskip calls stay confined to the known dynamic sites."""

    _, dynamic = collect_importorskip_targets(TESTS_DIR)
    unexpected = set(dynamic) - DYNAMIC_IMPORTORSKIP_SITES
    assert not unexpected, (
        "importorskip with a computed target evades the ledger; allowlist the site "
        "consciously in DYNAMIC_IMPORTORSKIP_SITES:\n  " + "\n  ".join(sorted(unexpected))
    )
    missing = DYNAMIC_IMPORTORSKIP_SITES - set(dynamic)
    assert not missing, (
        "Allowlisted dynamic importorskip sites no longer exist (delete them):\n  "
        + "\n  ".join(sorted(missing))
    )


@pytest.mark.smoke
def test_importorskip_scanner_is_red_capable(tmp_path: Path) -> None:
    """The inventory scanner catches a planted unledgered target and dynamic site."""

    planted = tmp_path / "test_planted_offender.py"
    planted.write_text(
        "import pytest\n"
        'pytest.importorskip("planted_unledgered_module_xyz")\n'
        "name = 'computed'\n"
        "pytest.importorskip(name)\n"
    )
    literal, dynamic = collect_importorskip_targets(tmp_path)
    assert "planted_unledgered_module_xyz" in literal
    assert dynamic == ["test_planted_offender.py"]


@pytest.mark.smoke
def test_test_extra_targets_import_in_full_env() -> None:
    """When the env claims the full [test] extra, every test-extra target resolves.

    The claim is the sentinel dep importing; a partial install (sentinel absent)
    legitimately skips, but can then never masquerade as full-extra coverage.
    Resolution is checked on each target's top-level module via ``find_spec``
    (cheap; no heavyweight imports), which is what importorskip availability
    hinges on.
    """

    if importlib.util.find_spec(FULL_TEST_EXTRA_SENTINEL) is None:
        pytest.skip(
            f"environment does not claim the full [test] extra "
            f"(sentinel {FULL_TEST_EXTRA_SENTINEL!r} is absent)"
        )
    missing = []
    for target, (tier, _note) in sorted(IMPORTORSKIP_LEDGER.items()):
        if tier != TEST_EXTRA:
            continue
        top_level = target.split(".", 1)[0]
        if importlib.util.find_spec(top_level) is None:
            missing.append(target)
    assert not missing, (
        "The environment claims the full [test] extra (sentinel imports) but these "
        "test-extra importorskip targets do not resolve -- their tests are silently "
        "skipping on what should be a fully-provisioned install:\n  " + "\n  ".join(missing)
    )


# ---------------------------------------------------------------------------
# 2. unconditional-skip ledger
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_no_unledgered_unconditional_skips() -> None:
    """Every unconditional skip in tests/ is a consciously ledgered placeholder."""

    findings = collect_unconditional_skips(TESTS_DIR)
    unledgered = set(findings) - set(UNCONDITIONAL_SKIP_LEDGER)
    stale = set(UNCONDITIONAL_SKIP_LEDGER) - set(findings)
    assert not unledgered, (
        "Unconditional skips make tests silently dead; either realize the test, "
        "delete it, or ledger it as a dated deliberate placeholder in "
        "UNCONDITIONAL_SKIP_LEDGER (tests/test_skip_audit.py):\n  "
        + "\n  ".join(f"{k}: {findings[k]}" for k in sorted(unledgered))
    )
    assert not stale, (
        "Ledgered unconditional skips no longer exist (delete the rows):\n  "
        + "\n  ".join(sorted(stale))
    )


@pytest.mark.smoke
def test_unconditional_skip_scanner_is_red_capable(tmp_path: Path) -> None:
    """The scanner catches all three planted offender shapes and no decoys."""

    planted = tmp_path / "test_planted_skips.py"
    planted.write_text(
        "import pytest\n"
        "\n"
        "\n"
        "@pytest.mark.skip(reason='planted decorator offender')\n"
        "def test_decorated():\n"
        "    pass\n"
        "\n"
        "\n"
        "def test_body_skip():\n"
        "    pytest.skip('planted body offender')\n"
        "\n"
        "\n"
        "def test_conditional_decoy():\n"
        "    if False:\n"
        "        pytest.skip('conditional; must NOT be flagged')\n"
        "\n"
        "\n"
        "@pytest.mark.skipif(True, reason='skipif decoy; must NOT be flagged')\n"
        "def test_skipif_decoy():\n"
        "    pass\n"
    )
    marked = tmp_path / "test_planted_pytestmark.py"
    marked.write_text(
        "import pytest\n"
        "pytestmark = [pytest.mark.smoke, pytest.mark.skip(reason='planted module offender')]\n"
    )
    findings = collect_unconditional_skips(tmp_path)
    assert set(findings) == {
        "test_planted_skips.py::test_decorated",
        "test_planted_skips.py::test_body_skip",
        "test_planted_pytestmark.py::<pytestmark>",
    }


# ---------------------------------------------------------------------------
# 3. requires_assertions / python -O leg self-verification
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_o_leg_sentinel_executes() -> None:
    """Trivial sentinel proving the ``-O`` subprocess really executes tests.

    Deliberately assert-free (``-O`` strips assert statements): failure is an
    explicit ``pytest.fail``.
    """

    if (1, 2)[0] != 1:  # pragma: no cover - arithmetic sanity sentinel
        pytest.fail("sentinel arithmetic failed")


@pytest.mark.heavy
def test_requires_assertions_marker_engages_under_python_O(tmp_path: Path) -> None:
    """``requires_assertions`` tests SKIP (never pass/fail) under ``python -O``.

    No CI workflow runs a ``-O`` leg, so without this subprocess probe the
    marker and its conftest hook would rot unverified. One targeted run under
    ``-O`` proves both directions: the marked tests are collected and SKIPPED
    with the documented reason, while an ordinary test executes and passes on
    the same interpreter. Red-capable end to end: removing the conftest hook
    makes the marked assert-based tests FAIL under ``-O`` (stripped asserts
    never raise), and removing the marker breaks the exact skip count.
    """

    marked_nodes = [
        "tests/test_postprocess_contract_arming.py::test_unknown_step_contract_is_rejected",
        "tests/test_postprocess_contract_arming.py::test_undeclared_write_is_rejected",
    ]
    plain_node = "tests/test_skip_audit.py::test_o_leg_sentinel_executes"
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-m",
            "pytest",
            "-q",
            "-rs",
            "-p",
            "no:randomly",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(tmp_path / "oleg-bt"),
            *marked_nodes,
            plain_node,
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"-O leg subprocess failed:\n{output}"
    assert "1 passed" in output, f"plain sentinel did not execute under -O:\n{output}"
    assert "2 skipped" in output, (
        f"expected exactly the two requires_assertions tests to skip under -O:\n{output}"
    )
    assert "requires assertions" in output, (
        f"skip reason does not name the requires_assertions contract:\n{output}"
    )
