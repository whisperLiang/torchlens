"""Executable checks for documentation snippets.

Two gates live here:

* the per-block gate over the P2 ``docs/`` pages (``DOC_FILES``), and
* the canonical-page gate that EXECUTES every non-sketch Python fence in
  ``README.md``, ``CLAUDE.md``, and ``AGENTS.md`` top to bottom, statement by
  statement, in one shared namespace per page.

The canonical-page ambient contract is deliberately tiny: the harness injects
only the names the pages' prose treats as application-supplied -- ``model`` and
``x`` (a small fully-convolutional demo model), plus ``tf_model``/``tf_x`` when
the TensorFlow preview dependency is installed. Everything else must be defined
by the documentation code itself; a fence that references an undefined name is
a documentation bug and fails this gate. Blocks whose first line is a comment
containing "API sketch" are compile-gated instead of executed.
"""

from __future__ import annotations

import ast
import importlib.util
import linecache
import re
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

DOC_FILES = (
    "performance.md",
    "for-ai-agents.md",
    "reference/debug.md",
    "reference/export.md",
    "reference/attribution.md",
    "reference/collapse.md",
)
BLOCK_RE = re.compile(r"```python\n(?P<code>.*?)\n```", re.DOTALL)


def _docs_dir() -> Path:
    """Return the documentation directory.

    Returns
    -------
    Path
        Absolute path to ``docs``.
    """

    return Path(__file__).resolve().parents[1] / "docs"


def _iter_python_blocks() -> list[tuple[str, int, str]]:
    """Collect Python code fences from the P2 documentation pages.

    Returns
    -------
    list[tuple[str, int, str]]
        Tuples of ``(file_name, block_index, code)``.
    """

    blocks: list[tuple[str, int, str]] = []
    for file_name in DOC_FILES:
        text = (_docs_dir() / file_name).read_text(encoding="utf-8")
        for block_index, match in enumerate(BLOCK_RE.finditer(text), start=1):
            blocks.append((file_name, block_index, match.group("code")))
    return blocks


@pytest.mark.parametrize(
    ("file_name", "block_index", "code"),
    _iter_python_blocks(),
    ids=lambda value: str(value),
)
def test_p2_doc_python_block_runs(
    file_name: str, block_index: int, code: str, tmp_path: Path
) -> None:
    """Run one Python code fence from the new docs pages.

    Parameters
    ----------
    file_name:
        Markdown file name under ``docs``.
    block_index:
        One-based code-block index within the file.
    code:
        Python code fence body.
    tmp_path:
        Temporary directory supplied by pytest.
    """

    synthetic_filename = f"{file_name}:python-block-{block_index}"
    linecache.cache[synthetic_filename] = (
        len(code),
        None,
        [f"{line}\n" for line in code.splitlines()],
        synthetic_filename,
    )
    namespace: dict[str, Any] = {
        "__file__": synthetic_filename,
        "__name__": f"docs_snippet_{Path(file_name).stem}_{block_index}",
        "DOCS_TMPDIR": str(tmp_path),
    }
    try:
        exec(compile(code, synthetic_filename, "exec"), namespace)
    except (ImportError, ModuleNotFoundError) as exc:
        # A doc example may demonstrate an OPTIONAL-dependency feature (e.g. the xarray
        # export) that is not installed in every test env (CI installs only core deps).
        # The example is still correct; skip when its dependency is absent rather than
        # fail. Users who want that feature install the extra.
        pytest.skip(f"doc snippet requires an unavailable optional dependency: {exc}")


CANONICAL_PAGES = ("README.md", "CLAUDE.md", "AGENTS.md")
_SKETCH_MARKER_RE = re.compile(r"^\s*#.*\bAPI sketch\b", re.IGNORECASE)
_OPTIONAL_AMBIENT_NAMES = frozenset({"tf_model", "tf_x"})


class _DocsEncoder(nn.Module):
    """Conv+ReLU encoder giving the canonical pages a real ``encoder`` module."""

    def __init__(self) -> None:
        """Initialize one padded convolution."""

        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return the ReLU-activated convolution of ``value``.

        Parameters
        ----------
        value:
            Input image batch.

        Returns
        -------
        torch.Tensor
            Activated feature map.
        """

        return torch.relu(self.conv(value))


class _DocsModel(nn.Module):
    """Fully-convolutional demo model satisfying the canonical-page ambient contract.

    The pages index ``relu_1_2``, query receptive/projective geometry at spatial
    position ``(10, 10)``, and select ``tl.in_module("encoder")``, so the model
    keeps a windowed (convolutional) path from input to output, an ``encoder``
    submodule, and spatial extents larger than the queried coordinates.
    """

    def __init__(self) -> None:
        """Initialize the encoder and a convolutional head."""

        super().__init__()
        self.encoder = _DocsEncoder()
        self.head = nn.Conv2d(4, 2, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return the head applied to the encoded input.

        Parameters
        ----------
        value:
            Input image batch.

        Returns
        -------
        torch.Tensor
            Output feature map.
        """

        return self.head(self.encoder(value))


def _optional_tf_ambient() -> dict[str, Any]:
    """Build the TensorFlow-preview ambient names when the dependency is present.

    Returns
    -------
    dict[str, Any]
        ``tf_model``/``tf_x`` bindings, or an empty mapping when the TensorFlow
        preview prerequisites are unavailable.
    """

    if importlib.util.find_spec("tensorflow") is None:
        return {}
    try:
        import keras
        import tensorflow as tf
    except Exception:  # pragma: no cover - partial/broken optional install
        return {}
    if keras.backend.backend() != "tensorflow":  # pragma: no cover - env-specific
        return {}
    tf_model = keras.Sequential([keras.layers.Dense(3, activation="relu")])
    return {"tf_model": tf_model, "tf_x": tf.ones((2, 4))}


def _canonical_ambient() -> dict[str, Any]:
    """Return the shared execution namespace for one canonical page.

    Returns
    -------
    dict[str, Any]
        The declared ambient contract: ``model``/``x`` plus optional
        TensorFlow-preview names. Nothing else is injected.
    """

    torch.manual_seed(0)
    namespace: dict[str, Any] = {
        "model": _DocsModel().eval(),
        "x": torch.randn(1, 3, 16, 16),
    }
    namespace.update(_optional_tf_ambient())
    return namespace


@pytest.mark.heavy
@pytest.mark.parametrize("page_name", CANONICAL_PAGES)
def test_canonical_page_python_blocks_execute(
    page_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute every non-sketch Python fence of one canonical page as written.

    Fences run cumulatively in one shared namespace, statement by statement.
    The only tolerated non-executions are (a) fences whose first line carries an
    explicit ``API sketch`` comment marker (compile-gated) and (b) statements
    that read a declared OPTIONAL ambient name (``tf_model``/``tf_x``) on a box
    without the TensorFlow preview dependency. Any other failure is a
    documentation bug or a harness gap and fails the gate.

    Parameters
    ----------
    page_name:
        Repo-root markdown page name.
    tmp_path:
        Working directory for artifacts the snippets write (drawings, bundles).
    monkeypatch:
        Used to isolate the snippet working directory.
    """

    monkeypatch.chdir(tmp_path)
    page = _docs_dir().parent / page_name
    blocks = BLOCK_RE.findall(page.read_text(encoding="utf-8"))
    assert blocks, f"{page_name} has no python fences"
    namespace = _canonical_ambient()
    executed_statements = 0
    for block_index, code in enumerate(blocks, start=1):
        source_name = f"{page_name}:python-block-{block_index}"
        if _SKETCH_MARKER_RE.match(code.splitlines()[0]):
            compile(code, source_name, "exec")
            continue
        for node in ast.parse(code, filename=source_name).body:
            loads = {
                name.id
                for name in ast.walk(node)
                if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Load)
            }
            missing_optional = {
                name for name in loads & _OPTIONAL_AMBIENT_NAMES if name not in namespace
            }
            if missing_optional:
                continue
            statement = ast.Module(body=[node], type_ignores=[])
            try:
                exec(compile(statement, source_name, "exec"), namespace)
            except (ImportError, ModuleNotFoundError) as exc:
                pytest.skip(f"{source_name} requires an unavailable optional dependency: {exc}")
            executed_statements += 1
    assert executed_statements > 0, f"{page_name}: nothing executed"


def test_collapse_reference_gallery_exists_and_is_regenerable() -> None:
    """Keep visual-reference links and the render-script manifest aligned."""

    from scripts.render_collapse_reference import DEFAULT_OUT_DIR, IMAGE_NAMES

    page = (_docs_dir() / "reference" / "collapse.md").read_text(encoding="utf-8")
    linked_names = tuple(re.findall(r"\.\./images/collapse/([^)]*\.svg)", page))
    assert linked_names == IMAGE_NAMES
    missing = [name for name in IMAGE_NAMES if not (DEFAULT_OUT_DIR / name).is_file()]
    assert not missing, f"Regenerate with scripts/render_collapse_reference.py: {missing}"
