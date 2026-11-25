"""Check that the workflow selectors cover every executable documentation snippet."""

from __future__ import annotations

import ast
import json
import shlex
import subprocess
import sys
from itertools import takewhile
from pathlib import Path
from typing import Any

import pytest
from _pytest.mark.expression import Expression
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from scripts.backend_ci_changes import leg_should_run
from tests import test_docs_snippets as snippets
from tests.test_backend_framework_pins import (
    _NIGHTLY_WORKFLOW,
    _PR_WORKFLOW,
    _framework,
    _legs,
    _workflow,
)

REPO = Path(__file__).resolve().parents[1]
PEER_FILES = ("tests/test_recipe_pages_executed.py", "tests/test_docs_snippets.py")

# Read the real pytest items after collection hooks, including inherited marks.
_COLLECT = """
import json
import sys
from pathlib import Path

import pytest

class Inventory:
    def pytest_collection_finish(self, session):
        Path(sys.argv[1]).write_text(json.dumps({
            item.nodeid: ([mark.name for mark in item.iter_markers()],
                          item.callspec.params if hasattr(item, "callspec") else {})
            for item in session.items
        }), encoding="utf-8")

raise SystemExit(pytest.main(["--collect-only", "-q", "-m", "", *sys.argv[2:]],
                             plugins=[Inventory()]))
"""


def _pytest_step(job: dict[str, Any], name: str) -> tuple[dict[str, Any], list[str], str]:
    """Read a workflow step's actual pytest paths and marker expression."""
    step = next(step for step in job["steps"] if step.get("name") == name)
    line = next(
        line for line in step["run"].replace("\\\n", "").splitlines() if "pytest tests/" in line
    )
    words = shlex.split(line)
    args = words[words.index("pytest") + 1 :]
    paths = list(takewhile(lambda arg: not arg.startswith("-"), args))
    return step, paths, args[args.index("-m") + 1] if "-m" in args else ""


def _installed_markers(rows: dict[str, dict[str, Any]], requirements: str) -> set[str]:
    """Read the frameworks installed together in one preview pytest invocation."""
    distributions = {
        canonicalize_name(Requirement(text).name) for text in shlex.split(requirements)
    }
    return {
        row["marker"]
        for backend, row in rows.items()
        if distributions.intersection(map(canonicalize_name, _framework(backend).distributions))
    }


def _attested_docs(
    repo: Path,
    manifest: str | None,
    selector: str,
    installed: set[str],
    marks: dict[str, set[str]],
    required: dict[str, set[str]],
) -> set[str]:
    """Check the exact required docs in a selection against its passed-ID floor."""
    expression = Expression.compile(selector)
    expected = {
        node
        for node, needs in required.items()
        if needs <= installed and expression.evaluate(marks[node].__contains__)
    }
    if not expected and not manifest:
        return set()
    assert manifest, f"{selector}: missing a passed-ID file for {sorted(expected)}"
    ids = {
        line.strip()
        for line in (repo / manifest).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert ids and ids == expected, (
        f"{manifest}: unexpected IDs {sorted(ids - expected)}, missing IDs {sorted(expected - ids)}"
    )
    assert not {node for node in ids if "xfail" in marks[node]}, (
        f"{manifest}: required tests may not carry xfail (JUnit can count XPASS as a pass)"
    )
    return ids


def _preview_coverage(
    workflow: Path,
    job_name: str,
    marks: dict[str, set[str]],
    required: dict[str, set[str]],
    pages: dict[str, str],
) -> tuple[set[str], set[str]]:
    """Check the existing preview suites, including the co-installed pair."""
    nightly = job_name == "preview-backends"
    job = _workflow(workflow)["jobs"][job_name]
    rows = _legs(workflow, job_name)
    suite = (
        "Run ${{ matrix.backend }} preview backend tests"
        if nightly
        else "Run the ${{ matrix.backend }} backend suite"
    )
    step, paths, selector = _pytest_step(job, suite)
    assert paths == ["tests/"]
    marker = "${{ matrix.marker }}" if nightly else "$MARKER"
    assert selector == f"{marker} and not slow"
    if not nightly:
        assert step["env"]["MARKER"] == "${{ matrix.marker }}"
    floor = next(
        step["run"]
        for step in job["steps"]
        if step.get("name", "").startswith("Declare the executed-test floor")
    )
    assert "format(' --passed-ids {0}', matrix.docs_passed_ids)" in floor

    single: set[str] = set()
    pair: set[str] = set()
    for backend, row in rows.items():
        docs = _attested_docs(
            REPO,
            row.get("docs_passed_ids"),
            selector.replace(marker, row["marker"]),
            _installed_markers(rows, row["install"]),
            marks,
            required,
        )
        single.update(docs)
        if not nightly:
            for node in docs:
                assert leg_should_run(backend, [pages[node]])[0], (
                    f"{pages[node]} must trigger the {backend} PR suite"
                )
        if "pair_install" in row:
            step, paths, pair_selector = _pytest_step(
                job, "Run the ${{ matrix.backend }} leg's cross-framework preview tests"
            )
            assert (paths, pair_selector) == (["tests/"], "${{ matrix.pair_marker }} and not slow")
            assert "uv pip install --system ${{ matrix.pair_install }}" in step["run"]
            assert "format(' --passed-ids {0}', matrix.pair_docs_passed_ids)" in step["run"]
            pair.update(
                _attested_docs(
                    REPO,
                    row.get("pair_docs_passed_ids"),
                    pair_selector.replace("${{ matrix.pair_marker }}", row["pair_marker"]),
                    _installed_markers(rows, f"{row['install']} {row['pair_install']}"),
                    marks,
                    required,
                )
            )
    return single, pair


def test_backend_docs_follow_workflow_routes(tmp_path: Path) -> None:
    """Generic recipes stay in Peer and each preview block has a passing venue."""
    inventory = tmp_path / "docs-collection.json"
    result = subprocess.run(
        [sys.executable, "-c", _COLLECT, str(inventory), *PEER_FILES],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    items = json.loads(inventory.read_text(encoding="utf-8"))
    marks = {node: set(names) for node, (names, _params) in items.items()}
    blocks = {
        (params["file_name"], params["block_index"], params["code"]): node
        for node, (_names, params) in items.items()
        if "code" in params
    }
    assert blocks.keys() == set(snippets._iter_python_blocks()), "missing or stale snippet cells"
    assert {
        params["page_name"] for _names, params in items.values() if "page_name" in params
    } == set(snippets.CANONICAL_PAGES)
    preview = set(snippets.BACKEND_MARKERS.values())
    required = {}
    pages = {}
    for (page, _index, code), node in blocks.items():
        needs = snippets._backend_markers(snippets._required_backend_modules(ast.parse(code), code))
        assert marks[node] & preview == needs, f"{node}: incorrect backend markers"
        if needs:
            required[node] = needs
            pages[node] = f"docs/{page}"

    peer = _workflow(_NIGHTLY_WORKFLOW)["jobs"]["peer-recipes"]
    _, paths, selector = _pytest_step(
        peer, "Run the recipe pages and doc snippets with the peers required"
    )
    assert paths == list(PEER_FILES)
    expression = Expression.compile(selector)
    selected = {
        node
        for node, names in marks.items()
        if not selector or expression.evaluate(names.__contains__)
    }
    assert selected == marks.keys() - required.keys(), (
        f"Recipe Peer selected preview docs {sorted(selected & required.keys())} "
        f"or omitted generic tests {sorted(marks.keys() - required.keys() - selected)}"
    )

    single, pair = _preview_coverage(_NIGHTLY_WORKFLOW, "preview-backends", marks, required, pages)
    assert single | pair == required.keys(), (
        f"no attested preview environment installs and selects {sorted(required.keys() - single - pair)}"
    )
    pr, _ = _preview_coverage(_PR_WORKFLOW, "backend-protocol", marks, required, pages)
    assert pr == single
    configured = {
        REPO / path
        for row in _legs(_NIGHTLY_WORKFLOW, "preview-backends").values()
        for field in ("docs_passed_ids", "pair_docs_passed_ids")
        if (path := row.get(field))
    }
    assert configured == set((REPO / "tests/fixtures/backend_docs").glob("*_passed_ids.txt"))


@pytest.mark.parametrize(
    ("code", "marks"),
    [
        ("import torch", set()),
        ("import tensorflow as tf", {"tf_backend"}),
        ("from tensorflow import keras", {"tf_backend"}),
        ("# import tensorflow\ntext = 'from jax import numpy'", set()),
        ("import jax.numpy as jnp", {"backend_jax"}),
        ("from jaxlib import xla_client", {"backend_jax"}),
        ("from tinygrad import Tensor", {"backend_tinygrad"}),
        ("import paddle as pd", {"backend_paddle"}),
        ("import mlx.core as mx", {"backend_mlx"}),
        ("import tensorflow\nimport jax", {"tf_backend", "backend_jax"}),
        ("import tensorflow\n# torch.compile", {"tf_backend", "heavy"}),
        ("if False:\n    import tensorflow", set()),
        ("# requires-backends: jax\ndef run():\n    import jax", {"backend_jax"}),
        ("# requires-backends: torch\nimport keras", set()),
    ],
)
def test_snippet_marks_follow_imports(
    monkeypatch: pytest.MonkeyPatch, code: str, marks: set[str]
) -> None:
    """Existing pytest parameters carry the required backend and cost markers."""
    monkeypatch.setattr(snippets, "_iter_python_blocks", lambda _pages: [("example.md", 1, code)])
    (param,) = snippets._doc_block_params("guides")
    assert {mark.name for mark in param.marks} == marks


@pytest.mark.parametrize(
    ("code", "diagnostic"),
    [
        ("import keras", "standalone Keras"),
        ("# requires-backends: unknown", "unknown requires-backends modules"),
    ],
)
def test_ambiguous_backend_requires_valid_declaration(
    monkeypatch: pytest.MonkeyPatch, code: str, diagnostic: str
) -> None:
    """An ambiguous runtime and a misspelled declaration fail collection."""
    monkeypatch.setattr(
        snippets,
        "_iter_python_blocks",
        lambda _pages: [("example.md", 1, code)],
    )
    with pytest.raises(ValueError, match=diagnostic):
        snippets._doc_block_params("guides")


def test_optional_backend_import_does_not_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A guarded import can use its fallback in an environment lacking the backend."""
    monkeypatch.setattr(snippets.importlib.util, "find_spec", lambda _module: None)
    code = "if False:\n    import tensorflow\nassert 2 + 2 == 4"
    snippets._run_doc_block("example.md", 1, code, tmp_path)
