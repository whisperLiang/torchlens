"""Import-time hygiene regression tests.

The measured import state is healthy -- TorchLens adds ~0.15 s and 30 of its own
modules on top of torch, pulls no heavy third-party dependency, and is
warning-clean. This file is the TRIPWIRE for that state, and grind b4 (R31-1)
found the tripwire could not fail on any of the three ways it can regress:

(a) the denylist covered torchvision and two ``torch._dynamo`` names and nothing
    else, while the live route to a heavy dependency exists -- ``options.py``
    executes ``visualization/__init__`` on every bare import, one hop from
    modules doing top-level ``import graphviz`` / ``from PIL import Image``;
(b) no module-count ceiling and no import-duration budget existed anywhere;
(c) only a handful of the lazy facades were asserted deferred, so a new eager
    ``from .viz import ...`` would silently pull PIL;
(d) the whole file sat outside the smoke/PR tier, so an eager-wrap regression
    merged green and surfaced a day later, misattributed;
(e) no warning-on-import gate;
(f) two guards did not pin WHICH torchlens they imported -- on a box with an
    installed wheel they audited the wheel, not the checkout.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]

_LAZY_MODULE_CASES = (
    ("fastlog", "record"),
    ("intervention", "func"),
    ("user_funcs", "trace"),
    ("data_classes", "Buffer"),
)

#: Third-party packages a bare ``import torchlens`` must never pull. FROZEN and
#: grow-only: every name here is either a real one-hop risk from the eagerly
#: imported ``torchlens.visualization`` (graphviz, PIL) or an expensive optional
#: integration that belongs behind a lazy facade. ``numpy`` is deliberately
#: absent -- torch imports it itself, so it is not TorchLens's to defer.
_HEAVY_IMPORT_DENYLIST = frozenset(
    {
        "IPython",
        "PIL",
        "graphviz",
        "matplotlib",
        "pandas",
        "pyarrow",
        "scipy",
        "sklearn",
        "torchvision",
        "transformers",
        "torch._dynamo",
        "torch._dynamo.eval_frame",
    }
)

#: Every torchlens module a bare import is allowed to execute. SHRINK-ONLY: the
#: test demands equality, so adding an eager module fails and must be argued for
#: in the diff, while making one lazy fails with "delete the row" -- the same
#: discipline tests/test_module_import_isolation.py uses for import cycles.
_EAGER_TORCHLENS_MODULES = frozenset(
    {
        "torchlens",
        "torchlens._deprecations",
        "torchlens._errors",
        "torchlens._io",
        "torchlens._literals",
        "torchlens._save_budget",
        "torchlens._state",
        "torchlens.captured_run",
        "torchlens.errors",
        "torchlens.errors._base",
        "torchlens.errors.runnable",
        "torchlens.ir",
        "torchlens.ir.capture_events",
        "torchlens.ir.container",
        "torchlens.ir.container_registry",
        "torchlens.ir.events",
        "torchlens.ir.intervention",
        "torchlens.ir.live_index",
        "torchlens.ir.op_record",
        "torchlens.ir.predicate",
        "torchlens.ir.refs",
        "torchlens.ir.semantics",
        "torchlens.ir.workspaces",
        "torchlens.observers",
        "torchlens.options",
        "torchlens.quantities",
        "torchlens.utils",
        "torchlens.utils._multipass_access",
        "torchlens.visualization",
        "torchlens.visualization.node_spec",
    }
)

#: Lazy-facade module paths that a bare import legitimately executes anyway,
#: with the reason. Shrink-only, like the eager set above.
_EAGERLY_IMPORTED_LAZY_TARGETS = {
    "torchlens._io": (
        "the package eagerly imports torchlens._io for the error/warning classes "
        "(ArtifactSchemaAgeWarning and friends) that the top-level surface "
        "re-exports; the lazy _LAZY_ATTRS entries pointing here are for its "
        "heavier members, which stay deferred inside the module"
    ),
}

#: Non-torchlens modules a bare import may add BEYOND what torch already
#: imported. Measured at 16 (html, packaging, sysconfig). The ceiling is
#: deliberately loose -- it exists to catch a heavy dependency creeping in as an
#: order-of-magnitude jump, which the denylist can only catch by name.
_MAX_MARGINAL_NON_TORCHLENS_MODULES = 40

#: Duration budget for the torchlens import itself, measured with torch already
#: imported so the number is not dominated by torch. Charged on min(wall, cpu)
#: -- the tier-budget convention (eec14a0f) -- because the pure-wall version of
#: this guard false-failed at 1.45s under orchestrator load while five fresh
#: control runs measured 0.32-0.78s and every structural guard stayed green
#: (grind b4, F31-A): wall stretches with box load, CPU time does not, and an
#: eager heavy import inflates BOTH. Measured ~0.15 s marginal CPU on the
#: devbox; the budget is ~10x that. This is a regression tripwire for an
#: order-of-magnitude change (an eager heavy import), NOT a performance gate --
#: perf lives in tests/bench/. Documented residual, shared with the tier
#: budgets: an import that only SLEEPS is no longer catchable here; the module
#: allowlist and denylist above remain the structural authority.
_TORCHLENS_IMPORT_BUDGET_S = 1.5


def _import_probe_script() -> str:
    """Build a fresh-interpreter probe emitting one JSON blob of import facts.

    One subprocess serves every ceiling/denylist assertion below, so the whole
    R31 block costs a single torch import instead of one per check.

    Returns
    -------
    str
        Python source printing a JSON object on its last line.
    """

    return """
import json, sys, time

before_torch = set(sys.modules)
import torch
after_torch = set(sys.modules)

start_wall = time.perf_counter()
start_cpu = time.process_time()
import torchlens
elapsed_cpu = time.process_time() - start_cpu
elapsed_wall = time.perf_counter() - start_wall

after = set(sys.modules)
torchlens_modules = sorted(
    name for name in after if name == "torchlens" or name.startswith("torchlens.")
)
marginal_foreign = sorted(
    name
    for name in after - after_torch
    if not (name == "torchlens" or name.startswith("torchlens."))
)
lazy_targets = sorted({target for target, _attr in torchlens._LAZY_ATTRS.values()})
print(json.dumps({
    "elapsed_wall": elapsed_wall,
    "elapsed_cpu": elapsed_cpu,
    "file": torchlens.__file__,
    "loaded": sorted(after),
    "torchlens_modules": torchlens_modules,
    "marginal_foreign": marginal_foreign,
    "lazy_targets": lazy_targets,
    "eager_lazy_targets": [t for t in lazy_targets if t in after],
}))
"""


def _run_import_script(script: str) -> None:
    """Run an import assertion against this checkout in a fresh interpreter.

    Parameters
    ----------
    script:
        Python source containing assertions for one import pattern.
    """

    _run_import_script_capturing(script)


def _run_import_script_capturing(script: str) -> str:
    """Run ``script`` against THIS checkout and return its stdout.

    Every guard in this file routes through here so none of them can silently
    audit an installed wheel instead of the working tree (grind b4, R31-1f): the
    checkout goes on ``PYTHONPATH`` and the probe asserts which ``torchlens``
    actually got imported.

    Parameters
    ----------
    script:
        Python source to execute.

    Returns
    -------
    str
        Captured stdout.
    """

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(_REPO_ROOT)
    prologue = (
        "import torchlens as _tl_checkout_probe, pathlib as _pathlib\n"
        f"_expected = _pathlib.Path({str(_REPO_ROOT)!r}).resolve()\n"
        "_actual = _pathlib.Path(_tl_checkout_probe.__file__).resolve()\n"
        "assert _expected in _actual.parents, (\n"
        "    'guard imported the wrong torchlens: ' + str(_actual)\n"
        ")\n"
    )
    # The checkout assertion runs AFTER the caller's script, so it cannot perturb
    # a measurement that times the torchlens import itself.
    completed = subprocess.run(
        [sys.executable, "-c", script + "\n" + prologue],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, (
        f"import guard subprocess failed ({completed.returncode}):\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )
    return completed.stdout


@pytest.fixture(scope="module")
def import_facts() -> dict[str, object]:
    """Return import facts measured once in a fresh interpreter.

    Returns
    -------
    dict[str, object]
        Parsed output of :func:`_import_probe_script`.
    """

    stdout = _run_import_script_capturing(_import_probe_script())
    payload = [line for line in stdout.splitlines() if line.startswith("{")][-1]
    return json.loads(payload)


def _lazy_facade_fidelity_script() -> str:
    """Build a fresh-interpreter check for every root lazy facade.

    Returns
    -------
    str
        Python source that compares root facade objects with direct imports.
    """

    return """
import importlib
import pkgutil
import torchlens as tl

facades = {
    name: module_path
    for name, (module_path, attr_name) in tl._LAZY_ATTRS.items()
    if attr_name is None
}
shim_private_targets = {"_trace": ("torchlens.user_funcs", "trace")}
for private_name, (module_path, attr_name) in shim_private_targets.items():
    assert getattr(tl, private_name) is getattr(importlib.import_module(module_path), attr_name)
collisions = {}
for facade_name, module_path in facades.items():
    eager_module = importlib.import_module(module_path)
    facade_module = getattr(tl, facade_name)
    assert facade_module is eager_module
    public_names = getattr(eager_module, "__all__", ())
    child_modules = {
        child.name.rsplit(".", 1)[-1]
        for child in getattr(eager_module, "__path__", ())
        and pkgutil.iter_modules(eager_module.__path__, eager_module.__name__ + ".")
    }
    collisions[facade_name] = sorted(set(public_names) & child_modules)
    expected_exports = {}
    if facade_name == "fastlog":
        for public_name in collisions[facade_name]:
            child_module = importlib.import_module(f"{module_path}.{public_name}")
            assert vars(eager_module)[public_name] is child_module
            source_module, source_name = eager_module._LAZY_ATTRS[public_name]
            expected_exports[public_name] = getattr(
                importlib.import_module(source_module), source_name
            )
    for public_name in public_names:
        expected = expected_exports.get(public_name, getattr(eager_module, public_name))
        assert getattr(facade_module, public_name) is expected

assert collisions == {
    "attribution": [], "compat": ["lovely", "torchextractor", "torchshow"],
    "data_classes": [], "debug": [], "distributed": [], "examples": [],
    "experimental": ["dagua", "node_styles"], "export": [],
    "fastlog": ["dry_run", "recover"], "intervention": ["replay", "rerun", "sites"],
    "hash": [], "io": [], "merged": [], "partial": [], "report": [], "repgeom": [],
    "receptive_field": ["rules"], "stats": [], "user_funcs": [], "validation": [], "viz": [],
}

"""


@pytest.mark.smoke
def test_bare_import_pulls_no_heavy_third_party_dependency(
    import_facts: dict[str, object],
) -> None:
    """No package on the frozen denylist may be imported by a bare import.

    R31-1a. The route is live, not hypothetical: ``options.py`` executes
    ``visualization/__init__`` on every bare import, and sibling modules in that
    package (``_render_common.py``, ``_render_utils.py``, ``renderers/
    graphviz.py``) import graphviz and PIL at module level. One accidental
    re-export away.
    """

    loaded = set(import_facts["loaded"])  # type: ignore[arg-type]
    offenders = sorted(loaded & _HEAVY_IMPORT_DENYLIST)
    assert not offenders, (
        "bare `import torchlens` pulled heavy dependencies -- move the import "
        f"behind a lazy facade or a function-local import: {offenders}"
    )


@pytest.mark.smoke
def test_eager_module_set_is_exactly_the_declared_allowlist(
    import_facts: dict[str, object],
) -> None:
    """The eager torchlens module set is pinned and shrink-only (R31-1b)."""

    actual = set(import_facts["torchlens_modules"])  # type: ignore[arg-type]
    added = sorted(actual - _EAGER_TORCHLENS_MODULES)
    assert not added, (
        "these torchlens modules became EAGER on a bare import. Defer them, or "
        "add them to _EAGER_TORCHLENS_MODULES in the same diff with a reason: "
        f"{added}"
    )
    removed = sorted(_EAGER_TORCHLENS_MODULES - actual)
    assert not removed, f"these modules are no longer eager (good -- delete their rows): {removed}"


@pytest.mark.smoke
def test_bare_import_adds_few_foreign_modules(import_facts: dict[str, object]) -> None:
    """A bare import adds few non-torchlens modules beyond torch (R31-1b).

    Complements the denylist, which can only catch dependencies it knows to name.
    """

    marginal = list(import_facts["marginal_foreign"])  # type: ignore[arg-type]
    assert len(marginal) <= _MAX_MARGINAL_NON_TORCHLENS_MODULES, (
        f"bare import added {len(marginal)} non-torchlens modules beyond torch "
        f"(ceiling {_MAX_MARGINAL_NON_TORCHLENS_MODULES}); a new dependency has "
        f"probably become eager: {marginal}"
    )


@pytest.mark.smoke
def test_bare_import_stays_within_its_duration_budget(
    import_facts: dict[str, object],
) -> None:
    """Importing torchlens over an already-imported torch stays cheap (R31-1b).

    Measured with torch pre-imported so the figure is TorchLens's own marginal
    cost rather than torch's; charged on min(wall, cpu) so orchestrator load
    cannot red the commit gate (F31-A). Budget is ~10x measured: this catches an
    eager heavy import, not a few milliseconds of drift.
    """

    wall = float(import_facts["elapsed_wall"])  # type: ignore[arg-type]
    cpu = float(import_facts["elapsed_cpu"])  # type: ignore[arg-type]
    charged = min(wall, cpu)
    assert charged < _TORCHLENS_IMPORT_BUDGET_S, (
        f"importing torchlens charged {charged:.3f}s (wall {wall:.3f}s, cpu "
        f"{cpu:.3f}s) over an already-imported torch (budget "
        f"{_TORCHLENS_IMPORT_BUDGET_S}s). Both measures are high, so this is "
        "real import work, not box load: something heavy likely became eager. "
        "The module allowlist/denylist guards in this file name the culprit "
        "when it is an eager import."
    )


@pytest.mark.smoke
def test_every_lazy_facade_target_is_actually_deferred(
    import_facts: dict[str, object],
) -> None:
    """EVERY ``_LAZY_ATTRS`` target module stays unimported (R31-1c).

    Derived from the shipped table inside the probe, so a facade added later is
    covered without editing this test -- previously only a handful of the 30
    distinct target modules were checked, and a new eager ``from .viz import ...``
    would have pulled PIL unnoticed.
    """

    targets = list(import_facts["lazy_targets"])  # type: ignore[arg-type]
    assert len(targets) > 20, f"only {len(targets)} lazy targets found; table misread"
    eager = set(import_facts["eager_lazy_targets"])  # type: ignore[arg-type]
    undeclared = sorted(eager - set(_EAGERLY_IMPORTED_LAZY_TARGETS))
    assert not undeclared, (
        "these lazy-facade targets were imported eagerly, defeating the facade. "
        "Defer them, or declare them in _EAGERLY_IMPORTED_LAZY_TARGETS with the "
        f"reason: {undeclared}"
    )
    healed = sorted(set(_EAGERLY_IMPORTED_LAZY_TARGETS) - eager)
    assert not healed, f"now properly deferred (delete their rows): {healed}"


@pytest.mark.smoke
def test_bare_import_is_warning_clean() -> None:
    """A bare import emits no warnings at all (R31-1e).

    Import is warning-clean today; nothing enforced it. Run with ``-W error`` in
    a fresh interpreter so any warning raised during import fails the guard.
    """

    _run_import_script_capturing(
        "import warnings\n"
        "warnings.simplefilter('error')\n"
        "import torchlens\n"
        "assert torchlens.__version__\n"
    )


def test_all_lazy_facades_match_direct_import_public_names() -> None:
    """Pin public-name fidelity and audit child-module collisions for every facade."""

    _run_import_script(_lazy_facade_fidelity_script())


@pytest.mark.parametrize(("module_name", "member_name"), _LAZY_MODULE_CASES)
def test_lazified_module_top_level_access_patterns(module_name: str, member_name: str) -> None:
    """Top-level attributes and from-imports should load each deferred module."""

    _run_import_script(
        "import torchlens as tl; "
        f"module = tl.{module_name}; "
        f"from torchlens import {module_name} as imported_module; "
        "assert module is imported_module; "
        f"assert hasattr(module, {member_name!r})"
    )


@pytest.mark.parametrize(("module_name", "member_name"), _LAZY_MODULE_CASES)
def test_lazified_module_direct_import_patterns(module_name: str, member_name: str) -> None:
    """Direct submodule and member imports should work from a bare package import."""

    _run_import_script(
        "import importlib; "
        "import torchlens; "
        f"module = importlib.import_module('torchlens.{module_name}'); "
        f"from torchlens.{module_name} import {member_name}; "
        f"assert getattr(module, {member_name!r}) is {member_name}"
    )


@pytest.mark.smoke
def test_bare_import_defers_lazified_feature_modules() -> None:
    """Bare imports must not eagerly initialize the deferred feature islands."""

    _run_import_script(
        "import sys; import torchlens; "
        "blocked = ('torchlens.fastlog', 'torchlens.intervention', "
        "'torchlens.user_funcs', 'torchlens.data_classes'); "
        "assert not any(name == prefix or name.startswith(prefix + '.') "
        "for prefix in blocked for name in sys.modules)"
    )


@pytest.mark.smoke
def test_bare_import_leaves_torch_functions_undecorated() -> None:
    """Bare TorchLens import must not eagerly wrap torch operators."""

    _run_import_script(
        "import torch; "
        "import torchlens; "
        "from torchlens import _state; "
        "from torchlens.backends.torch._tl import is_decorated_function; "
        "assert _state._is_decorated is False; "
        "assert not is_decorated_function(torch.cos)"
    )


def test_first_trace_lazily_wraps_torch_functions() -> None:
    """The first torch capture should install the persistent wrappers."""

    _run_import_script(
        "import torch; "
        "import torchlens as tl; "
        "from torchlens import _state; "
        "from torchlens.backends.torch._tl import is_decorated_function; "
        "model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU()); "
        "tl.trace(model, torch.ones(1, 2)); "
        "assert _state._is_decorated is True; "
        "assert is_decorated_function(torch.cos)"
    )


@pytest.mark.smoke
def test_import_torchlens_does_not_import_torchvision_when_installed() -> None:
    """Bare TorchLens import should not import torchvision even when installed."""

    if importlib.util.find_spec("torchvision") is None:
        pytest.skip("torchvision is not installed")

    # Routed through the helper (R31-1f): the bare subprocess.run this used to
    # call set no PYTHONPATH, so on a box with torchlens installed as a wheel it
    # audited the wheel and reported green about code that was never checked.
    _run_import_script_capturing("import torchlens, sys; assert 'torchvision' not in sys.modules")


@pytest.mark.smoke
def test_import_torchlens_does_not_import_heavy_torch_submodules() -> None:
    """Bare TorchLens import should not force deferred torch internals."""

    _run_import_script_capturing(
        "import torchlens, sys; "
        "assert 'torch._dynamo' not in sys.modules; "
        "assert 'torch._dynamo.eval_frame' not in sys.modules"
    )


@pytest.mark.heavy
@pytest.mark.optional
def test_torchvision_model_trace_still_covers_torchvision_ops() -> None:
    """First wrapper use should still include torchvision ops and trace torchvision models."""

    torchvision_models = pytest.importorskip("torchvision.models")

    import torchlens as tl
    from torchlens.constants import TORCHVISION_FUNCS, get_orig_torch_funcs

    wrapped_targets = set(get_orig_torch_funcs())
    assert set(TORCHVISION_FUNCS).issubset(wrapped_targets)

    model = torchvision_models.resnet18(weights=None).eval()
    x = torch.randn(1, 3, 32, 32)
    trace = tl.trace(model, x, save=tl.func("conv2d"))

    assert trace.num_layers > 0
    assert any(op.func_name == "conv2d" for op in trace.ops)


@pytest.mark.optional
def test_torchvision_cpp_ops_record_real_func_name() -> None:
    """Torchvision PyCapsule ops keep their public op name in TorchLens metadata."""

    torchvision_ops = pytest.importorskip("torchvision.ops")

    import torchlens as tl

    class NmsModel(torch.nn.Module):
        """Tiny module that calls torchvision NMS."""

        def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
            """Run torchvision NMS on boxes and scores."""

            boxes, scores = inputs
            return torchvision_ops.nms(boxes, scores, 0.5)

    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 1.1, 1.1], [3.0, 3.0, 4.0, 4.0]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    trace = tl.trace(NmsModel(), (boxes, scores), layers_to_save="all")

    assert any(op.func_name == "nms" for op in trace.ops)
    assert any(op.has_saved_activation for op in trace.ops if op.func_name == "nms")
