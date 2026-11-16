"""Observer fitted provenance and explicit Split optimizer ownership contracts.

Observation and fitted transforms must not train a caller's model. The explicit
Split training API may execute a caller-supplied optimizer in its six existing
step functions. It never constructs one. The AST check enforces this distinction,
and planted violations prove that licensing a step cannot license construction.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO / "torchlens"

#: Reason-bearing exemption ledger for the no-optimizer lint. CLOSED: a new
#: row requires the same justification class (a self-contained diagnostic
#: that constructs its own model and never mutates user state).
OPTIMIZER_LINT_EXEMPT: dict[str, str] = {
    "_observability/_memory_parity.py": (
        "pinned torch-memory-profiler parity scenario: builds its own MLP + "
        "SGD to exercise every allocator category; never touches user state"
    ),
}

#: Receivers whose ``.step()`` is optimizer-shaped. TorchLens's own step
#: APIs (collector.step, watch.step) are counters, not parameter updates,
#: and stay out of this pattern deliberately.
_OPTIMIZER_RECEIVER = re.compile(r"(^|\.)_?(optimizer|optimizers?\[[^]]*\]|optim|opt|scaler)$")


def _package_files() -> list[Path]:
    return [path for path in sorted(PACKAGE_ROOT.rglob("*.py")) if "__pycache__" not in path.parts]


def _is_optimizer_construction(node: ast.Call) -> bool:
    """True for ``torch.optim.X(...)`` / ``optim.X(...)`` constructions."""

    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    parent = func.value
    dotted = []
    while isinstance(parent, ast.Attribute):
        dotted.append(parent.attr)
        parent = parent.value
    if isinstance(parent, ast.Name):
        dotted.append(parent.id)
    return "optim" in dotted


# Explicit training functions consume an optimizer supplied by their caller.
# A license covers only .step(), never optimizer construction or other functions.
SPLIT_OPTIMIZER_STEPS: dict[tuple[str, str], str] = {
    ("split/_microbatch.py", "train_torch_microbatches"): "one logical batch update",
    ("split/training.py", "_execute_torch_suffix"): "suffix parameter update",
    ("split/training.py", "_train_suffix_paddle"): "native Paddle suffix update",
    ("split/training.py", "_tinygrad_optimizer_step"): "native tinygrad update",
    ("split/training.py", "_backward_prefix_torch"): "prefix update after gradient handoff",
    ("split/training.py", "_backward_prefix_paddle"): "native Paddle prefix update",
}


def _optimizer_findings(source: str, rel: str) -> list[str]:
    """Find construction and unlicensed steps, including inside licensed functions."""

    tree = ast.parse(source)
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _is_optimizer_construction(node):
            findings.append(f"{rel}:{node.lineno} optimizer construction")
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "step"
            and _OPTIMIZER_RECEIVER.search(ast.unparse(func.value))
        ):
            continue
        owners = [owner for owner in functions if owner.lineno <= node.lineno <= owner.end_lineno]
        owner = min(owners, key=lambda item: item.end_lineno - item.lineno) if owners else None
        if owner is not None and (rel, owner.name) in SPLIT_OPTIMIZER_STEPS:
            args = (*owner.args.posonlyargs, *owner.args.args, *owner.args.kwonlyargs)
            if any(arg.arg == "optimizer" for arg in args) or (
                owner.name == "_execute_torch_suffix"
                and any(
                    arg.arg == "options" and ast.unparse(arg.annotation) == "_TorchSuffixOptions"
                    for arg in args
                )
            ):
                continue
        findings.append(f"{rel}:{node.lineno} optimizer .step() call")
    return findings


def test_no_optimizer_construction_or_step_in_package() -> None:
    """Observers never fit; Split training only steps caller-owned optimizers."""

    findings: list[str] = []
    for path in _package_files():
        rel = str(path.relative_to(PACKAGE_ROOT))
        if rel not in OPTIMIZER_LINT_EXEMPT:
            findings.extend(_optimizer_findings(path.read_text(), rel))
    assert findings == [], f"optimizer ownership violations: {findings}"


@pytest.mark.parametrize("rel,name", sorted(SPLIT_OPTIMIZER_STEPS))
def test_split_step_license_cannot_hide_construction_or_unlicensed_training(rel, name) -> None:
    """An allowed function still rejects construction, missing ownership and nested steps."""

    source = f"def {name}(optimizer):\n    optimizer.step()\n"
    assert _optimizer_findings(source, rel) == []
    assert (
        "optimizer construction"
        in _optimizer_findings(source + "    torch.optim.SGD(parameters)\n", rel)[0]
    )
    assert (
        "optimizer .step() call"
        in _optimizer_findings(source.replace(name, "unlicensed_training"), rel)[0]
    )
    assert (
        "optimizer .step() call" in _optimizer_findings(source.replace("(optimizer)", "()"), rel)[0]
    )
    assert (
        "optimizer .step() call"
        in _optimizer_findings(source + "    def nested():\n        optimizer.step()\n", rel)[0]
    )


def test_split_step_license_rows_still_name_caller_owned_steps() -> None:
    """Deleted functions and removed caller parameters cannot leave a stale license."""

    for (rel, name), reason in SPLIT_OPTIMIZER_STEPS.items():
        tree = ast.parse((PACKAGE_ROOT / rel).read_text())
        functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        assert len(functions) == 1 and reason
        args = (
            *functions[0].args.posonlyargs,
            *functions[0].args.args,
            *functions[0].args.kwonlyargs,
        )
        assert any(arg.arg == "optimizer" for arg in args) or (
            name == "_execute_torch_suffix"
            and any(
                arg.arg == "options" and ast.unparse(arg.annotation) == "_TorchSuffixOptions"
                for arg in args
            )
        )


def test_exemption_ledger_rows_still_exist() -> None:
    """The ledger is shrink-only: a stale row must be deleted, not kept."""

    stale = sorted(rel for rel in OPTIMIZER_LINT_EXEMPT if not (PACKAGE_ROOT / rel).exists())
    assert stale == [], f"exemption rows for deleted files: {stale}"


def test_fitted_seam_spec_carries_the_provenance_field_set() -> None:
    """The accepting seam's emitted spec is FittedProvenance-shaped."""

    import torch

    import torchlens as tl

    pca = tl.stats.PCA(num_components=2)
    pca.update(torch.randn(20, 5))
    fitted = pca.fitted(fit_scope="f35-contract-pin")

    # The record itself: algorithm identity + data identity + geometry.
    assert fitted.digest.startswith("sha256:"), "fitting-data identity must be a content digest"
    assert fitted.num_samples == 20 and fitted.num_features == 5

    spec = tl.transforms.pca_apply(fitted)
    params = dict(spec.params)
    missing = {
        "source",  # the fitted record's content digest (refuse-silent-refit key)
        "fit_scope",  # fit population identity
        "in_extent",  # geometry a planner prices without executing
        "out_extent",
        "basis_digest",
    } - set(params)
    assert missing == set(), (
        f"fitted-provenance fields dropped from the accepting seam's spec: "
        f"{sorted(missing)} -- the field set is the contract (memo 3.2), not an "
        "implementation detail"
    )
    assert params["source"] == fitted.digest
    assert params["fit_scope"] == "f35-contract-pin"
    assert (params["in_extent"], params["out_extent"]) == (5, 2)
