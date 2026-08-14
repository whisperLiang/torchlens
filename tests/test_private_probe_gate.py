"""r-b4 R26-2: structural gate -- private torch API touches stay inside `_torch_compat`.

The CLAUDE.md rule ("every fragile torch-private-API probe routes through
``torchlens/utils/_torch_compat.py`` and flips a named ``HAS_*`` flag") was
unenforced: probes kept escaping the boundary (the fail-open census probe, the
unguarded dispatcher-schema census, the silent DTensor geometry import, the
``python -O``-stripped TensorBase assert, ...). This AST gate makes the rule
REAL: any private ``torch.*`` attribute access, private ``torch.*`` import, or
``getattr(torch..., "_name")`` literal outside ``_torch_compat`` must appear in
the reason-bearing ledger below, so a new escape is a reviewed contract diff,
never a silent drift.

The ledger is exact-equality in BOTH directions: fixing a touch requires
deleting its row (shrink-only), adding one requires writing a reason here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "torchlens"

#: The ONE sanctioned boundary module (plus its TF sibling for tf-private APIs).
_BOUNDARY_FILES = {
    "torchlens/utils/_torch_compat.py",
}

#: Reason-bearing ledger of every sanctioned private-torch touch OUTSIDE the
#: boundary. Every entry is fail-closed in direction and was individually
#: reviewed in the b4 R26 census; new rows require the same review.
_ALLOWED_PRIVATE_TOUCHES: dict[str, frozenset[str]] = {
    # Fail-closed CVE-surface / unpickler hardening checks: absence REFUSES the
    # load or shrinks the admitted class set, never degrades silently
    # (b4 R26 census, fable inventory).
    "torchlens/_io/_safe_unpickle.py": frozenset(
        {
            "torch._C",
            "getattr(torch, '_storage_classes')",
            "getattr(torch, '_tensor_classes')",
            "getattr(torch.storage, '_StorageBase')",
        }
    ),
    # Tracing-tensor classification fallback: absence falls through to the
    # compat-routed exact-type path plus structural matching; the residual is
    # disclosed by the compat row (HAS_TRACING_TENSOR_TYPES).
    "torchlens/_robustness.py": frozenset({"getattr(torch, '_is_functional_tensor')"}),
    # ``torch.ops.*`` __call__ class enumeration for the cross-thread escape
    # observers; an EMPTY structural scan fails CLOSED into
    # _HOST_ESCAPE_OBSERVER_FAILED.
    "torchlens/backends/torch/_completeness_cross_thread.py": frozenset({"import torch._ops"}),
    # Zero-copy escape-target enumeration: if the private ``_to_dlpack``
    # binding disappears, the escape route it guards disappears with it
    # (fail-neutral; nothing to patch means nothing can escape through it).
    "torchlens/backends/torch/_completeness_finalize.py": frozenset({"getattr(torch, '_C')"}),
    # TorchDispatchMode base class (public-by-usage, import fails the module
    # loudly at import time) and the same fail-closed torch.ops enumeration.
    "torchlens/backends/torch/completeness_witness.py": frozenset(
        {"from torch.utils._python_dispatch", "import torch._ops"}
    ),
    # Guarded getattr probes with explicit degraded fallbacks: graph-task-id
    # attribution returns None (capability absent, callers degrade typed).
    "torchlens/backends/torch/tensor_tracking.py": frozenset(
        {"torch._C", "getattr(torch._C, '_current_graph_task_id')"}
    ),
    # Expanded-weights identity shim (SF-53 census): the conv/RNN per-sample-grad
    # picker compares the dispatched func against torch's OWN
    # ``_cudnn_rnn_flatten_weight`` symbol, so the shim must read that exact private
    # binding to normalize the identity basis torch itself uses -- routing through a
    # compat wrapper would change the object identity the comparison depends on.
    # Guarded getattr, fail-neutral: if the symbol disappears, torch's special
    # case disappears with it and the shim falls through to the alias-table path.
    "torchlens/backends/torch/identity_shims.py": frozenset(
        {"getattr(torch, '_cudnn_rnn_flatten_weight')"}
    ),
}


def _is_private_segment(segment: str) -> bool:
    """Whether one dotted-path segment is private (single-underscore, not dunder)."""

    return segment.startswith("_") and not (segment.startswith("__") and segment.endswith("__"))


def _private_touches(tree: ast.AST) -> set[str]:
    """Collect private torch touches from one module's AST."""

    touches: set[str] = set()
    nested_attribute_values = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute)
    }

    def _torch_chain(node: ast.AST) -> list[str] | None:
        parts: list[str] = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name) and current.id == "torch":
            return list(reversed(parts))
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and id(node) not in nested_attribute_values:
            chain = _torch_chain(node)
            if chain and any(_is_private_segment(part) for part in chain):
                touches.add("torch." + ".".join(chain))
        elif isinstance(node, ast.Call):
            # getattr(torch..., "_private", ...) literal probes.
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
                and _is_private_segment(node.args[1].value)
            ):
                base = node.args[0]
                chain = _torch_chain(base) if isinstance(base, ast.Attribute) else None
                if (isinstance(base, ast.Name) and base.id == "torch") or chain is not None:
                    prefix = "torch" if chain is None else "torch." + ".".join(chain)
                    touches.add(f"getattr({prefix}, {node.args[1].value!r})")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                segments = alias.name.split(".")
                if segments[0] == "torch" and any(_is_private_segment(s) for s in segments[1:]):
                    touches.add(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            segments = module.split(".")
            if (
                node.level == 0
                and segments[0] == "torch"
                and any(_is_private_segment(s) for s in segments[1:])
            ):
                touches.add(f"from {module}")
    return touches


def _scan_package() -> dict[str, set[str]]:
    """Scan every torchlens module for private torch touches."""

    found: dict[str, set[str]] = {}
    repo_root = _PACKAGE_ROOT.parent
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(repo_root).as_posix()
        if rel in _BOUNDARY_FILES:
            continue
        touches = _private_touches(ast.parse(path.read_text(encoding="utf-8"), filename=rel))
        if touches:
            found[rel] = touches
    return found


def test_private_torch_touches_match_the_ledger_exactly() -> None:
    """Every private torch touch outside `_torch_compat` is ledgered, both ways."""

    found = _scan_package()
    unsanctioned = {
        rel: sorted(touches - _ALLOWED_PRIVATE_TOUCHES.get(rel, frozenset()))
        for rel, touches in found.items()
        if touches - _ALLOWED_PRIVATE_TOUCHES.get(rel, frozenset())
    }
    assert not unsanctioned, (
        "Private torch API touch(es) outside torchlens/utils/_torch_compat.py: "
        f"{unsanctioned}. Route the probe through a _torch_compat accessor with a "
        "named HAS_* flag (CLAUDE.md rule), or -- for a genuinely fail-closed "
        "touch -- add a reason-bearing ledger row in this test."
    )
    stale = {
        rel: sorted(allowed - found.get(rel, set()))
        for rel, allowed in _ALLOWED_PRIVATE_TOUCHES.items()
        if allowed - found.get(rel, set())
    }
    assert not stale, (
        f"Ledgered private-touch row(s) no longer exist in the code: {stale}. "
        "Delete the stale ledger rows (the ledger is shrink-only)."
    )


def test_gate_scanner_detects_planted_offenders() -> None:
    """Planted positives: the scanner sees attribute, import, and getattr forms."""

    planted = ast.parse(
        "import torch._dynamo\n"
        "from torch.utils._python_dispatch import _get_current_dispatch_mode_stack\n"
        "x = torch._C._jit_get_all_schemas()\n"
        "y = getattr(torch._C, '_TensorBase', None)\n"
        "z = getattr(torch, '_VF', None)\n"
    )
    touches = _private_touches(planted)
    assert "import torch._dynamo" in touches
    assert "from torch.utils._python_dispatch" in touches
    assert "torch._C._jit_get_all_schemas" in touches
    assert "getattr(torch._C, '_TensorBase')" in touches
    assert "getattr(torch, '_VF')" in touches
