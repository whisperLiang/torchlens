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

# Module-wide smoke dropped (r3settle2 budget lint): the full-package AST
# ledger scan below measures over the 5s smoke partition; per-test marks.

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
    # Expanded-weights identity shim (SF-53 census): the conv/RNN per-sample-grad
    # picker compares the dispatched func against torch's OWN
    # ``_cudnn_rnn_flatten_weight`` symbol, so the shim must read that exact private
    # binding to normalize the identity basis torch itself uses -- routing through a
    # compat wrapper would change the object identity the comparison depends on.
    # Guarded getattr, fail-neutral: if the symbol disappears, torch's special
    # case disappears with it and the shim falls through to the alias-table path.
    # The three string-literal ``import_module`` reads (surfaced by the r6 R26
    # scanner extension) are the SAME shim's module handles: gated on the
    # HAS_EXPANDED_WEIGHTS_CONV_PICKER capability flag and wrapped in
    # ``except ImportError: return`` (fail-neutral -- no expanded-weights
    # machinery means nothing to shim).
    "torchlens/backends/torch/identity_shims.py": frozenset(
        {
            "getattr(torch, '_cudnn_rnn_flatten_weight')",
            "import_module('torch.nn.utils._expanded_weights.conv_expanded_weights')",
            "import_module('torch.nn.utils._expanded_weights.conv_utils')",
            "import_module('torch.nn.utils._expanded_weights.expanded_weights_impl')",
        }
    ),
    # Forward-pre-hook provenance interposition (grind-r6 b4 R26, sol MED --
    # the aliased-import touch this scanner extension exists to see): the
    # interposer must read and patch torch's REAL global pre-hook registry
    # (``torch.nn.modules.module._global_forward_pre_hooks``) because the
    # registry OBJECT IDENTITY is what torch's own Module.__call__ consults;
    # a compat-layer copy would observe nothing. Reversible interposition;
    # a registration that bypasses it is disclosed per-snapshot as
    # ``registration_interposition_bypassed``, never silently missed.
    "torchlens/backends/torch/prehook_provenance.py": frozenset(
        {"torch.nn.modules.module._global_forward_pre_hooks"}
    ),
    # Conventional stable private BASE CLASSES read for isinstance
    # classification (norm/dropout family detection). Unguarded on purpose:
    # if torch ever removes them the read fails LOUDLY at call time -- there
    # is no silent-degradation path for a capability flag to disclose.
    "torchlens/data_classes/_trace_validation.py": frozenset(
        {"torch.nn.modules.batchnorm._BatchNorm"}
    ),
    "torchlens/intervention/rerun.py": frozenset(
        {
            "torch.nn.modules.batchnorm._BatchNorm",
            "torch.nn.modules.dropout._DropoutNd",
        }
    ),
}


def _is_private_segment(segment: str) -> bool:
    """Whether one dotted-path segment is private (single-underscore, not dunder)."""

    return segment.startswith("_") and not (segment.startswith("__") and segment.endswith("__"))


def _torch_import_aliases(tree: ast.AST) -> dict[str, str]:
    """Map locally-bound names to the full ``torch.*`` dotted paths they alias.

    grind-r6 b4 R26 (sol MED): ``import torch.nn.modules.module as
    torch_module`` followed by ``torch_module._global_forward_pre_hooks``
    was invisible to the gate -- the import path has no private segment and
    the attribute chain roots at the alias, not at ``torch``. Both aliased
    ``import ... as`` bindings and ``from torch.x import y [as z]`` bindings
    become recognized chain roots.
    """

    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "torch" and alias.asname:
                    aliases[alias.asname] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0 and module.split(".")[0] == "torch":
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    aliases[alias.asname or alias.name] = f"{module}.{alias.name}"
    return aliases


def _private_touches(tree: ast.AST) -> set[str]:
    """Collect private torch touches from one module's AST."""

    touches: set[str] = set()
    aliases = _torch_import_aliases(tree)
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
        if isinstance(current, ast.Name) and current.id in aliases:
            # Resolve the alias to its full dotted path, dropping the
            # leading "torch" so callers can re-prefix uniformly.
            return aliases[current.id].split(".")[1:] + list(reversed(parts))
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
            # importlib.import_module("torch._x") / __import__("torch._x")
            # string-literal forms (grind-r6 b4 R26): a private torch module
            # imported by string never appears as an Import node.
            func = node.func
            is_import_call = (isinstance(func, ast.Name) and func.id == "__import__") or (
                isinstance(func, ast.Attribute)
                and func.attr == "import_module"
                and isinstance(func.value, ast.Name)
                and func.value.id == "importlib"
            )
            if (
                is_import_call
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                target = node.args[0].value
                segments = target.split(".")
                if segments[0] == "torch" and any(_is_private_segment(s) for s in segments[1:]):
                    touches.add(f"import_module({target!r})")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                segments = alias.name.split(".")
                if segments[0] == "torch" and any(_is_private_segment(s) for s in segments[1:]):
                    touches.add(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            segments = module.split(".")
            if node.level == 0 and segments[0] == "torch":
                if any(_is_private_segment(s) for s in segments[1:]):
                    touches.add(f"from {module}")
                else:
                    # Private NAME imported from a public torch module
                    # (``from torch.utils import _pytree``): the module path
                    # alone carries no private segment (grind-r6 b4 R26).
                    for alias in node.names:
                        if _is_private_segment(alias.name):
                            touches.add(f"from {module} import {alias.name}")
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


@pytest.mark.heavy
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


@pytest.mark.smoke
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


@pytest.mark.smoke
def test_gate_scanner_detects_aliased_and_string_literal_offenders() -> None:
    """grind-r6 b4 R26 (sol MED): the scanner blind spots, planted.

    Before the extension every one of these forms passed the gate silently:
    the aliased module import roots the attribute chain at the alias name,
    the from-import binds a private name off a public module path, and the
    string-literal import never produces an Import node at all.
    """

    planted = ast.parse(
        "import importlib\n"
        "import torch.nn.modules.module as torch_module\n"
        "from torch import nn\n"
        "from torch.utils import _pytree\n"
        "from torch.nn.modules import module as mod_alias\n"
        "a = torch_module._global_forward_pre_hooks\n"
        "b = nn.modules.batchnorm._BatchNorm\n"
        "c = mod_alias._global_backward_hooks\n"
        "d = importlib.import_module('torch.nn.utils._expanded_weights.conv_utils')\n"
        "e = __import__('torch._dynamo')\n"
    )
    touches = _private_touches(planted)
    assert "torch.nn.modules.module._global_forward_pre_hooks" in touches
    assert "torch.nn.modules.batchnorm._BatchNorm" in touches
    assert "torch.nn.modules.module._global_backward_hooks" in touches
    assert "from torch.utils import _pytree" in touches
    assert "import_module('torch.nn.utils._expanded_weights.conv_utils')" in touches
    assert "import_module('torch._dynamo')" in touches
