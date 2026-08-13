"""The mechanical BELT: protocol-invisible stale-reference coverage.

Stage-2 safety net, part 2 (tri-lab verdict). A small class of wrapped torch
functions is invisible to EVERY ``TorchFunctionMode`` — their C
implementations never enter the override-protocol dispatch (measured: zero
callbacks), so a stale pre-wrap reference to one of them produces NO signal
anywhere: no mode callback, no aten event, and with the shipped-default
escape detector off the op would silently vanish from the trace. The rescue
re-run cannot fix what nothing detects, so these functions keep the targeted
stale-reference patching (module-level attributes, the measured reachable
holder class) after the broad crawler's deletion.

The belt membership is DERIVED MECHANICALLY per build, never hand-listed:

1. Candidates are every ``ORIG_TORCH_FUNCS`` entry whose ORIGINAL callable is
   absent from torch's own override registries
   (``get_overridable_functions`` + ``get_testing_overrides``) — the static
   not-mode-visible superset.
2. Each candidate with a registered probe recipe is CALLED under a counting
   ``TorchFunctionMode`` (with logging paused). A successful tensor-touching
   call with zero callbacks is a belt member; a call that fires the mode is
   excluded (this resolves build-dependent visibility such as ``from_file``
   automatically per version).
3. Candidates without a runnable probe are DISCLOSED in the report, never
   silently classified.

On this torch build the derived set is ``{torch.from_numpy,
torch.frombuffer, torch.Tensor.as_subclass}`` (pinned in
``tests/test_mechanical_belt.py``).
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
import weakref
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from torch.overrides import TorchFunctionMode, get_overridable_functions, get_testing_overrides

from ... import _state

__all__ = [
    "BeltReport",
    "belt_report",
    "restore_belt_references",
    "sweep_stale_belt_references",
]


_SKIP_MODULE_PREFIXES = (
    # Same shallow-scan skip set the historical crawler used, plus torchlens
    # itself: these namespaces are either torch-owned (already decorated at
    # their public slots) or known torch-free.
    "torchlens",
    "torch.",
    "numpy.",
    "pytest",
    "pluggy",
    "setuptools",
)


class _CountingMode(TorchFunctionMode):
    """Count protocol dispatches during a probe call, passing through."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def __torch_function__(
        self,
        func: Any,
        types_: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        self.calls += 1
        return func(*args, **(kwargs or {}))


class _ProbeSubTensor(torch.Tensor):
    """Minimal Tensor subclass for the ``as_subclass`` probe."""


def _from_file_args() -> tuple[tuple[Any, ...], dict[str, Any]]:
    array = np.array([0.25, 0.5], dtype=np.float32)
    fd, path = tempfile.mkstemp()
    os.write(fd, array.tobytes())
    os.close(fd)
    return (path,), {"shared": False, "size": 2, "dtype": torch.float32}


PROBE_RECIPES: dict[tuple[str, str], Callable[[], tuple[tuple[Any, ...], dict[str, Any]]]] = {
    # Tensor-source candidates in the statically not-mode-visible superset.
    # A recipe only synthesizes ARGUMENTS (built BEFORE the counting mode is
    # armed, so a factory call in the recipe cannot fire it); MEMBERSHIP is
    # decided by the measured callback count of the candidate call itself, so
    # a recipe for a mode-visible function (e.g. from_file on builds where it
    # dispatches) is harmless and self-excluding.
    ("torch", "from_numpy"): lambda: ((np.array([0.25, 0.5], dtype=np.float32),), {}),
    ("torch", "frombuffer"): lambda: (
        (bytearray(np.array([0.25, 0.5], dtype=np.float32).tobytes()),),
        {"dtype": torch.float32},
    ),
    ("torch", "from_file"): _from_file_args,
    ("torch.Tensor", "as_subclass"): lambda: (
        (torch.tensor([0.25, 0.5]), _ProbeSubTensor),
        {},
    ),
    ("torch", "manual_seed"): lambda: ((7,), {}),
}


@dataclass(frozen=True)
class BeltReport:
    """Derivation evidence for the protocol-invisible belt."""

    members: tuple[tuple[str, str], ...]
    probed_visible: tuple[tuple[str, str], ...]
    probe_failures: tuple[tuple[str, str], ...]
    unprobed_candidate_count: int


_report: BeltReport | None = None
_member_map: dict[int, Any] | None = None
"""id(original callable) -> decorated wrapper, for the derived belt members."""

_swept_module_ids: dict[int, Callable[[], Any | None]] = {}
"""Module identities already swept this wrapper epoch (weak where possible)."""

_ledger: list[tuple[Callable[[], Any | None], str, Any, Any]] = []
"""(module_ref, attr_name, original, replacement) reversal entries."""


def _touches_tensor(result: Any, func: Callable[..., Any]) -> bool:
    """A belt candidate must produce or consume tensors to matter."""

    if isinstance(result, torch.Tensor):
        return True
    qualname = getattr(func, "__qualname__", "")
    return qualname.startswith(("Tensor.", "TensorBase."))


def _resolve_namespace(namespace_name: str) -> Any | None:
    obj: Any = torch
    for part in namespace_name.replace("torch.", "").split("."):
        if part:
            obj = getattr(obj, part, None)
            if obj is None:
                return None
    return obj


def _derive() -> tuple[BeltReport, dict[int, Any]]:
    """Measure the protocol-invisible wrapped set on the running build."""

    from ...constants import get_orig_torch_funcs

    inventory = get_orig_torch_funcs(include_torchvision=False)
    statically_visible: set[int] = set()
    for functions in get_overridable_functions().values():
        statically_visible.update(id(func) for func in functions)
    statically_visible.update(id(func) for func in get_testing_overrides())

    members: list[tuple[str, str]] = []
    probed_visible: list[tuple[str, str]] = []
    probe_failures: list[tuple[str, str]] = []
    member_map: dict[int, Any] = {}
    unprobed = 0
    seen_original_ids: set[int] = set()

    for namespace_name, func_name in inventory:
        namespace = _resolve_namespace(namespace_name)
        if namespace is None or not hasattr(namespace, func_name):
            continue
        current = getattr(namespace, func_name)
        original = _state._decorated_to_orig.get(id(current), current)
        if id(original) in seen_original_ids:
            continue
        seen_original_ids.add(id(original))
        if id(original) in statically_visible:
            continue
        recipe = PROBE_RECIPES.get((namespace_name, func_name))
        if recipe is None:
            unprobed += 1
            continue
        mode = _CountingMode()
        try:
            with _state.pause_logging():
                args, kwargs = recipe()
                with mode:
                    result = original(*args, **kwargs)
        except Exception:
            probe_failures.append((namespace_name, func_name))
            continue
        if mode.calls:
            probed_visible.append((namespace_name, func_name))
            continue
        if not _touches_tensor(result, original):
            continue
        members.append((namespace_name, func_name))
        wrapper = _state._orig_to_decorated.get(id(original))
        if wrapper is not None:
            member_map[id(original)] = wrapper

    report = BeltReport(
        members=tuple(members),
        probed_visible=tuple(probed_visible),
        probe_failures=tuple(probe_failures),
        unprobed_candidate_count=unprobed,
    )
    return report, member_map


def belt_report() -> BeltReport | None:
    """Return the derivation report, deriving on first use after wrapping."""

    global _report, _member_map
    if not _state._is_decorated:
        return _report
    if _report is None:
        _report, _member_map = _derive()
    return _report


def _weak_module_ref(module: types.ModuleType) -> Callable[[], Any | None]:
    try:
        return weakref.ref(module)
    except TypeError:
        return lambda: module


def sweep_stale_belt_references() -> int:
    """Patch stale module-level references to belt members, with a ledger.

    Scans each live module identity ONCE per wrapper epoch (new imports are
    picked up on the next sweep). Only module ``__dict__`` slots are
    patched — the measured reachable holder class for the protocol-invisible
    functions — and every mutation is recorded for conditional reversal at
    ``unwrap_torch()``.

    Returns
    -------
    int
        Number of slots patched by this sweep.
    """

    report = belt_report()
    if report is None or _member_map is None or not _member_map:
        return 0
    patched = 0
    for mod_key, module in list(sys.modules.items()):
        if not isinstance(module, types.ModuleType):
            continue
        if id(module) in _swept_module_ids:
            continue
        _swept_module_ids[id(module)] = _weak_module_ref(module)
        if mod_key.startswith(_SKIP_MODULE_PREFIXES) or ".dist-info" in mod_key:
            continue
        try:
            module_dict = vars(module)
        except TypeError:
            continue
        for attr_name, attr_val in list(module_dict.items()):
            replacement = _member_map.get(id(attr_val))
            if replacement is None:
                continue
            try:
                if module_dict.get(attr_name) is not attr_val:
                    continue
                module_dict[attr_name] = replacement
            except (KeyError, TypeError):
                continue
            _ledger.append((_weak_module_ref(module), attr_name, attr_val, replacement))
            patched += 1
    return patched


def restore_belt_references() -> None:
    """Conditionally reverse belt mutations and reset the epoch sweep memo.

    A slot is restored only when it still holds the exact wrapper the belt
    installed; user reassignments made after the sweep are preserved.
    """

    for module_ref, attr_name, original, replacement in reversed(_ledger):
        module = module_ref()
        if module is None:
            continue
        try:
            module_dict = vars(module)
            if module_dict.get(attr_name) is replacement:
                module_dict[attr_name] = original
        except (KeyError, TypeError):
            continue
    _ledger.clear()
    _swept_module_ids.clear()
