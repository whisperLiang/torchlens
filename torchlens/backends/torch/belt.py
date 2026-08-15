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
3. Candidates without a runnable probe are DISCLOSED in the report — by name
   (``unprobed_candidates``) and count — never silently classified. Probing
   itself is recipe-gated: only ``PROBE_RECIPES`` entries can be measured, so
   the mechanical derivation resolves visibility for the RECIPE-COVERED
   tensor-source family and honestly discloses the rest.

On this torch build the derived set is ``{torch.from_numpy, torch.from_dlpack,
torch.frombuffer, torch.Tensor.as_subclass, torch.Tensor._make_subclass}``
(pinned in ``tests/test_mechanical_belt.py``).
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

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
    """Build args/kwargs for the ``torch.from_file`` probe, backed by a temp file.

    The caller removes the temporary path after the probe call. ``shared=False``
    means the returned tensor does not require the pathname to remain present.
    """

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
    # The modern DLPack interop boundary is protocol-invisible for the same reason
    # ``from_numpy`` is: it builds a tensor from a FOREIGN buffer, so no
    # ``__torch_function__`` mode ever sees the call. It was not even a belt CANDIDATE
    # before, because candidacy is derived from ORIG_TORCH_FUNCS and the function was
    # absent from both of torch's override registries (see constants.py).
    ("torch", "from_dlpack"): lambda: ((np.array([0.25, 0.5], dtype=np.float32),), {}),
    ("torch", "frombuffer"): lambda: (
        (bytearray(np.array([0.25, 0.5], dtype=np.float32).tobytes()),),
        {"dtype": torch.float32},
    ),
    ("torch", "from_file"): _from_file_args,
    ("torch.Tensor", "as_subclass"): lambda: (
        (torch.tensor([0.25, 0.5]), _ProbeSubTensor),
        {},
    ),
    # Exact sibling of ``as_subclass``: builds a subclass VIEW from raw
    # storage below the override protocol, so a stale pre-wrap
    # ``_make_subclass`` reference loses the op with zero signal. Without a
    # recipe the pair sat disclosed-but-unprobed forever (b6-fable carried).
    ("torch.Tensor", "_make_subclass"): lambda: (
        (_ProbeSubTensor, torch.tensor([0.25, 0.5])),
        {},
    ),
    ("torch", "manual_seed"): lambda: ((7,), {}),
}


@contextmanager
def _probe_rng_bracket() -> Iterator[None]:
    """Snapshot/restore the global torch RNG around one probe evaluation.

    The probe framework executes candidate ORIGINALS with synthesized
    arguments at first wrap, inside the user's first capture. The candidate
    inventory is build-derived, so a state-mutating factory row entering it
    (``manual_seed`` already has a recipe that would call ``manual_seed(7)``;
    it is merely dead on current builds) would silently clobber the user's
    global seed. The bracket makes probe evaluation RNG-neutral by
    construction (b8-fable R56 latent-reseed hardening).
    """

    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@dataclass(frozen=True)
class BeltReport:
    """Derivation evidence for the protocol-invisible belt."""

    members: tuple[tuple[str, str], ...]
    probed_visible: tuple[tuple[str, str], ...]
    probe_failures: tuple[tuple[str, str], ...]
    unprobed_candidate_count: int
    unprobed_candidates: tuple[tuple[str, str], ...] = ()


_report: BeltReport | None = None
_member_map: dict[int, Any] | None = None
"""id(original callable) -> decorated wrapper, for the derived belt members."""

_swept_module_ids: dict[int, Callable[[], Any | None]] = {}
"""Module identities already swept this wrapper epoch (weak where possible)."""

_swept_sys_modules_size = -1
"""``len(sys.modules)`` at the last complete belt sweep."""

_swept_modules_dirty = False
"""Whether a previously swept weak module reference has died."""

_ledger: list[tuple[Callable[[], Any | None], str, Any, Any]] = []
"""(module_ref, attr_name, original, replacement) reversal entries."""


def _touches_tensor(result: Any, func: Callable[..., Any]) -> bool:
    """A belt candidate must produce or consume tensors to matter."""

    if isinstance(result, torch.Tensor):
        return True
    qualname = getattr(func, "__qualname__", "")
    return qualname.startswith(("Tensor.", "TensorBase."))


def _resolve_namespace(namespace_name: str) -> Any | None:
    """Resolve a dotted ``torch.*`` namespace name, or ``None`` if any part is absent."""

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
    unprobed_candidates: list[tuple[str, str]] = []
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
            unprobed_candidates.append((namespace_name, func_name))
            continue
        mode = _CountingMode()
        cleanup_path: str | None = None
        try:
            with _state.pause_logging(), _probe_rng_bracket():
                args, kwargs = recipe()
                if (namespace_name, func_name) == ("torch", "from_file"):
                    cleanup_path = str(args[0])
                with mode:
                    result = original(*args, **kwargs)
        except Exception:
            probe_failures.append((namespace_name, func_name))
            continue
        finally:
            if cleanup_path is not None:
                try:
                    os.unlink(cleanup_path)
                except FileNotFoundError:
                    pass
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
        unprobed_candidate_count=len(unprobed_candidates),
        unprobed_candidates=tuple(unprobed_candidates),
    )
    return report, member_map


def _member_map_is_current() -> bool:
    """True while every cached wrapper is still the live one for its original.

    The cached derivation binds wrapper OBJECT identities. A fresh full
    decoration pass (first-wrap retry after a partial failure) mints a new
    wrapper generation, at which point sweeping the cached objects would
    patch dead wrappers into user modules.
    """

    if _member_map is None:
        return False
    return all(
        _state._orig_to_decorated.get(original_id) is wrapper
        for original_id, wrapper in _member_map.items()
    )


def belt_report() -> BeltReport | None:
    """Return the derivation report, deriving on first use after wrapping.

    The derivation is re-validated against the live wrapper registries: if
    the wrapper generation changed underneath the cache, mutations made with
    the dead generation are reversed and the belt re-derives.
    """

    global _report, _member_map
    if not _state._is_decorated:
        return _report
    if _report is not None and not _member_map_is_current():
        restore_belt_references()
        _report = None
        _member_map = None
    if _report is None:
        _report, _member_map = _derive()
    return _report


def _weak_module_ref(module: types.ModuleType) -> Callable[[], Any | None]:
    """Weak reference to ``module``, degrading to a strong closure when unsupported."""

    try:
        return weakref.ref(module)
    except TypeError:
        return lambda: module


def _weak_swept_module_ref(
    module: types.ModuleType,
) -> Callable[[], Any | None]:
    """Return a module reference that invalidates the sweep watermark on collection."""

    def _mark_sweep_dirty(_reference: weakref.ReferenceType[types.ModuleType]) -> None:
        """Mark the module inventory dirty after a swept module is collected."""

        global _swept_modules_dirty
        _swept_modules_dirty = True

    try:
        return weakref.ref(module, _mark_sweep_dirty)
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

    global _swept_modules_dirty, _swept_sys_modules_size
    report = belt_report()
    if report is None or _member_map is None or not _member_map:
        return 0
    if len(sys.modules) == _swept_sys_modules_size and not _swept_modules_dirty:
        return 0
    patched = 0
    for mod_key, module in list(sys.modules.items()):
        if not isinstance(module, types.ModuleType):
            continue
        previous_ref = _swept_module_ids.get(id(module))
        if previous_ref is not None and previous_ref() is module:
            continue
        _swept_module_ids[id(module)] = _weak_swept_module_ref(module)
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
    _swept_sys_modules_size = len(sys.modules)
    _swept_modules_dirty = False
    return patched


def restore_belt_references() -> None:
    """Conditionally reverse belt mutations and reset the epoch sweep memo.

    A slot is restored only when it still holds the exact wrapper the belt
    installed; user reassignments made after the sweep are preserved.
    """

    global _swept_modules_dirty, _swept_sys_modules_size
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
    _swept_sys_modules_size = -1
    _swept_modules_dirty = False
