"""Internal model state, cache, and input helpers for public trace capture."""

from __future__ import annotations

import collections.abc
import copy
import dataclasses
import hashlib
import json
import os
import types
import warnings
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

import torch
from torch import nn

from . import _state
from .data_classes.trace import Trace
from .utils._torch_compat import (
    get_dynamo_optimized_module_type,
    get_fsdp_wrapper_type,
    is_dynamo_compiled_callable,
)


def _clone_state_dict_with_metadata(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    """Clone a module ``state_dict`` while preserving PyTorch metadata.

    Parameters
    ----------
    model:
        Module whose state should be cloned.

    Returns
    -------
    OrderedDict[str, torch.Tensor]
        Detached tensor clones plus any private ``_metadata`` needed by module
        implementations such as torchvision MNASNet during ``load_state_dict``.
    """

    original_state = model.state_dict()
    cloned_state = OrderedDict(
        (name, tensor.detach().clone()) for name, tensor in original_state.items()
    )
    if hasattr(original_state, "_metadata"):
        cloned_state._metadata = copy.deepcopy(original_state._metadata)  # type: ignore[attr-defined]
    return cloned_state


_VALIDATION_DEEPCOPY_WARNING_TYPES: set[type[nn.Module]] = set()
_PLAIN_ATTR_IGNORED_NAMES = frozenset({"_parameters", "_buffers", "_modules"})
_PLAIN_ATTR_MAX_CONTAINER_ITEMS = 128
_PLAIN_ATTR_MAX_TENSOR_NUMEL = 4096
_COMPILED_MODEL_UNWRAP_WARNED = False


@dataclasses.dataclass(frozen=True)
class _PlainAttrIdentitySnapshot:
    """Identity snapshot for opaque but identity-stable plain attributes.

    Parameters
    ----------
    value:
        Original object to restore if the plain attribute is reassigned.
    value_type_name:
        Human-readable type name for diagnostics.
    """

    value: Any
    value_type_name: str


@dataclasses.dataclass(frozen=True)
class _PlainAttrManagedTensorSnapshot:
    """Snapshot marker for PyTorch-managed derived tensor attributes.

    Parameters
    ----------
    module:
        Owning module that exposes the derived tensor attribute.
    name:
        Attribute name on ``module``.
    shape:
        Tensor shape at snapshot time.
    dtype:
        Tensor dtype at snapshot time.
    device:
        Tensor device at snapshot time.
    manager:
        Human-readable manager kind for diagnostics.
    """

    module: nn.Module
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    manager: str


@dataclasses.dataclass(frozen=True)
class _PlainAttrE3nnTupleSnapshot:
    """Value snapshot for e3nn's immutable tuple-backed irreps specs.

    Parameters
    ----------
    value:
        Independent copy used to restore attribute replacement.
    items:
        Snapshots of the raw tuple payload, obtained without calling the
        deliberately unsupported ``Irreps.__len__`` implementation.
    attributes:
        Snapshot of any instance attributes attached to the spec.
    """

    value: Any
    items: tuple[Any, ...]
    attributes: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class _CompiledSubmoduleSwap:
    """Snapshot of one temporarily unwrapped compiled submodule slot.

    Parameters
    ----------
    parent:
        Parent module that owns the child slot.
    name:
        Child name inside ``parent._modules``.
    compiled_module:
        Original compiled wrapper to restore.
    """

    parent: nn.Module
    name: str
    compiled_module: nn.Module


@dataclasses.dataclass(frozen=True)
class _CompiledCallableSwap:
    """Snapshot of one temporarily bypassed compiled callable attribute.

    Parameters
    ----------
    module:
        Module that owns the plain attribute.
    name:
        Attribute name in the module instance dictionary.
    compiled_callable:
        Original Dynamo-compiled callable restored after capture.
    bypass_callable:
        Temporary wrapper that invokes it with TorchLens logging paused.
    """

    module: nn.Module
    name: str
    compiled_callable: Callable[..., Any]
    bypass_callable: Callable[..., Any]


def reset_compiled_model_unwrap_warning_state() -> None:
    """Reset the process-local compiled-model unwrap warning flag.

    Returns
    -------
    None
        The next compiled-model unwrap emits the user-facing note again.
    """

    global _COMPILED_MODEL_UNWRAP_WARNED

    _COMPILED_MODEL_UNWRAP_WARNED = False


def _warn_compiled_model_unwrapped_once() -> None:
    """Emit the compiled-model eager-source note at most once per process.

    Returns
    -------
    None
        Emits a ``UserWarning`` only on the first call in the process.
    """

    global _COMPILED_MODEL_UNWRAP_WARNED

    if _COMPILED_MODEL_UNWRAP_WARNED:
        return
    _COMPILED_MODEL_UNWRAP_WARNED = True
    warnings.warn(
        "TorchLens: compiled model detected; tracing the eager source module it wraps "
        "(torch.compile semantics such as fusion are not traced).",
        UserWarning,
        stacklevel=3,
    )


def _compiled_model_orig_module(model: nn.Module) -> nn.Module | None:
    """Return the eager source module for a Dynamo compiled wrapper, if present.

    Parameters
    ----------
    model:
        Module to inspect.

    Returns
    -------
    nn.Module | None
        ``model._orig_mod`` when ``model`` is an OptimizedModule and the original
        object is an ``nn.Module``; otherwise ``None``.
    """

    optimized_module_type = get_dynamo_optimized_module_type()
    if optimized_module_type is None or not isinstance(model, optimized_module_type):
        return None
    orig_mod = getattr(model, "_orig_mod", None)
    if isinstance(orig_mod, nn.Module):
        return orig_mod
    return None


def unwrap_compiled_model(model: nn.Module) -> nn.Module:
    """Return the eager source module for a top-level compiled wrapper.

    Parameters
    ----------
    model:
        User-supplied model.

    Returns
    -------
    nn.Module
        ``model._orig_mod`` when ``model`` is a Dynamo OptimizedModule, otherwise
        ``model`` unchanged.
    """

    orig_mod = _compiled_model_orig_module(model)
    if orig_mod is None:
        return model
    _warn_compiled_model_unwrapped_once()
    return orig_mod


def compiled_plain_callable_sites(model: nn.Module) -> tuple[str, ...]:
    """Return plain module attributes carrying Dynamo-compiled callables.

    Parameters
    ----------
    model:
        Model whose module instance dictionaries are inspected.

    Returns
    -------
    tuple[str, ...]
        Stable module-attribute paths for compiled non-module callables.

    Notes
    -----
    The direct ``vars`` walk never executes descriptors. This inventory is what
    makes cache-hit handling independent of the transient ``is_compiling()`` flag:
    a hot compiled callable can bypass Python torch wrappers entirely.
    """

    sites: list[str] = []
    try:
        modules = tuple(model.named_modules())
    except Exception:
        modules = (("", model),)
    for module_name, module in modules:
        try:
            attributes = vars(module)
        except (TypeError, AttributeError):
            continue
        for attr_name, value in attributes.items():
            if attr_name in _PLAIN_ATTR_IGNORED_NAMES or isinstance(value, nn.Module):
                continue
            if is_dynamo_compiled_callable(value):
                prefix = module_name or "<root>"
                sites.append(f"{prefix}.{attr_name}")
    return tuple(dict.fromkeys(sites))


def _make_compiled_callable_bypass(
    compiled_callable: Callable[..., Any],
) -> Callable[..., Any]:
    """Return a wrapper that executes one compiled callable outside logging.

    Parameters
    ----------
    compiled_callable:
        Dynamo-compiled callable to invoke.

    Returns
    -------
    Callable[..., Any]
        Plain wrapper preserving compiled execution while pausing TorchLens logging.
    """

    def bypass(*args: Any, **kwargs: Any) -> Any:
        """Invoke the compiled boundary without exposing FakeTensor internals.

        Parameters
        ----------
        *args:
            Positional callable arguments.
        **kwargs:
            Keyword callable arguments.

        Returns
        -------
        Any
            Compiled callable result.
        """

        with _state.pause_logging():
            return compiled_callable(*args, **kwargs)

    return bypass


@contextmanager
def bypass_compiled_plain_callables(model: nn.Module) -> Iterator[None]:
    """Temporarily route direct compiled attributes through a logging pause.

    Parameters
    ----------
    model:
        Root model whose direct instance attributes are inspected.

    Yields
    ------
    None
        Control while compiled callable attributes have safe boundary wrappers.
    """

    swaps: list[_CompiledCallableSwap] = []
    try:
        modules = tuple(model.named_modules())
    except Exception:
        modules = (("", model),)
    try:
        for _module_name, module in modules:
            try:
                attributes = tuple(vars(module).items())
            except (TypeError, AttributeError):
                continue
            for attr_name, value in attributes:
                if attr_name in _PLAIN_ATTR_IGNORED_NAMES or isinstance(value, nn.Module):
                    continue
                if not callable(value) or not is_dynamo_compiled_callable(value):
                    continue
                bypass = _make_compiled_callable_bypass(value)
                setattr(module, attr_name, bypass)
                swaps.append(
                    _CompiledCallableSwap(
                        module=module,
                        name=attr_name,
                        compiled_callable=value,
                        bypass_callable=bypass,
                    )
                )
    except BaseException:
        for swap in reversed(swaps):
            setattr(swap.module, swap.name, swap.compiled_callable)
        raise

    try:
        yield
    finally:
        for swap in reversed(swaps):
            if vars(swap.module).get(swap.name) is swap.bypass_callable:
                setattr(swap.module, swap.name, swap.compiled_callable)


@contextmanager
def prepare_compiled_capture(model: nn.Module) -> Iterator[tuple[str, ...]]:
    """Prepare compiled submodules and plain callables for honest eager capture.

    Parameters
    ----------
    model:
        Root model to prepare temporarily.

    Yields
    ------
    tuple[str, ...]
        Direct compiled-callable sites whose interiors remain unlogged.
    """

    with unwrap_compiled_submodules(model):
        sites = compiled_plain_callable_sites(model)
        with bypass_compiled_plain_callables(model):
            yield sites


@contextmanager
def unwrap_compiled_submodules(model: nn.Module) -> Iterator[None]:
    """Temporarily replace compiled child slots with their eager source modules.

    Parameters
    ----------
    model:
        Root model whose descendants should execute through eager modules during
        TorchLens capture.

    Yields
    ------
    None
        Control while compiled child slots are unwrapped.
    """

    swaps: list[_CompiledSubmoduleSwap] = []
    try:
        traversal_queue: list[nn.Module] = [model]
        seen_module_ids: set[int] = set()
        while traversal_queue:
            parent = traversal_queue.pop()
            parent_id = id(parent)
            if parent_id in seen_module_ids:
                continue
            seen_module_ids.add(parent_id)
            for child_name, child_module in list(parent._modules.items()):
                if child_module is None:
                    continue
                orig_mod = _compiled_model_orig_module(child_module)
                if orig_mod is None:
                    traversal_queue.append(child_module)
                    continue
                swaps.append(
                    _CompiledSubmoduleSwap(
                        parent=parent,
                        name=child_name,
                        compiled_module=child_module,
                    )
                )
                parent._modules[child_name] = orig_mod
                traversal_queue.append(orig_mod)
    except BaseException:
        for swap in reversed(swaps):
            swap.parent._modules[swap.name] = swap.compiled_module
        raise

    try:
        if swaps:
            _warn_compiled_model_unwrapped_once()
        yield
    finally:
        for swap in reversed(swaps):
            swap.parent._modules[swap.name] = swap.compiled_module


def _is_identity_stable_plain_attr(value: Any) -> bool:
    """Return whether a plain attr should be tracked by object identity.

    Parameters
    ----------
    value:
        Attribute value to classify.

    Returns
    -------
    bool
        True for callable/type/module objects whose mutation signal is
        reassignment rather than value drift.
    """

    return isinstance(
        value,
        (
            types.FunctionType,
            types.MethodType,
            types.BuiltinFunctionType,
            types.BuiltinMethodType,
            type,
            nn.Module,
        ),
    ) or callable(value)


def _legacy_parametrization_manager(module: nn.Module, name: str) -> str | None:
    """Return the legacy PyTorch parametrization hook managing an attribute.

    Parameters
    ----------
    module:
        Module that owns the candidate plain tensor attribute.
    name:
        Attribute name to classify.

    Returns
    -------
    str | None
        Manager kind when the attribute is a computed view backed by registered
        module state, otherwise ``None``.
    """

    for hook in getattr(module, "_forward_pre_hooks", {}).values():
        if getattr(hook, "name", None) != name:
            continue
        hook_type = type(hook)
        hook_key = f"{hook_type.__module__}.{hook_type.__name__}"
        if (
            hook_key == "torch.nn.utils.weight_norm.WeightNorm"
            and f"{name}_g" in module._parameters
            and f"{name}_v" in module._parameters
        ):
            return "legacy_weight_norm"
        if (
            hook_key == "torch.nn.utils.spectral_norm.SpectralNorm"
            and f"{name}_orig" in module._parameters
            and f"{name}_u" in module._buffers
            and f"{name}_v" in module._buffers
        ):
            return "legacy_spectral_norm"
    return None


def _parametrize_manager(module: nn.Module, name: str) -> str | None:
    """Return whether PyTorch's parametrization API manages an attribute.

    Parameters
    ----------
    module:
        Module that owns the candidate plain tensor attribute.
    name:
        Attribute name to classify.

    Returns
    -------
    str | None
        Manager kind when the attribute is parametrized, otherwise ``None``.
    """

    try:
        from torch.nn.utils import parametrize
    except ImportError:
        return None
    try:
        if parametrize.is_parametrized(module, name):
            return "parametrize"
    except (AttributeError, ValueError, TypeError):
        return None
    return None


def _managed_plain_tensor_attr_snapshot(
    module: nn.Module,
    name: str,
    value: Any,
) -> _PlainAttrManagedTensorSnapshot | None:
    """Return a marker for a legitimate derived plain tensor attribute.

    Parameters
    ----------
    module:
        Module that owns the candidate attribute.
    name:
        Attribute name to classify.
    value:
        Current attribute value.

    Returns
    -------
    _PlainAttrManagedTensorSnapshot | None
        Snapshot marker when PyTorch registered state manages this plain tensor
        attribute, otherwise ``None``.
    """

    if not isinstance(value, torch.Tensor):
        return None
    manager = _parametrize_manager(module, name) or _legacy_parametrization_manager(module, name)
    if manager is None:
        return None
    return _PlainAttrManagedTensorSnapshot(
        module=module,
        name=name,
        shape=tuple(value.shape),
        dtype=value.dtype,
        device=value.device,
        manager=manager,
    )


def _snapshot_module_plain_attr_value(module: nn.Module, name: str, attr_path: str) -> Any:
    """Return a snapshot for a named plain attribute on a module.

    Parameters
    ----------
    module:
        Module that owns the plain attribute.
    name:
        Attribute name.
    attr_path:
        Human-readable module/attribute path for diagnostics.

    Returns
    -------
    Any
        Snapshot suitable for later comparison and restoration.
    """

    value = getattr(module, name)
    managed_snapshot = _managed_plain_tensor_attr_snapshot(module, name, value)
    if managed_snapshot is not None:
        return managed_snapshot
    return _snapshot_plain_attr_value(value, attr_path)


def _snapshot_plain_attr_value(value: Any, attr_path: str) -> Any:
    """Return a value snapshot for a plain module-tree attribute.

    Parameters
    ----------
    value:
        Attribute value to snapshot.
    attr_path:
        Human-readable module/attribute path for diagnostics.

    Returns
    -------
    Any
        Detached value snapshot that can later be compared and restored.

    Raises
    ------
    RuntimeError
        If the value is not small and value-comparable enough for the fallback
        validation restore. External objects remain unsupported in this path.
    """

    if isinstance(
        value,
        (
            _PlainAttrIdentitySnapshot,
            _PlainAttrManagedTensorSnapshot,
            _PlainAttrE3nnTupleSnapshot,
        ),
    ):
        return value
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return value
    if _is_identity_stable_plain_attr(value):
        return _PlainAttrIdentitySnapshot(value=value, value_type_name=type(value).__name__)
    if isinstance(value, torch.Tensor):
        if value.numel() > _PLAIN_ATTR_MAX_TENSOR_NUMEL:
            raise RuntimeError(
                "TorchLens validation deepcopy fallback cannot snapshot plain "
                f"attribute '{attr_path}' because its tensor value has {value.numel()} "
                "elements. Non-registered large mutable state is unsupported."
            )
        return value.detach().clone()
    if isinstance(value, list):
        if len(value) > _PLAIN_ATTR_MAX_CONTAINER_ITEMS:
            raise RuntimeError(
                "TorchLens validation deepcopy fallback cannot snapshot plain "
                f"attribute '{attr_path}' because its list has {len(value)} items."
            )
        return [
            _snapshot_plain_attr_value(item, f"{attr_path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        value_type = type(value)
        is_e3nn_irreps_tuple = (
            value_type.__module__ == "e3nn.o3._irreps"
            and value_type.__qualname__ in {"Irreps", "_MulIr", "Irrep"}
        )
        if is_e3nn_irreps_tuple:
            tuple_len = tuple.__len__(value)
            if tuple_len > _PLAIN_ATTR_MAX_CONTAINER_ITEMS:
                raise RuntimeError(
                    "TorchLens validation deepcopy fallback cannot snapshot plain "
                    f"attribute '{attr_path}' because its e3nn Irreps has {tuple_len} items."
                )
            items = tuple(
                _snapshot_plain_attr_value(tuple.__getitem__(value, index), f"{attr_path}[{index}]")
                for index in range(tuple_len)
            )
            attributes = _snapshot_plain_attr_value(dict(vars(value)), f"{attr_path}.__dict__")
            return _PlainAttrE3nnTupleSnapshot(
                value=copy.deepcopy(value),
                items=items,
                attributes=cast(dict[str, Any], attributes),
            )
        try:
            tuple_len = len(value)
        except Exception as exc:
            raise RuntimeError(
                "TorchLens validation deepcopy fallback cannot snapshot plain "
                f"attribute '{attr_path}' because its tuple length is unavailable."
            ) from exc
        if tuple_len > _PLAIN_ATTR_MAX_CONTAINER_ITEMS:
            raise RuntimeError(
                "TorchLens validation deepcopy fallback cannot snapshot plain "
                f"attribute '{attr_path}' because its tuple has {tuple_len} items."
            )
        return tuple(
            _snapshot_plain_attr_value(item, f"{attr_path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, dict):
        if len(value) > _PLAIN_ATTR_MAX_CONTAINER_ITEMS:
            raise RuntimeError(
                "TorchLens validation deepcopy fallback cannot snapshot plain "
                f"attribute '{attr_path}' because its dict has {len(value)} items."
            )
        snapshot = {}
        for key, item in value.items():
            try:
                key_snapshot = _snapshot_plain_attr_value(key, f"{attr_path}.<key>")
            except RuntimeError as exc:
                raise RuntimeError(
                    "TorchLens validation deepcopy fallback cannot snapshot plain "
                    f"attribute '{attr_path}' because one of its dict keys is unsupported."
                ) from exc
            try:
                hash(key_snapshot)
            except TypeError as exc:
                raise RuntimeError(
                    "TorchLens validation deepcopy fallback cannot snapshot plain "
                    f"attribute '{attr_path}' because one of its dict keys is unhashable "
                    "after snapshotting."
                ) from exc
            snapshot[key_snapshot] = _snapshot_plain_attr_value(item, f"{attr_path}[{key!r}]")
        return snapshot
    if isinstance(value, (set, frozenset)):
        if len(value) > _PLAIN_ATTR_MAX_CONTAINER_ITEMS:
            raise RuntimeError(
                "TorchLens validation deepcopy fallback cannot snapshot plain "
                f"attribute '{attr_path}' because its set has {len(value)} items."
            )
        snapshot_items = [_snapshot_plain_attr_value(item, f"{attr_path}.<item>") for item in value]
        try:
            return type(value)(snapshot_items)
        except TypeError as exc:
            raise RuntimeError(
                "TorchLens validation deepcopy fallback cannot snapshot plain "
                f"attribute '{attr_path}' because its set items are not hashable."
            ) from exc
    raise RuntimeError(
        "TorchLens validation deepcopy fallback cannot snapshot plain "
        f"attribute '{attr_path}' of type {type(value).__name__}. Non-registered "
        "external objects and other opaque mutable state are unsupported."
    )


def _plain_attr_values_equal(left: Any, right: Any, attr_path: str) -> bool:
    """Return whether two snapshotted plain-attribute values are equal by value.

    Parameters
    ----------
    left:
        First value.
    right:
        Second value.
    attr_path:
        Human-readable module/attribute path for diagnostics.

    Returns
    -------
    bool
        True when values match.

    Raises
    ------
    RuntimeError
        If the values cannot be compared without ambiguity.
    """

    if isinstance(left, _PlainAttrIdentitySnapshot) or isinstance(
        right, _PlainAttrIdentitySnapshot
    ):
        if not isinstance(left, _PlainAttrIdentitySnapshot) or not isinstance(
            right, _PlainAttrIdentitySnapshot
        ):
            return False
        return left.value is right.value
    if isinstance(left, _PlainAttrManagedTensorSnapshot) or isinstance(
        right, _PlainAttrManagedTensorSnapshot
    ):
        if not isinstance(left, _PlainAttrManagedTensorSnapshot) or not isinstance(
            right, _PlainAttrManagedTensorSnapshot
        ):
            return False
        return (
            left.module is right.module
            and left.name == right.name
            and left.shape == right.shape
            and left.dtype == right.dtype
            and left.device == right.device
            and left.manager == right.manager
        )
    if isinstance(left, _PlainAttrE3nnTupleSnapshot) or isinstance(
        right, _PlainAttrE3nnTupleSnapshot
    ):
        if not isinstance(left, _PlainAttrE3nnTupleSnapshot) or not isinstance(
            right, _PlainAttrE3nnTupleSnapshot
        ):
            return False
        return _plain_attr_values_equal(left.items, right.items, f"{attr_path}.<items>") and (
            _plain_attr_values_equal(
                left.attributes,
                right.attributes,
                f"{attr_path}.__dict__",
            )
        )
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return False
        return bool(torch.equal(left, right))
    if isinstance(left, list) or isinstance(right, list):
        if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
            return False
        return all(
            _plain_attr_values_equal(left_item, right_item, f"{attr_path}[{index}]")
            for index, (left_item, right_item) in enumerate(zip(left, right))
        )
    if isinstance(left, tuple) or isinstance(right, tuple):
        if not isinstance(left, tuple) or not isinstance(right, tuple) or len(left) != len(right):
            return False
        return all(
            _plain_attr_values_equal(left_item, right_item, f"{attr_path}[{index}]")
            for index, (left_item, right_item) in enumerate(zip(left, right))
        )
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict) or left.keys() != right.keys():
            return False
        return all(
            _plain_attr_values_equal(left[key], right[key], f"{attr_path}[{key!r}]") for key in left
        )
    if isinstance(left, (set, frozenset)) or isinstance(right, (set, frozenset)):
        try:
            return bool(left == right)
        except Exception as exc:
            raise RuntimeError(
                "TorchLens validation deepcopy fallback cannot compare plain "
                f"attribute '{attr_path}' by value."
            ) from exc
    try:
        result = left == right
    except Exception as exc:
        raise RuntimeError(
            "TorchLens validation deepcopy fallback cannot compare plain "
            f"attribute '{attr_path}' by value."
        ) from exc
    if isinstance(result, bool):
        return result
    raise RuntimeError(
        "TorchLens validation deepcopy fallback cannot compare plain "
        f"attribute '{attr_path}' because equality returned {type(result).__name__}."
    )


def _plain_attr_restore_value(snapshot: Any) -> Any:
    """Return the assignable value represented by a plain-attribute snapshot.

    Parameters
    ----------
    snapshot:
        Snapshot produced by :func:`_snapshot_plain_attr_value`.

    Returns
    -------
    Any
        Value to assign back to the module attribute.
    """

    if isinstance(snapshot, _PlainAttrIdentitySnapshot):
        return snapshot.value
    if isinstance(snapshot, _PlainAttrManagedTensorSnapshot):
        return getattr(snapshot.module, snapshot.name)
    if isinstance(snapshot, _PlainAttrE3nnTupleSnapshot):
        return copy.deepcopy(snapshot.value)
    return _snapshot_plain_attr_value(snapshot, "<snapshot>")


def _module_plain_attr_names(module: nn.Module) -> set[str]:
    """Return plain attribute names to snapshot for a module.

    Parameters
    ----------
    module:
        Module whose non-registered attributes should be inspected.

    Returns
    -------
    set[str]
        Attribute names excluding registered parameter, buffer, and child
        module storage.
    """

    return set(module.__dict__) - _PLAIN_ATTR_IGNORED_NAMES


class _ModuleTreePlainAttrSnapshot:
    """Snapshot of small value-comparable plain attributes in a module tree."""

    def __init__(self, model: nn.Module) -> None:
        """Capture a model's plain module-tree attributes.

        Parameters
        ----------
        model:
            Model whose ``modules()`` tree should be snapshotted.
        """

        self._entries: list[tuple[nn.Module, str, str, Any]] = []
        self._module_attr_names: dict[int, tuple[nn.Module, set[str], str]] = {}
        self._unsupported_attr_paths: list[str] = []
        module_counts: dict[str, int] = {}
        for module in model.modules():
            module_type = type(module).__name__
            module_index = module_counts.get(module_type, 0)
            module_counts[module_type] = module_index + 1
            module_path = f"{module_type}[{module_index}]"
            attr_names = _module_plain_attr_names(module)
            self._module_attr_names[id(module)] = (module, attr_names, module_path)
            for name in sorted(attr_names):
                attr_path = f"{module_path}.{name}"
                try:
                    snapshot = _snapshot_module_plain_attr_value(module, name, attr_path)
                except RuntimeError as exc:
                    self._unsupported_attr_paths.append(attr_path)
                    warnings.warn(
                        "TorchLens validation deepcopy fallback could not snapshot plain "
                        f"attribute '{attr_path}'; skipping restoration for this attribute "
                        f"only ({exc}).",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    continue
                self._entries.append(
                    (
                        module,
                        name,
                        attr_path,
                        snapshot,
                    )
                )

    @property
    def is_complete(self) -> bool:
        """Return whether every discovered plain attribute was snapshotted.

        Returns
        -------
        bool
            True only when validation can restore every plain attribute.
        """

        return not self._unsupported_attr_paths

    @property
    def unsupported_attr_paths(self) -> tuple[str, ...]:
        """Return paths whose values could not be snapshotted.

        Returns
        -------
        tuple[str, ...]
            Stable-order unsupported attribute paths.
        """

        return tuple(self._unsupported_attr_paths)

    def restore_changed_attrs(self) -> None:
        """Restore attributes whose values changed since the snapshot.

        Raises
        ------
        RuntimeError
            If any attribute cannot be compared, assigned, deleted, or verified
            after restoration.
        """

        for module, original_names, module_path in self._module_attr_names.values():
            added_names = _module_plain_attr_names(module) - original_names
            for name in sorted(added_names):
                try:
                    delattr(module, name)
                except Exception as exc:
                    raise RuntimeError(
                        "TorchLens validation deepcopy fallback could not remove "
                        f"new plain attribute '{module_path}.{name}' before the logged run."
                    ) from exc
        for module, name, attr_path, snapshot in self._entries:
            try:
                current_snapshot = _snapshot_module_plain_attr_value(module, name, attr_path)
            except AttributeError:
                current_snapshot = None
                changed = True
            else:
                changed = not _plain_attr_values_equal(current_snapshot, snapshot, attr_path)
            if not changed:
                continue
            restore_value = _plain_attr_restore_value(snapshot)
            try:
                setattr(module, name, restore_value)
            except Exception as exc:
                raise RuntimeError(
                    "TorchLens validation deepcopy fallback could not restore plain "
                    f"attribute '{attr_path}' before the logged run."
                ) from exc
            restored_snapshot = _snapshot_plain_attr_value(getattr(module, name), attr_path)
            if not _plain_attr_values_equal(restored_snapshot, snapshot, attr_path):
                raise RuntimeError(
                    "TorchLens validation deepcopy fallback restored plain "
                    f"attribute '{attr_path}', but verification by value failed."
                )


def _model_for_ground_truth_validation(
    model: nn.Module,
) -> tuple[nn.Module, _ModuleTreePlainAttrSnapshot | None]:
    """Return an isolated model for the validation ground-truth run.

    Parameters
    ----------
    model:
        Original model that will later be traced by TorchLens.

    Returns
    -------
    tuple[nn.Module, _ModuleTreePlainAttrSnapshot | None]
        A deep copy and no plain-attribute snapshot when possible; otherwise
        the original model plus a snapshot used to restore non-registered
        mutable state before the logged run.
    """

    try:
        copied_model = copy.deepcopy(model)
        _strip_copied_forward_decorations(copied_model)
        return copied_model, None
    except Exception:
        model_type = type(model)
        if model_type not in _VALIDATION_DEEPCOPY_WARNING_TYPES:
            _VALIDATION_DEEPCOPY_WARNING_TYPES.add(model_type)
            warnings.warn(
                "TorchLens validate_forward_pass could not deepcopy the model for the "
                "ground-truth run; falling back to state_dict snapshot/restore. "
                "Non-registered mutable state may cause false negatives for this model.",
                RuntimeWarning,
                stacklevel=3,
            )
        try:
            return model, _ModuleTreePlainAttrSnapshot(model)
        except RuntimeError:
            warnings.warn(
                "TorchLens validate_forward_pass could not snapshot non-registered mutable "
                "state after deepcopy failed; falling back to live model state without "
                "plain-attribute restoration.",
                RuntimeWarning,
                stacklevel=3,
            )
            return model, None


def _strip_copied_forward_decorations(model: nn.Module) -> None:
    """Remove TorchLens forward wrappers copied from another module instance.

    Parameters
    ----------
    model:
        Deep-copied validation model whose submodules may carry instance-level
        TorchLens forward wrappers closing over the source modules.
    """

    from .backends.torch._tl import is_forward_call_decorated

    for module in model.modules():
        current_forward = module.__dict__.get("forward", None)
        if current_forward is None or not is_forward_call_decorated(current_forward):
            continue
        original_forward = getattr(current_forward, "__wrapped__", None)
        if original_forward is not None and getattr(original_forward, "__self__", module) is module:
            module.forward = cast(Callable[..., Any], original_forward)
        else:
            module.__dict__.pop("forward", None)


def _restore_simple_plain_attrs_on_copy(source: nn.Module, copied: nn.Module) -> None:
    """Align simple plain attributes on a validation copy with its source.

    Parameters
    ----------
    source:
        Original module tree.
    copied:
        Deep-copied module tree that should represent ``source`` at validation
        entry.
    """

    simple_types = (type(None), bool, int, float, complex, str, bytes)
    for source_module, copied_module in zip(source.modules(), copied.modules()):
        source_names = _module_plain_attr_names(source_module)
        copied_names = _module_plain_attr_names(copied_module)
        for name in sorted(source_names & copied_names):
            source_value = getattr(source_module, name)
            copied_value = getattr(copied_module, name)
            if not isinstance(source_value, simple_types):
                continue
            if not isinstance(copied_value, simple_types):
                continue
            if source_value != copied_value:
                setattr(copied_module, name, source_value)


def _model_for_validation_replay(
    model: nn.Module,
) -> tuple[nn.Module, _ModuleTreePlainAttrSnapshot | None, bool]:
    """Return a model instance to use for validation capture and replay.

    Parameters
    ----------
    model:
        Original model being validated.

    Returns
    -------
    tuple[nn.Module, _ModuleTreePlainAttrSnapshot | None, bool]
        A deep-copied model when possible, optional plain-attribute snapshot
        for fallback restoration, and whether the copy succeeded.
    """

    try:
        copied_model = copy.deepcopy(model)
        _strip_copied_forward_decorations(copied_model)
        _restore_simple_plain_attrs_on_copy(model, copied_model)
        return copied_model, None, True
    except Exception:
        warnings.warn(
            "TorchLens validation replay against live model state; model could not be copied.",
            RuntimeWarning,
            stacklevel=3,
        )
        try:
            return model, _ModuleTreePlainAttrSnapshot(model), False
        except RuntimeError:
            return model, None, False


def decide_recording_of_batch(trace: Trace, predicate: Callable[[Trace], bool]) -> bool:
    """Retroactively keep or discard a captured batch log.

    Parameters
    ----------
    trace:
        Captured log to decide on.
    predicate:
        Callable receiving the log and returning whether to keep it.

    Returns
    -------
    bool
        True when the log was kept.
    """

    keep = bool(predicate(trace))
    if not keep:
        trace.cleanup()
    trace.recording_kept = keep
    return keep


def _qualname_for_model(model: nn.Module) -> str:
    """Return a stable class name for relationship evidence.

    Parameters
    ----------
    model:
        Model being captured.

    Returns
    -------
    str
        Module-qualified class name.
    """

    model_type = type(model)
    return f"{model_type.__module__}.{model_type.__qualname__}"


def _fingerprint_model_weights(model: nn.Module) -> str:
    """Fingerprint model parameter metadata for relationship evidence.

    Phase 4a does not depend on tensor values. The deterministic scheme hashes
    ``(name, shape, dtype)`` for every named parameter, which is stable across
    devices and avoids retaining parameter references.

    Parameters
    ----------
    model:
        Model whose parameters should be fingerprinted.

    Returns
    -------
    str
        SHA-256 hex digest of parameter metadata.
    """

    entries = [
        (name, tuple(param.shape), str(param.dtype)) for name, param in model.named_parameters()
    ]
    return hashlib.sha256(repr(entries).encode("utf-8")).hexdigest()


def _iter_tensor_inputs(obj: Any) -> list[torch.Tensor]:
    """Collect tensor leaves from a nested input object.

    Parameters
    ----------
    obj:
        Arbitrary nested input object.

    Returns
    -------
    list[torch.Tensor]
        Tensor leaves in traversal order.
    """

    tensors: list[torch.Tensor] = []
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, dict):
        for key in sorted(obj.keys(), key=repr):
            tensors.extend(_iter_tensor_inputs(obj[key]))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            tensors.extend(_iter_tensor_inputs(item))
    return tensors


def _input_id_for_relationship_evidence(input_args: Any) -> int:
    """Return the input identity used for relationship evidence.

    Parameters
    ----------
    input_args:
        User-provided positional input container.

    Returns
    -------
    int
        ``id`` of the sole input tensor when available, otherwise ``id`` of
        the input container.
    """

    tensors = _iter_tensor_inputs(input_args)
    if len(tensors) == 1:
        return id(tensors[0])
    return id(input_args)


def _hash_input_signatures(input_args: Any, input_kwargs: Any) -> str:
    """Fingerprint input tensor shape metadata for relationship evidence.

    Parameters
    ----------
    input_args:
        Positional input container.
    input_kwargs:
        Keyword input container.

    Returns
    -------
    str
        SHA-256 hex digest over tensor shapes, dtypes, and devices.
    """

    tensors = _iter_tensor_inputs((input_args, input_kwargs))
    entries = [(tuple(tensor.shape), str(tensor.dtype), str(tensor.device)) for tensor in tensors]
    return hashlib.sha256(repr(entries).encode("utf-8")).hexdigest()


def _hash_tensor_content(tensor: torch.Tensor) -> str:
    """Return a content hash for a tensor.

    Parameters
    ----------
    tensor:
        Tensor to hash.

    Returns
    -------
    str
        SHA-256 digest over tensor metadata and CPU bytes.
    """

    with _state.pause_logging():
        cpu = tensor.detach().cpu().contiguous()
        if cpu.dtype is torch.bfloat16:
            cpu = cpu.to(torch.float32)
        payload = cpu.numpy().tobytes()
    hasher = hashlib.sha256()
    hasher.update(repr((tuple(cpu.shape), str(cpu.dtype))).encode("utf-8"))
    hasher.update(payload)
    return hasher.hexdigest()


def _hash_nested_tensor_content(value: Any) -> str:
    """Return a deterministic content hash for nested tensor inputs.

    Parameters
    ----------
    value:
        Nested tensor container.

    Returns
    -------
    str
        SHA-256 digest.
    """

    tensors = _iter_tensor_inputs(value)
    entries = [_hash_tensor_content(tensor) for tensor in tensors]
    return hashlib.sha256(repr(entries).encode("utf-8")).hexdigest()


def _fingerprint_model_content(model: nn.Module) -> str:
    """Fingerprint model tensor contents for the capture cache.

    Parameters
    ----------
    model:
        Model to fingerprint.

    Returns
    -------
    str
        SHA-256 digest.
    """

    hasher = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        hasher.update(name.encode("utf-8"))
        hasher.update(_hash_tensor_content(tensor).encode("utf-8"))
    return hasher.hexdigest()


def _capture_cache_dir(cache_dir: str | Path | None) -> Path:
    """Resolve the capture-cache directory.

    Parameters
    ----------
    cache_dir:
        Optional user-specified directory.

    Returns
    -------
    pathlib.Path
        Cache directory path.
    """

    if cache_dir is not None:
        return Path(cache_dir)
    return Path(os.environ.get("TORCHLENS_CACHE_DIR", "~/.cache/torchlens")).expanduser()


def _capture_cache_key(
    model: nn.Module,
    input_args: Any,
    input_kwargs: Any,
    config: dict[str, Any],
) -> str:
    """Build a content-hash capture-cache key.

    Parameters
    ----------
    model:
        Model being captured.
    input_args:
        Positional inputs.
    input_kwargs:
        Keyword inputs.
    config:
        Capture configuration values.

    Returns
    -------
    str
        SHA-256 cache key.
    """

    payload = {
        "schema": 1,
        "torch": torch.__version__,
        "model": _fingerprint_model_content(model),
        "inputs": _hash_nested_tensor_content((input_args, input_kwargs)),
        "config": config,
    }
    encoded = json.dumps(payload, sort_keys=True, default=repr).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _facet_recipe_cache_key(
    recipes: list[Callable[[Any], dict[str, Any]]]
    | tuple[Callable[[Any], dict[str, Any]], ...]
    | None,
) -> tuple[str, ...]:
    """Return stable-ish recipe identities for capture cache separation.

    Parameters
    ----------
    recipes:
        Per-trace recipe functions supplied to ``trace``.

    Returns
    -------
    tuple[str, ...]
        Function module/qualname identities for cache configuration.
    """

    if recipes is None:
        return ()
    return tuple(f"{recipe.__module__}.{recipe.__qualname__}" for recipe in recipes)


def _capture_output_metadata_from_model_config(trace: Trace, model: nn.Module) -> None:
    """Capture portable output metadata from ``model.config`` into ``trace``.

    Parameters
    ----------
    trace:
        Trace receiving the in-band output metadata.
    model:
        Model being captured.
    """

    try:
        config = getattr(model, "config", None)
    except Exception:
        # ``config`` may be a property whose getter raises for reasons unrelated to
        # attribute existence (e.g. delegating to a submodule that only partially
        # implements it). This is best-effort output metadata capture, not a
        # validation check, so a raising getter degrades to "no config metadata"
        # rather than aborting the whole capture.
        return
    if config is None:
        return

    id2label = getattr(config, "id2label", None)
    if isinstance(id2label, dict):
        normalized_id2label: dict[int, str] = {}
        for key, value in id2label.items():
            try:
                normalized_key = int(key)
            except (TypeError, ValueError):
                continue
            normalized_id2label[normalized_key] = str(value)
        trace.output_id2label = normalized_id2label or None

    num_labels = getattr(config, "num_labels", None)
    if num_labels is None and trace.output_id2label is not None:
        num_labels = len(trace.output_id2label)
    if num_labels is not None:
        try:
            trace.output_num_classes = int(num_labels)
        except (TypeError, ValueError):
            trace.output_num_classes = None


def _prepare_log_for_capture_cache(trace: Trace) -> None:
    """Detach non-leaf tensors and autograd objects before cache serialization.

    Parameters
    ----------
    trace:
        Log to make pickle-compatible in place.
    """

    for layer in getattr(trace, "layer_list", []):
        for field_name in (
            "out",
            "transformed_out",
            "grad",
            "transformed_grad",
        ):
            value = _raw_cache_payload_field(layer, field_name)
            if isinstance(value, torch.Tensor):
                layer._internal_set(field_name, value.detach().cpu())
        layer.grad_fn_handle = None
        layer._internal_set("saved_args", _detach_nested_for_cache(layer.saved_args))
        layer._internal_set("saved_kwargs", _detach_nested_for_cache(layer.saved_kwargs))
    for layer_log in getattr(trace, "layer_logs", {}).values():
        # ``Layer.transformed_out``/``transformed_grad`` are read-only proxies that read
        # through to the underlying ``Layer.ops[i]`` -- the real tensors live on the Op
        # objects, which expose ``_internal_set`` (the Layer view does not, so a raw
        # ``setattr`` here raised ``can't set attribute``). Detach on the ops directly,
        # mirroring the ``layer_list`` loop above, so a cached capture with an
        # activation/grad transform stays pickle-safe.
        for op in layer_log.ops.values():
            for field_name in ("transformed_out", "transformed_grad"):
                value = _raw_cache_payload_field(op, field_name)
                if isinstance(value, torch.Tensor):
                    op._internal_set(field_name, value.detach().cpu())
        layer_log.grad_fn_handle = None
    trace.__dict__.pop("_container_ordinals_by_output_op_label", None)
    trace.__dict__.pop("_container_ordinals_by_input_func_call_id", None)
    if trace.__dict__.get("_predicate_save_options") is not None:
        trace.__dict__["_predicate_save_options"] = "cache_predicate_capture"
    trace.__dict__.pop("_capture_config", None)
    trace.__dict__.pop("_stop_directive", None)
    trace.__dict__.pop("_capture_events", None)
    wrapper_ws = trace.__dict__.get("_wrapper_runtime_ws")
    if wrapper_ws is not None:
        registry = getattr(wrapper_ws, "container_registry", None)
        if registry is not None:
            registry.clear_live_state()
    for workspace_key in ("_raw_graph_ws", "_module_capture_ws", "_wrapper_runtime_ws"):
        trace.__dict__.pop(workspace_key, None)


def _raw_cache_payload_field(layer: Any, field_name: str) -> Any:
    """Return a cache payload field without invoking strict public accessors.

    Parameters
    ----------
    layer:
        Op-like object being prepared for pickle serialization.
    field_name:
        Payload field to read.

    Returns
    -------
    Any
        Raw slot value when present, otherwise ``None``.
    """

    try:
        return object.__getattribute__(layer, field_name)
    except AttributeError:
        return None


def _detach_nested_for_cache(value: Any) -> Any:
    """Detach tensors inside a nested cache payload.

    Parameters
    ----------
    value:
        Nested value.

    Returns
    -------
    Any
        Value with tensors detached to CPU.
    """

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_detach_nested_for_cache(item) for item in value)
    if isinstance(value, list):
        return [_detach_nested_for_cache(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach_nested_for_cache(item) for key, item in value.items()}
    return value


if TYPE_CHECKING:
    pass


def _unwrap_data_parallel(model: nn.Module) -> nn.Module:
    """Return the underlying ``nn.Module`` if ``model`` is a data-parallel wrapper.

    Handles:
      * ``nn.DataParallel``              -> unwrap via ``.module``
      * ``nn.parallel.DistributedDataParallel`` -> unwrap via ``.module``
      * ``torch.distributed.fsdp.FullyShardedDataParallel`` -> raise

    FSDP cannot be unwrapped the same way: its parameters are sharded across
    ranks, so there is no single unsharded module to log. Users who want to
    log an FSDP-wrapped model should ``trace`` a rank-local
    *un-wrapped* copy of the underlying module instead.

    The function is kept under its original name to avoid churn at call sites;
    the historical ``_unwrap_data_parallel`` now covers the full data-parallel
    family.
    """
    # FSDP: fail loudly rather than silently mis-attributing sharded params.
    # The lazy probe never imports torch.distributed.fsdp on plain captures.
    fsdp_wrapper_type = get_fsdp_wrapper_type()
    if fsdp_wrapper_type is not None and isinstance(model, fsdp_wrapper_type):
        raise RuntimeError(
            "torchlens.trace does not support "
            "FullyShardedDataParallel (FSDP): parameters are sharded "
            "across ranks and there is no unsharded module to log. "
            "Run trace on a rank-local copy of the underlying "
            "module (before FSDP wrapping) instead."
        )

    # DistributedDataParallel: unwrap via ``.module`` (same layout as DataParallel).
    try:
        from torch.nn.parallel import DistributedDataParallel
    except ImportError:
        pass
    else:
        if isinstance(model, DistributedDataParallel):
            return cast(nn.Module, model.module)

    # DataParallel: the original case this helper covered.
    if isinstance(model, nn.DataParallel):
        return cast(nn.Module, model.module)

    return model


def _reject_opaque_wrappers(model: nn.Module) -> None:
    """Raise a clear error if ``model`` is one of the opaque wrappers TorchLens cannot trace.

    TorchLens logs a model by wrapping every torch callable and running an
    ordinary Python forward pass.  The following wrappers all replace that
    Python execution with a traced / scripted / exported graph — by design,
    our wrappers don't see the original ops, so the Trace would be
    empty or misleading:

    * ``torch.jit.ScriptModule`` / ``torch.jit.RecursiveScriptModule``
      (``torch.jit.script`` / ``torch.jit.trace``) — the forward runs on the
      TorchScript interpreter, not Python, so no Python-level decoration fires.
    * ``torch.export.ExportedProgram`` — a serialised IR, not a callable
      ``nn.Module`` that can be re-executed in Python.
    * ``torch.distributed.fsdp.FullyShardedDataParallel`` — FSDP controls
      parameter materialization and sharding around forward execution in ways
      TorchLens cannot currently validate.

    In these cases the fix is the same: call ``trace`` on the
    *un-wrapped* model before scripting, exporting, or sharding.
    """
    # FullyShardedDataParallel; the lazy probe never imports
    # torch.distributed.fsdp on plain captures.
    fsdp_wrapper_type = get_fsdp_wrapper_type()
    if fsdp_wrapper_type is not None and isinstance(model, fsdp_wrapper_type):
        raise RuntimeError(
            "torchlens.trace does not support "
            "FullyShardedDataParallel models: FSDP controls parameter "
            "materialization and sharding around forward execution in ways "
            "TorchLens cannot validate. Call trace on the "
            "underlying unwrapped nn.Module."
        )

    # torch.jit.script / torch.jit.trace -> ScriptModule. Descendants are just
    # as opaque as a scripted root: their forward executes in the TorchScript
    # interpreter and their non-state_dict attributes cannot be restored by the
    # validation fallback.
    scripted_module = next(
        (
            (address, module)
            for address, module in model.named_modules()
            if isinstance(module, torch.jit.ScriptModule)
        ),
        None,
    )
    if scripted_module is not None:
        address, _module = scripted_module
        location = "model root" if address == "" else f"submodule '{address}'"
        raise RuntimeError(
            "torchlens.trace does not support torch.jit ScriptModule "
            f"or traced models ({location}): the forward runs on the TorchScript interpreter "
            "rather than Python, so TorchLens' function wrappers don't fire. "
            "Call trace on the original (un-scripted / un-traced) "
            "model."
        )

    # torch.export.ExportedProgram
    try:
        from torch.export import ExportedProgram
    except ImportError:
        pass
    else:
        if isinstance(model, ExportedProgram):
            raise RuntimeError(
                "torchlens.trace does not support "
                "torch.export.ExportedProgram: the exported IR is not a "
                "callable nn.Module that can be re-executed in Python. "
                "Call trace on the original nn.Module before "
                "export."
            )


def _move_tensors_to_device(obj: Any, device: torch.device | str) -> Any:
    """Recursively move tensors in a nested structure (lists, tuples, dicts) to *device*.

    Handles common dict-like types (OrderedDict, HuggingFace BatchEncoding, etc.)
    by attempting to reconstruct the original container type after moving values.
    NamedTuple subclasses (e.g. a GNN model's batch container, which is also an
    ``isinstance(obj, tuple)`` match) are reconstructed through their own type so
    downstream named-field access keeps working instead of silently degrading to
    a plain ``tuple``.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, (list, tuple)):
        moved_sequence = [_move_tensors_to_device(item, device) for item in obj]
        if not isinstance(obj, tuple):
            return type(obj)(moved_sequence)
        obj_type = type(obj)
        if hasattr(obj_type, "_fields"):
            return obj_type(*moved_sequence)
        return tuple(moved_sequence)
    elif isinstance(obj, collections.abc.MutableMapping):
        # Handles dict, UserDict, BatchEncoding, OrderedDict, etc.
        moved_mapping = {k: _move_tensors_to_device(v, device) for k, v in obj.items()}
        if type(obj) is dict:
            return moved_mapping
        try:
            return cast(Any, type(obj))(moved_mapping)
        except Exception:
            return moved_mapping
    return obj
