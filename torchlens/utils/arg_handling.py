"""Argument copying, normalization, and input validation for model forward calls.

Provides safe argument copying that avoids ``copy.deepcopy`` (which can
trigger infinite loops on complex tensor wrappers with circular references,
e.g. ESCNN GeometricTensor) and normalizes user-supplied ``input_args`` into
the ``list`` form expected by ``model(*input_args)``.
"""

import copy
import inspect
from collections import defaultdict
from typing import Any, cast

import torch
from torch import nn

from .tensor_utils import (
    _clone_tensor_payload,
    _copy_tensor_payload,
    touched_bytes_relation,
)

INPUT_WAS_PARAMETER_ATTR = "_torchlens_input_was_parameter"


def _clone_input_tensor_payload(arg: torch.Tensor) -> torch.Tensor:
    """Clone a forward-input tensor without preserving ``nn.Parameter`` type.

    Parameters
    ----------
    arg
        Forward input tensor or parameter to clone.

    Returns
    -------
    torch.Tensor
        Plain tensor clone for model input replay. Parameter inputs intentionally
        become tensors so input provenance wins over Python wrapper type.
    """

    if isinstance(arg, torch.nn.Parameter):
        from .._state import pause_logging

        with pause_logging():
            cloned = _copy_tensor_payload(arg, detach_tensor=False, save_mode="copy")
        setattr(cloned, INPUT_WAS_PARAMETER_ATTR, True)
        return cloned
    return cast(torch.Tensor, _clone_tensor_payload(arg, detach_tensor=False, save_mode="copy"))


def copy_arg_tree(arg: Any, _in_progress: dict[int, Any] | None = None) -> Any:
    """Copy an input argument tree, cloning tensors and recursing built-in containers.

    Why not ``copy.deepcopy``?  Many third-party tensor wrappers hold
    circular references back to their parent modules.  ``deepcopy`` follows
    these references recursively and either hangs or blows the stack.
    Instead we clone tensors (which copies storage without following Python
    object references) and recurse only into standard containers.

    Clones tensors, recurses into standard containers (list, tuple, dict),
    and leaves everything else as-is.  This prevents infinite loops that
    copy.deepcopy can trigger on complex tensor wrappers (e.g. ESCNN
    GeometricTensor) while still protecting user inputs from in-place
    mutations like device moves.

    Standard containers that refer back to themselves (directly or through a
    cycle of mutable containers) are handled without a ``RecursionError``: a
    mutable container being built is registered in ``_in_progress`` before its
    elements are recursed, so a self-reference resolves to the same in-progress
    copy and the cycle is reproduced in the copy.  The registration is scoped to
    the active recursion path only, so a non-cyclic structure that reuses the
    same sub-container twice is still copied twice (unchanged behavior).

    Note: custom objects containing tensors are passed by reference.  If the
    model is on a different device, _fetch_label_move_input_tensors may
    mutate the wrapper's tensor attribute in-place.  This is acceptable
    because pre-fix such inputs caused an infinite hang.

    Parameters
    ----------
    arg
        Argument value to copy.
    _in_progress
        Internal recursion state mapping ``id()`` of a mutable container being
        built to its (partially populated) copy, used to terminate reference
        cycles. Callers should not supply this.

    Returns
    -------
    Any
        Argument copy with nested tensors cloned and custom objects preserved
        by reference.
    """
    if _in_progress is None:
        _in_progress = {}
    if isinstance(arg, torch.Tensor):
        # Tensors are leaves and are cloned per occurrence (never memoized), so a
        # structure that reuses the same tensor keeps its historical per-slot
        # clone semantics.
        return _clone_input_tensor_payload(arg)
    arg_id = id(arg)
    existing = _in_progress.get(arg_id)
    if existing is not None:
        # A container on the current recursion path referred back to itself.
        return existing
    if isinstance(arg, defaultdict):
        # defaultdict(factory, {k: v, ...}) — preserve the default_factory (#127).
        # A plain dict() constructor would lose default_factory.
        copied: Any = defaultdict(arg.default_factory)
        _in_progress[arg_id] = copied
        try:
            for key, value in arg.items():
                copied[key] = copy_arg_tree(value, _in_progress)
        finally:
            _in_progress.pop(arg_id, None)
        return copied
    elif isinstance(arg, dict):
        # type(arg)() preserves OrderedDict and other dict subclasses; populate
        # after registering so a cyclic value can point back at this copy.
        copied = type(arg)()
        _in_progress[arg_id] = copied
        try:
            for key, value in arg.items():
                copied[key] = copy_arg_tree(value, _in_progress)
        finally:
            _in_progress.pop(arg_id, None)
        return copied
    elif isinstance(arg, list):
        copied = type(arg)()
        _in_progress[arg_id] = copied
        try:
            for item in arg:
                copied.append(copy_arg_tree(item, _in_progress))
        finally:
            _in_progress.pop(arg_id, None)
        return copied
    elif isinstance(arg, tuple):
        # Tuples are immutable and cannot self-reference directly; any cycle
        # through a tuple passes through a mutable container that is already
        # registered above, so recursing eagerly here is safe.
        items = [copy_arg_tree(item, _in_progress) for item in arg]
        # NamedTuples have _fields and need *args construction; plain tuples
        # take an iterable.
        return type(arg)(*items) if hasattr(type(arg), "_fields") else type(arg)(items)
    else:
        # Non-container, non-tensor objects (ints, strings, custom wrappers)
        # are returned by reference — shallow enough to avoid circular ref issues.
        return arg


def _safe_copy_arg(arg: Any) -> Any:
    """Compatibility alias for :func:`copy_arg_tree`.

    Parameters
    ----------
    arg
        Argument value to copy.

    Returns
    -------
    Any
        Recursive input-argument copy result.
    """

    return copy_arg_tree(arg)


def safe_copy_args(args: list[Any]) -> list[Any]:
    """Safely copy a list of positional arguments.

    Each element is copied via :func:`copy_arg_tree`: tensors are cloned,
    containers are recursed into, and everything else is passed by reference.
    """
    return [copy_arg_tree(arg) for arg in args]


def safe_copy_kwargs(kwargs: dict[Any, Any]) -> dict[Any, Any]:
    """Safely copy a dict of keyword arguments.

    Same semantics as :func:`safe_copy_args` but for ``**kwargs``.
    """
    return {key: copy_arg_tree(val) for key, val in kwargs.items()}


def _prepare_input_deepcopy_memo(
    value: Any,
    *,
    path: str,
    memo: dict[int, Any],
    visited: set[int],
    semantic_gaps: list[str],
    tensor_records: list[tuple[str, torch.Tensor, bool]],
) -> None:
    """Prepare a shared ``deepcopy`` memo for one input-tree value.

    Parameters
    ----------
    value:
        Input-tree value to inspect.
    path:
        Human-readable location used for fail-closed diagnostics.
    memo:
        Shared ``deepcopy`` memo for positional and keyword inputs.
    visited:
        Object identities already traversed while preparing the memo.
    semantic_gaps:
        Accumulator for tensor kinds that cannot use PyTorch's topology-preserving
        deepcopy protocol.
    tensor_records:
        Tensor paths, originals, and whether storage-preserving deepcopy remains
        available for cross-tensor alias checks.

    Returns
    -------
    None
        Mutates ``memo``, ``visited``, and ``semantic_gaps`` in place.
    """

    if isinstance(value, torch.nn.Parameter):
        value_id = id(value)
        tensor_records.append((path, value, False))
        if value_id in visited:
            return
        visited.add(value_id)
        cloned = _clone_input_tensor_payload(value)
        memo[value_id] = cloned
        return
    if isinstance(value, torch.Tensor):
        value_id = id(value)
        deepcopy_preserves_contract = (
            value.is_leaf and not value.requires_grad and type(value) is torch.Tensor
        )
        tensor_records.append((path, value, deepcopy_preserves_contract))
        if value_id in visited:
            return
        visited.add(value_id)
        if not deepcopy_preserves_contract:
            cloned = _clone_input_tensor_payload(value)
            memo[value_id] = cloned
            try:
                physical_metadata_changed = (
                    tuple(cloned.shape) != tuple(value.shape)
                    or tuple(cloned.stride()) != tuple(value.stride())
                    or cloned.storage_offset() != value.storage_offset()
                )
            except (RuntimeError, TypeError, NotImplementedError):
                physical_metadata_changed = True
            if physical_metadata_changed:
                semantic_gaps.append(
                    f"{path}: grad-preserving tensor clone changed physical view metadata"
                )
        return
    value_id = id(value)
    if value_id in visited:
        return
    visited.add(value_id)
    if isinstance(value, dict):
        for index, (key, child) in enumerate(value.items()):
            _prepare_input_deepcopy_memo(
                key,
                path=f"{path}.<key:{index}>",
                memo=memo,
                visited=visited,
                semantic_gaps=semantic_gaps,
                tensor_records=tensor_records,
            )
            _prepare_input_deepcopy_memo(
                child,
                path=f"{path}.<value:{index}>",
                memo=memo,
                visited=visited,
                semantic_gaps=semantic_gaps,
                tensor_records=tensor_records,
            )
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _prepare_input_deepcopy_memo(
                child,
                path=f"{path}.{index}",
                memo=memo,
                visited=visited,
                semantic_gaps=semantic_gaps,
                tensor_records=tensor_records,
            )
        return
    # Preserve the established contract for custom wrappers: pass them by
    # reference rather than following arbitrary attributes through deepcopy.
    memo[value_id] = value


def _record_unpreserved_tensor_aliases(
    tensor_records: list[tuple[str, torch.Tensor, bool]],
    semantic_gaps: list[str],
    *,
    require_distinct_tensor_sites: bool,
) -> None:
    """Record aliases involving tensors that require grad-preserving clones.

    Parameters
    ----------
    tensor_records:
        Tensor input paths, original tensors, and storage-deepcopy eligibility.
    semantic_gaps:
        Accumulator receiving fail-closed alias-topology diagnostics.
    require_distinct_tensor_sites:
        Whether the downstream runnable descriptor requires every model-input
        site to have distinct identity and storage.

    Returns
    -------
    None
        Appends one diagnostic for every potentially overlapping pair whose
        topology cannot be preserved without severing autograd linkage.
    """

    for left_index, (left_path, left, left_transparent) in enumerate(tensor_records):
        for right_path, right, right_transparent in tensor_records[left_index + 1 :]:
            if left is right:
                if require_distinct_tensor_sites:
                    semantic_gaps.append(
                        f"{left_path} <-> {right_path}: runnable input sites share "
                        "one tensor identity, which the sparse descriptor cannot encode"
                    )
                continue
            if left_transparent and right_transparent and not require_distinct_tensor_sites:
                continue
            try:
                relation = touched_bytes_relation(left, right)
            except (RuntimeError, TypeError, NotImplementedError):
                relation = "unknown"
            if relation == "disjoint":
                continue
            if require_distinct_tensor_sites:
                semantic_gaps.append(
                    f"{left_path} <-> {right_path}: runnable input sites have "
                    f"{relation} storage, which the sparse descriptor cannot encode"
                )
            else:
                semantic_gaps.append(
                    f"{left_path} <-> {right_path}: grad-preserving clones cannot prove "
                    f"the original tensor alias topology ({relation})"
                )


def safe_copy_input_tree(
    args: list[Any],
    kwargs: dict[Any, Any],
    *,
    require_distinct_tensor_sites: bool = False,
) -> tuple[list[Any], dict[Any, Any], tuple[str, ...]]:
    """Copy one complete model-input graph while preserving tensor topology.

    Positional and keyword inputs share one ``deepcopy`` memo. For ordinary leaf
    tensors without autograd history, PyTorch's deepcopy protocol copies each
    underlying storage once and rebuilds every view with its original size,
    stride, and storage offset. Grad-tracked tensors use the historical clone
    path so gradients still reach the caller's input. Every path shares one memo,
    so the same tensor repeated at multiple call sites remains one object.

    Parameters
    ----------
    args:
        Normalized positional model inputs.
    kwargs:
        Keyword model inputs.
    require_distinct_tensor_sites:
        Whether aliases between distinct input sites must fail closed because a
        downstream sparse runnable descriptor cannot encode them.

    Returns
    -------
    tuple[list[Any], dict[Any, Any], tuple[str, ...]]
        Copied positional inputs, copied keyword inputs, and semantic gaps that
        require capture verification to fail closed. Distinct overlapping
        grad-tracked views and unexpected deepcopy failures use the historical
        clone fallback but are explicitly reported as unverifiable.
    """

    memo: dict[int, Any] = {}
    visited: set[int] = set()
    semantic_gaps: list[str] = []
    tensor_records: list[tuple[str, torch.Tensor, bool]] = []
    _prepare_input_deepcopy_memo(
        args,
        path="input.args",
        memo=memo,
        visited=visited,
        semantic_gaps=semantic_gaps,
        tensor_records=tensor_records,
    )
    _prepare_input_deepcopy_memo(
        kwargs,
        path="input.kwargs",
        memo=memo,
        visited=visited,
        semantic_gaps=semantic_gaps,
        tensor_records=tensor_records,
    )
    _record_unpreserved_tensor_aliases(
        tensor_records,
        semantic_gaps,
        require_distinct_tensor_sites=require_distinct_tensor_sites,
    )
    try:
        from .._state import pause_logging

        with pause_logging():
            copied_args, copied_kwargs = copy.deepcopy((args, kwargs), memo)
    except Exception as exc:
        semantic_gaps.append(
            f"input: topology-preserving deepcopy failed with {type(exc).__name__}: {exc}"
        )
        copied_args = safe_copy_args(args)
        copied_kwargs = safe_copy_kwargs(kwargs)
    return copied_args, copied_kwargs, tuple(semantic_gaps)


def _model_expects_single_arg(model: nn.Module) -> bool | None:
    """Check if the model's forward expects exactly 1 positional arg (excluding self).

    Used by :func:`normalize_input_args` to disambiguate whether a user-supplied
    tuple is "multiple positional args" or "one arg that is a tuple."

    Returns:
        True if exactly 1 positional param, False if more or uses ``*args``,
        None if introspection fails (e.g. C-extension forward).
    """
    try:
        spec = inspect.getfullargspec(model.forward)
    except (TypeError, ValueError):
        # Introspection can fail on C-extension or dynamically-generated forward custom_methods.
        return None
    named_args = [a for a in spec.args if a != "self"]
    if spec.varargs is not None:
        # Has *args — could accept any number of positional args; can't tell.
        return False
    return len(named_args) == 1


def normalize_input_args(input_args: Any, model: nn.Module) -> list[Any]:
    """Normalize ``input_args`` into a list suitable for ``model(*input_args)``.

    Handles the ambiguity when the user ops a tuple or list: it could be
    multiple positional args, or a single arg that happens to be a
    tuple/list (issue #43).  Resolves the ambiguity by inspecting the
    model's ``forward`` signature via :func:`_model_expects_single_arg`:

    * If the model expects exactly one positional param and the user passed
      a multi-element tuple/list, wrap it in a list so it arrives as a
      single argument: ``model(the_tuple)``.
    * Otherwise, treat each element of the tuple/list as a separate
      positional argument: ``model(arg0, arg1, ...)``.
    """
    if type(input_args) in (tuple, list):
        single = _model_expects_single_arg(model)
        if single and len(input_args) != 1:
            # Model expects 1 arg but user passed a multi-element tuple/list.
            # The tuple/list itself IS the single argument.
            input_args = [input_args]
        elif type(input_args) is tuple:
            # Multiple args — convert tuple to mutable list for internal use.
            input_args = list(input_args)
        # If already a list and not wrapping, leave as-is.
    elif input_args is not None:
        # Bare value (single tensor, etc.) — wrap in a list.
        input_args = [input_args]
    if not input_args:
        input_args = []
    return cast(list[Any], input_args)
