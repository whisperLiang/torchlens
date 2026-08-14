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

from .._input_walk import INPUT_TREE_MAX_DEPTH
from .tensor_utils import (
    TensorByteFootprint,
    _clone_tensor_payload,
    _copy_tensor_payload,
    tensor_byte_footprint,
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


def copy_arg_tree(arg: Any, _in_progress: dict[int, Any] | None = None, _depth: int = 0) -> Any:
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
    copy and the cycle is reproduced in the copy.  The memo is CALL-scoped
    (retained across siblings, r-b4 R29-3): a DAG-shaped input that reuses one
    sub-container under several paths is copied ONCE and stays aliased in the
    copy -- which both matches the aliasing topology the model itself would see
    and makes the copy O(nodes).  The historical path-scoped memo copied a
    shared node once per PATH, i.e. exponentially in shared-substructure depth
    (measured x2 per level; depth 25 hung capture entry for ~4 minutes).
    Tensors remain leaves cloned per DISTINCT container occurrence and are
    never memoized.

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
    _depth
        Internal recursion depth used to enforce the shared input-boundary
        nesting ceiling (r-b4 R27-1). Callers should not supply this.

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
        # Cycle (a container reachable from itself) or DAG reuse (one container
        # under several paths): both resolve to the one memoized copy.
        return existing
    if isinstance(arg, (defaultdict, dict, list, tuple)) and _depth >= INPUT_TREE_MAX_DEPTH:
        # r-b4 R27-1: the canonical per-capture input copier is depth-bounded with the
        # SAME shared ceiling as every other input-boundary walker -- a deeper tree
        # refuses typed at capture entry instead of dying in a raw RecursionError.
        from .._input_walk import raise_input_tree_depth_refusal

        raise_input_tree_depth_refusal(depth=_depth)
    if isinstance(arg, defaultdict):
        # defaultdict(factory, {k: v, ...}) — preserve the default_factory (#127).
        # A plain dict() constructor would lose default_factory.
        copied: Any = defaultdict(arg.default_factory)
        _in_progress[arg_id] = copied
        for key, value in arg.items():
            copied[key] = copy_arg_tree(value, _in_progress, _depth + 1)
        return copied
    elif isinstance(arg, dict):
        # type(arg)() preserves OrderedDict and other dict subclasses; populate
        # after registering so a cyclic value can point back at this copy.
        copied = type(arg)()
        _in_progress[arg_id] = copied
        for key, value in arg.items():
            copied[key] = copy_arg_tree(value, _in_progress, _depth + 1)
        return copied
    elif isinstance(arg, list):
        copied = type(arg)()
        _in_progress[arg_id] = copied
        for item in arg:
            copied.append(copy_arg_tree(item, _in_progress, _depth + 1))
        return copied
    elif isinstance(arg, tuple):
        # Tuples are immutable and cannot self-reference directly; any cycle
        # through a tuple passes through a mutable container that is already
        # registered above, so recursing eagerly here is safe.
        items = [copy_arg_tree(item, _in_progress, _depth + 1) for item in arg]
        # NamedTuples have _fields and need *args construction; plain tuples
        # take an iterable. Memoized after construction (immutable, so no cycle
        # can pass through the tuple itself) so tuple-shaped DAGs are O(nodes).
        copied = type(arg)(*items) if hasattr(type(arg), "_fields") else type(arg)(items)
        _in_progress[arg_id] = copied
        return copied
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

    Notes
    -----
    r-b4 R29-2: the historical implementation ran ``touched_bytes_relation`` on
    every pair -- O(T^2) in tensor-leaf count (measured 0.72 s at 4k leaves even
    with every storage disjoint). Candidate pairs are now pre-filtered by a
    device-scoped byte-interval sweep (:func:`_alias_candidate_pairs`) that drops
    ONLY pairs the relation ladder provably answers ``disjoint`` from bounding
    intervals alone; every surviving pair still runs the exact same per-pair
    ladder, so verdicts and diagnostic text are byte-identical.
    """

    if len(tensor_records) < 2:
        return
    footprints: list[TensorByteFootprint | None] = []
    for _left_path, tensor, _transparent in tensor_records:
        try:
            footprints.append(tensor_byte_footprint(tensor))
        except (RuntimeError, TypeError, NotImplementedError):
            footprints.append(None)
    for left_index, right_index in sorted(_alias_candidate_pairs(tensor_records, footprints)):
        left_path, left, left_transparent = tensor_records[left_index]
        right_path, right, right_transparent = tensor_records[right_index]
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


def _alias_candidate_pairs(
    tensor_records: list[tuple[str, torch.Tensor, bool]],
    footprints: list["TensorByteFootprint | None"],
) -> set[tuple[int, int]]:
    """Return the record-index pairs the alias scan cannot silently skip (r-b4 R29-2).

    A pair is EXCLUDED only when ``touched_bytes_relation`` provably answers
    ``disjoint`` without a per-pair proof: both footprints known, distinct
    objects, and either (a) an empty view on one side, (b) distinct device
    types, (c) same device type with two CONCRETE, different indexes, or
    (d) the same device key with non-overlapping absolute byte intervals.
    Everything else -- identity pairs, unprovable footprints, same-type
    None-vs-concrete device indexes, interval overlaps -- stays a candidate and
    runs the unchanged exact ladder.

    Parameters
    ----------
    tensor_records:
        Tensor input paths, original tensors, and storage-deepcopy eligibility.
    footprints:
        Pre-computed ``tensor_byte_footprint`` per record (``None`` = unprovable).

    Returns
    -------
    set[tuple[int, int]]
        Candidate ``(left_index, right_index)`` pairs with ``left < right``.
    """

    total = len(tensor_records)
    candidates: set[tuple[int, int]] = set()

    # Identity pairs (one object at several sites) always reach the loop body:
    # the caller's identity branch decides them before any footprint logic.
    by_identity: dict[int, list[int]] = {}
    for index, (_path, tensor, _transparent) in enumerate(tensor_records):
        by_identity.setdefault(id(tensor), []).append(index)
    for indices in by_identity.values():
        for position, left_index in enumerate(indices):
            for right_index in indices[position + 1 :]:
                candidates.add((left_index, right_index))

    # An unprovable footprint relates ``unknown`` to every partner (checked
    # before the empty-view rule in the ladder, so zero-numel partners count).
    for index, footprint in enumerate(footprints):
        if footprint is not None:
            continue
        for other in range(total):
            if other != index:
                candidates.add((min(index, other), max(index, other)))

    # Provable footprints with a nonzero span, grouped by exact device key as
    # (start_byte, end_byte, record_index) interval entries.
    by_device_key: dict[tuple[str, int | None], list[tuple[int, int, int]]] = {}
    for index, footprint in enumerate(footprints):
        if footprint is None or footprint.numel == 0:
            continue
        by_device_key.setdefault(footprint.device_key, []).append(
            (footprint.start_byte, footprint.end_byte, index)
        )

    # Same device TYPE under two different keys is provably disjoint only when
    # both indexes are concrete; a None-vs-concrete index answers ``unknown``.
    keys_by_type: dict[str, list[tuple[str, int | None]]] = {}
    for device_key in by_device_key:
        keys_by_type.setdefault(device_key[0], []).append(device_key)
    for device_keys in keys_by_type.values():
        for key_position, left_key in enumerate(device_keys):
            for right_key in device_keys[key_position + 1 :]:
                if left_key[1] is not None and right_key[1] is not None:
                    continue
                for _, _, left_index in by_device_key[left_key]:
                    for _, _, right_index in by_device_key[right_key]:
                        candidates.add(
                            (min(left_index, right_index), max(left_index, right_index))
                        )

    # Interval sweep inside one exact device key: only pairs whose absolute
    # byte spans overlap survive (disjoint spans are the ladder's own verdict).
    for entries in by_device_key.values():
        if len(entries) < 2:
            continue
        entries.sort()
        active: list[tuple[int, int]] = []  # (end_byte, record_index)
        for start_byte, end_byte, index in entries:
            active = [entry for entry in active if entry[0] > start_byte]
            for _, other in active:
                candidates.add((min(index, other), max(index, other)))
            active.append((end_byte, index))

    return candidates


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
