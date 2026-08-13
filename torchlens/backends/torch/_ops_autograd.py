"""Autograd saved-tensor statistics and container snapshots."""

from typing import TYPE_CHECKING, Any
import torch
from ... import _state as _st
from ..._state import pause_logging
from ._tl import (
    get_tensor_label,
)
from .completeness_witness import internal_scalar_read
from ...utils.tensor_utils import (
    is_functorch_wrapped_tensor,
)
from ...utils.collections import ensure_iterable
from ...ir.container import (
    ContainerSpec,
    OutputPathComponent,
)
from ...ir.container_registry import (
    ContainerLeafOccurrence,
    FuncSite,
    Phase,
    Role,
    walk_container,
)

if TYPE_CHECKING:
    from ...data_classes.trace import Trace

from .ops import (
    _OutputTensorEntry,
)

if TYPE_CHECKING:
    from .ops import (
        _collect_tensor_values,
        _iter_autograd_saved_candidates,
        _walk_output_tensors_with_paths,
    )

__all__ = (
    "_add_autograd_saved_tensor",
    "_get_autograd_saved_stats_by_output",
    "_partition_output_entries_with_autograd_stats",
    "_register_call_output_container_snapshot",
    "register_call_input_container_snapshots",
    "_container_leaf_occurrences_from_entries",
    "_get_autograd_saved_stats_for_tensor",
)


def _add_autograd_saved_tensor(
    tensor: torch.Tensor,
    seen_data_ptrs: set[int],
) -> tuple[int, int]:
    """Return byte/count contribution for a deduped autograd-saved tensor.

    Parameters
    ----------
    tensor
        Saved tensor to measure.
    seen_data_ptrs
        Data pointers already counted for this operation.

    Returns
    -------
    tuple of int
        ``(bytes, tensor_count)`` contribution for this tensor.
    """
    if is_functorch_wrapped_tensor(tensor):
        return 0, 0
    try:
        # TorchLens-internal autograd-saved-tensor dedup read: mark it as an internal read so
        # the r14-H3 storage-bridge host-write watch (which patches ``Tensor.data_ptr``) does not
        # mistake this bookkeeping pointer read for a user storage-alias exposure. W8C: also
        # pause logging -- ``data_ptr`` is a wrapped tensor method and an unpaused read pays the
        # full per-op dispatch while emitting no op. The bypassed wrapped call still consumes
        # its ``next_func_call_id()`` so the id sequence stamped on real ops is unchanged.
        logging_enabled = _st._logging_enabled
        with pause_logging(), internal_scalar_read():
            if logging_enabled:
                _st.next_func_call_id()
            data_ptr = tensor.data_ptr()
    except Exception:
        return 0, 0
    if data_ptr in seen_data_ptrs:
        return 0, 0
    seen_data_ptrs.add(data_ptr)
    with pause_logging():
        return tensor.numel() * tensor.element_size(), 1


def _get_autograd_saved_stats_by_output(
    output: Any,
) -> dict[int, tuple[int | None, int | None]]:
    """Measure autograd-saved tensor bytes/counts for each tensor output.

    Parameters
    ----------
    output
        Raw function output from a decorated torch operation.

    Returns
    -------
    dict
        Mapping from output index to ``(autograd_memory,
        num_autograd_tensors)``. Non-tensor outputs and tensors without
        ``grad_fn_handle`` are omitted and handled by callers as ``None`` values.
    """
    stats_by_index: dict[int, tuple[int | None, int | None]] = {}
    seen_grad_fns: set[int] = set()
    seen_data_ptrs: set[int] = set()

    for output_index, maybe_tensor in enumerate(ensure_iterable(output)):
        # TorchLens bookkeeping ``grad_fn`` read (same r65 receiver aliasing
        # note as _partition_output_entries_with_autograd_stats).
        with internal_scalar_read():
            grad_fn_handle = (
                maybe_tensor.grad_fn if isinstance(maybe_tensor, torch.Tensor) else None
            )
        if grad_fn_handle is None:
            continue
        grad_fn_object_id = id(grad_fn_handle)
        if grad_fn_object_id in seen_grad_fns:
            stats_by_index[output_index] = (0, 0)
            continue
        seen_grad_fns.add(grad_fn_object_id)

        total_bytes = 0
        tensor_count = 0
        for saved_value in _iter_autograd_saved_candidates(grad_fn_handle):
            for saved_tensor in _collect_tensor_values(saved_value):
                bytes_added, count_added = _add_autograd_saved_tensor(saved_tensor, seen_data_ptrs)
                total_bytes += bytes_added
                tensor_count += count_added
        stats_by_index[output_index] = (total_bytes, tensor_count)

    return stats_by_index


def _partition_output_entries_with_autograd_stats(output: Any) -> list[_OutputTensorEntry]:
    """Collect output entries and autograd stats in one pass.

    Parameters
    ----------
    output
        Raw function output from a decorated torch operation.

    Returns
    -------
    list[_OutputTensorEntry]
        Output entries in logging order, each paired with its autograd saved
        tensor byte/count stats when available.
    """

    raw_entries: list[tuple[Any, tuple[OutputPathComponent, ...], ContainerSpec | None]] = list(
        _walk_output_tensors_with_paths(output)
    )
    if not raw_entries:
        raw_entries = [(out, (), None) for out in ensure_iterable(output)]

    partitioned_entries: list[_OutputTensorEntry] = []
    seen_grad_fns: set[int] = set()
    seen_data_ptrs: set[int] = set()
    for maybe_tensor, container_path, container_spec in raw_entries:
        autograd_stats: tuple[int | None, int | None] = (None, None)
        # TorchLens's own bookkeeping read: an in-place op's output IS its
        # receiver, so an unmarked ``grad_fn`` read on a registered buffer
        # (BN ``num_batches_tracked.add_(1)``) would record a phantom
        # declared-state fact (r65 unread-bit contract).
        with internal_scalar_read():
            grad_fn_handle = (
                maybe_tensor.grad_fn if isinstance(maybe_tensor, torch.Tensor) else None
            )
        if grad_fn_handle is not None:
            grad_fn_object_id = id(grad_fn_handle)
            if grad_fn_object_id in seen_grad_fns:
                autograd_stats = (0, 0)
            else:
                seen_grad_fns.add(grad_fn_object_id)
                total_bytes = 0
                tensor_count = 0
                for saved_value in _iter_autograd_saved_candidates(grad_fn_handle):
                    for saved_tensor in _collect_tensor_values(saved_value):
                        bytes_added, count_added = _add_autograd_saved_tensor(
                            saved_tensor, seen_data_ptrs
                        )
                        total_bytes += bytes_added
                        tensor_count += count_added
                autograd_stats = (total_bytes, tensor_count)
        partitioned_entries.append(
            _OutputTensorEntry(
                value=maybe_tensor,
                container_path=container_path,
                container_spec=container_spec,
                autograd_stats=autograd_stats,
            )
        )

    return partitioned_entries


def _register_call_output_container_snapshot(
    trace: "Trace",
    output: Any,
    *,
    output_entries: list[_OutputTensorEntry],
    func_call_id: int,
    event_index: int,
) -> None:
    """Register a function-call output container snapshot when opt-in capture is active.

    Parameters
    ----------
    trace
        Active trace.
    output
        Raw function output object.
    output_entries
        Path-aware tensor output entries from the existing output walker.
    func_call_id
        Function call identifier for the output boundary.
    event_index
        Capture event index used for ordering.
    """

    if not (
        getattr(trace, "intervention_ready", False)
        or getattr(trace, "_capture_container_structure", False)
    ):
        return
    spec = next((entry.container_spec for entry in output_entries if entry.container_spec), None)
    if spec is None:
        return
    registry = trace._wrapper_runtime_ws.container_registry
    registry.register_snapshot(
        output,
        site=FuncSite(func_call_id=func_call_id, position="return"),
        role=Role.CALL_OUTPUT,
        phase=Phase.POST_CALL,
        observed_at_event_index=event_index,
        spec=spec,
        leaf_occurrences=_container_leaf_occurrences_from_entries(output_entries),
        reconstructable=True,
    )


def register_call_input_container_snapshots(
    trace: "Trace",
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    func_call_id: int,
    event_index: int,
) -> None:
    """Register tensor-bearing function-call input containers.

    Parameters
    ----------
    trace:
        Active trace.
    args:
        Function positional arguments at call entry.
    kwargs:
        Function keyword arguments at call entry.
    func_call_id:
        Function-call identifier for this boundary.
    event_index:
        Monotonic ordering value for the pre-call snapshot.
    """

    if not getattr(trace, "_capture_container_structure", False):
        return
    registry = trace._wrapper_runtime_ws.container_registry
    for index, arg in enumerate(args):
        result = walk_container(arg, role=Role.CALL_INPUT, capability="full_spec")
        if result is None:
            continue
        registry.register_snapshot(
            arg,
            site=FuncSite(func_call_id=func_call_id, position=("arg", index)),
            role=Role.CALL_INPUT,
            phase=Phase.PRE_CALL,
            observed_at_event_index=event_index,
            spec=result.spec,
            leaf_occurrences=result.leaf_occurrences,
            reconstructable=result.reconstructable,
        )
    for key, value in kwargs.items():
        result = walk_container(value, role=Role.CALL_INPUT, capability="full_spec")
        if result is None:
            continue
        registry.register_snapshot(
            value,
            site=FuncSite(func_call_id=func_call_id, position=("kwarg", key)),
            role=Role.CALL_INPUT,
            phase=Phase.PRE_CALL,
            observed_at_event_index=event_index,
            spec=result.spec,
            leaf_occurrences=result.leaf_occurrences,
            reconstructable=result.reconstructable,
        )


def _container_leaf_occurrences_from_entries(
    output_entries: list[_OutputTensorEntry],
) -> tuple[ContainerLeafOccurrence, ...]:
    """Build ordered leaf occurrences from path-aware output entries.

    Parameters
    ----------
    output_entries
        Path-aware tensor output entries.

    Returns
    -------
    tuple[ContainerLeafOccurrence, ...]
        Occurrence records preserving repeated tensors at multiple paths.
    """

    occurrences: list[ContainerLeafOccurrence] = []
    for occ_index, entry in enumerate(output_entries):
        producer_label = (
            get_tensor_label(entry.value) if isinstance(entry.value, torch.Tensor) else None
        )
        occurrences.append(
            ContainerLeafOccurrence(
                path=entry.container_path,
                producer_op_label=producer_label,
                tensor_identity=producer_label,
                occ_index=occ_index,
            )
        )
    return tuple(occurrences)


def _get_autograd_saved_stats_for_tensor(
    tensor: torch.Tensor,
) -> tuple[int | None, int | None]:
    """Measure autograd-saved tensor bytes/counts for a single tensor output.

    Parameters
    ----------
    tensor
        Tensor output from a decorated torch operation.

    Returns
    -------
    tuple
        ``(autograd_memory, num_autograd_tensors)``. Both values
        are ``None`` when no grad_fn_handle exists.
    """
    if tensor.grad_fn is None:
        return None, None
    stats_by_index = _get_autograd_saved_stats_by_output(tensor)
    return stats_by_index.get(0, (None, None))
