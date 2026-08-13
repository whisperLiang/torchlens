"""Backend-neutral edges and payload metadata invariants."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_classes.trace import Trace
    from .invariants import (
        MetadataInvariantError,
        _dtype_values_match,
        _resolve_trace_label,
        op_has_genuine_replacement_evidence,
    )

__all__ = (
    "_check_backend_neutral_graph_topology",
    "_check_edge_use_parent_arg_invariants",
    "_check_op_log_fields",
    "_check_payload_metadata_invariants",
    "_live_payload_value",
    "_check_live_payload_metadata",
    "_payload_shape",
    "_payload_dtype",
    "_payload_memory",
)


def _check_backend_neutral_graph_topology(ml: Trace) -> None:
    """Check parent/child symmetry for non-torch traces where fields exist.

    Parameters
    ----------
    ml:
        Postprocessed non-torch trace to validate.

    Raises
    ------
    MetadataInvariantError
        If a populated parent or child list references a missing layer or lacks
        the reciprocal edge.
    """

    name = "backend_neutral_graph_topology"
    labels = {
        getattr(layer, "label", getattr(layer, "layer_label", ""))
        for layer in getattr(ml, "layer_list", ())
    } | {
        getattr(layer, "layer_label", getattr(layer, "label", ""))
        for layer in getattr(ml, "layer_list", ())
    }
    for layer in getattr(ml, "layer_list", ()):
        label = getattr(layer, "layer_label", getattr(layer, "label", type(layer).__name__))
        parents = list(getattr(layer, "parents", ()) or ())
        children = list(getattr(layer, "children", ()) or ())
        for parent_label in parents:
            if parent_label not in labels:
                raise MetadataInvariantError(
                    name,
                    f"Layer {label} has parent {parent_label!r} outside trace labels",
                )
            parent = ml[parent_label]
            parent_children = set(getattr(parent, "children", ()) or ())
            if (
                label not in parent_children
                and getattr(layer, "label", label) not in parent_children
            ):
                raise MetadataInvariantError(
                    name,
                    f"Layer {label} lists {parent_label!r} as parent, but reciprocal child is missing",
                )
        for child_label in children:
            if child_label not in labels:
                raise MetadataInvariantError(
                    name,
                    f"Layer {label} has child {child_label!r} outside trace labels",
                )
            child = ml[child_label]
            child_parents = set(getattr(child, "parents", ()) or ())
            if label not in child_parents and getattr(layer, "label", label) not in child_parents:
                raise MetadataInvariantError(
                    name,
                    f"Layer {label} lists {child_label!r} as child, but reciprocal parent is missing",
                )


def _check_edge_use_parent_arg_invariants(ml: Trace) -> None:
    """Check existing edge-use records and parent-arg references.

    Precondition contract: edge-use metadata is optional on torch graph edges.
    The torch eager builder emits ``_edge_uses`` only for args/kwargs-derived
    parent entries. Buffer-source, output, control, module, and
    intervention-injected edges may legitimately have no edge-use record. When
    an ``_edge_uses`` record exists, its kind must be in ``EdgeUseKind`` and
    its parent/child labels must resolve. When a ``parent_arg_positions`` entry
    exists, its referenced parent label must resolve. This invariant never
    asserts that every parent edge has a corresponding edge-use record.

    Parameters
    ----------
    ml:
        Postprocessed torch trace to validate.

    Raises
    ------
    MetadataInvariantError
        If populated edge-use or parent-arg-position metadata is malformed or
        references labels that do not resolve.
    """

    name = "edge_use_parent_arg_consistency"
    valid_edge_uses = {"arg", "kwarg", "container", "module", "buffer", "output", "control"}
    valid_arg_kinds = {"positional", "keyword"}
    for layer in ml.layer_list:
        layer_label = getattr(layer, "layer_label", type(layer).__name__)
        for record in getattr(layer, "_edge_uses", ()) or ():
            edge_use = getattr(record, "edge_use", None)
            if edge_use not in valid_edge_uses:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{layer_label}' has invalid edge_use kind {edge_use!r}",
                )
            arg_kind = getattr(record, "arg_kind", None)
            if arg_kind not in valid_arg_kinds:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{layer_label}' has invalid edge arg_kind {arg_kind!r}",
                )
            parent_label = getattr(record, "parent_label", None)
            if not isinstance(parent_label, str) or _resolve_trace_label(ml, parent_label) is None:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{layer_label}' has edge-use record with unresolved parent "
                    f"{parent_label!r}",
                )
            child_label = getattr(record, "child_label", None)
            if not isinstance(child_label, str) or _resolve_trace_label(ml, child_label) is None:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{layer_label}' has edge-use record with unresolved child "
                    f"{child_label!r}",
                )

        parent_arg_positions = getattr(layer, "parent_arg_positions", {}) or {}
        if not isinstance(parent_arg_positions, Mapping):
            raise MetadataInvariantError(
                name,
                f"Layer '{layer_label}' has non-mapping parent_arg_positions",
            )
        for arg_domain in ("args", "kwargs"):
            entries = parent_arg_positions.get(arg_domain, {}) or {}
            if not isinstance(entries, Mapping):
                raise MetadataInvariantError(
                    name,
                    f"Layer '{layer_label}' parent_arg_positions[{arg_domain!r}] is not a mapping",
                )
            for position, parent_label in entries.items():
                if not isinstance(parent_label, str):
                    raise MetadataInvariantError(
                        name,
                        f"Layer '{layer_label}' parent_arg_positions[{arg_domain!r}]"
                        f"[{position!r}] is not a label string",
                    )
                if _resolve_trace_label(ml, parent_label) is None:
                    raise MetadataInvariantError(
                        name,
                        f"Layer '{layer_label}' parent_arg_positions[{arg_domain!r}]"
                        f"[{position!r}] references missing parent {parent_label!r}",
                    )


def _check_op_log_fields(ml: Trace) -> None:
    """Check D: per-layer field consistency (shape, dtype, pass numbering, func, nesting).

    Validates:
    - Saved tensor shape/dtype match actual out (when saved).
    - Pass numbering: pass_index >= 1, num_passes >= pass_index.
    - Computational layers have callable func and non-empty func_name.
    - step_index >= 1 for non-input/non-buffer layers.
    - module_call_depth matches len(modules).
    - Label format: pass-qualified label has ':' iff multi-pass; no-pass label never has ':'.
    """
    name = "op_log_fields"

    for lpl in ml.layer_list:
        label = lpl.layer_label

        # Tensor shape/dtype consistency when outs are saved
        if lpl.has_saved_activation and lpl.out is not None:
            actual_shape = tuple(lpl.out.shape)
            if lpl.shape != actual_shape:
                raise MetadataInvariantError(
                    name,
                    f"Layer {label}: shape={lpl.shape} != actual shape={actual_shape}",
                )
            if lpl.dtype != lpl.out.dtype:
                raise MetadataInvariantError(
                    name,
                    f"Layer {label}: dtype={lpl.dtype} != actual dtype={lpl.out.dtype}",
                )

        # Pass numbering
        if lpl.pass_index < 1:
            raise MetadataInvariantError(name, f"Layer {label}: pass_index={lpl.pass_index} < 1")
        if lpl.num_passes < lpl.pass_index:
            raise MetadataInvariantError(
                name,
                f"Layer {label}: num_passes={lpl.num_passes} < pass_index={lpl.pass_index}",
            )

        # A GENUINE raw-forward-hook output replacement is legitimately
        # functionless: the user substituted an opaque tensor for a module's
        # output, so there is no torch function to validate. This exemption is
        # deliberately narrow -- it must NOT cover auto-synthesized placeholders
        # during plain capture (a previous band-aid widened it to silence the
        # vmap-built attention mask, disarming this tripwire). Round-26 W3-2
        # hardening: the per-op attributes below are written by the placeholder
        # synthesizer itself, so the exemption additionally requires the
        # trace-level replacement-event ledger to corroborate the claim; a
        # placeholder stamped during PLAIN capture (no recorded replacement
        # event) now fails this invariant, per the 2026-06-02 lesson.
        is_functionless_replacement = (
            lpl.func_name == "intervention_replacement"
            and getattr(lpl, "intervention_replaced", False)
            and not getattr(lpl, "is_internal_source", False)
            and op_has_genuine_replacement_evidence(lpl, ml)
        )

        # An internally generated *source* tensor whose construction TorchLens
        # could not trace (e.g. an attention mask built inside torch.vmap) is a
        # genuine functionless graph source, exactly like a buffer: func is None
        # and func_name is "none". Traced ops that merely have an internal-source
        # ancestor still carry a real callable func and are NOT exempted here.
        is_functionless_internal_source = (
            getattr(lpl, "is_internal_source", False) and lpl.func is None
        )

        # Function applied (non-input, non-buffer, non-output, non-source,
        # non-hook-replacement layers).
        if not (
            lpl.is_input
            or lpl.is_buffer
            or lpl.is_output
            or is_functionless_internal_source
            or is_functionless_replacement
        ):
            if not callable(lpl.func):
                raise MetadataInvariantError(name, f"Layer {label}: func is not callable")
            if not lpl.func_name:
                raise MetadataInvariantError(name, f"Layer {label}: func_name is empty")

        # Operation numbering (input/buffer/output bookkeeping layers have step_index=0)
        if not (lpl.is_input or lpl.is_buffer or lpl.is_output):
            if lpl.step_index is not None and lpl.step_index < 1:
                raise MetadataInvariantError(
                    name, f"Layer {label}: step_index={lpl.step_index} < 1"
                )
        if lpl.raw_index < 1:
            raise MetadataInvariantError(
                name,
                f"Layer {label}: raw_index={lpl.raw_index} < 1",
            )

        # Module nesting depth
        if lpl.module_call_depth != len(lpl.modules):
            raise MetadataInvariantError(
                name,
                f"Layer {label}: module_call_depth={lpl.module_call_depth} != "
                f"len(modules)="
                f"{len(lpl.modules)}",
            )

        # Label format: pass-qualified label has ":" iff multi-pass
        if lpl.num_passes > 1 and ":" not in lpl.label:
            raise MetadataInvariantError(
                name,
                f"Layer {label}: multi-pass but label='{lpl.label}' has no ':'",
            )
        if ":" in lpl.layer_label:
            raise MetadataInvariantError(
                name,
                f"Layer {label}: layer_label='{lpl.layer_label}' contains ':'",
            )


def _check_payload_metadata_invariants(ml: Trace) -> None:
    """Check saved and transformed live payload metadata.

    Precondition contract: tensor payload fields may be legitimately absent
    because of selective save, loaded traces, detached/audit-only metadata,
    disk-only storage, streaming finalization, or gradient eviction. This check
    compares shape, dtype, and memory only when a live payload object is
    present. Presence of a live raw or transformed activation requires
    ``has_saved_activation=True``; presence of a live raw or transformed
    gradient requires ``has_grad=True``. Missing payloads never imply
    corruption by themselves.

    Parameters
    ----------
    ml:
        Postprocessed torch trace to validate.

    Raises
    ------
    MetadataInvariantError
        If a present payload disagrees with its recorded metadata.
    """

    name = "payload_metadata_invariants"
    for op in ml.layer_list:
        label = getattr(op, "label", getattr(op, "layer_label", type(op).__name__))
        _check_live_payload_metadata(
            name,
            label,
            payload=_live_payload_value(op, "out"),
            shape=getattr(op, "shape", None),
            dtype=getattr(op, "dtype", None),
            memory=getattr(op, "activation_memory", None),
            presence_flag=getattr(op, "has_saved_activation", False),
            presence_flag_name="has_saved_activation",
            payload_name="out",
        )
        _check_live_payload_metadata(
            name,
            label,
            payload=_live_payload_value(op, "transformed_out"),
            shape=getattr(op, "transformed_out_shape", None),
            dtype=getattr(op, "transformed_out_dtype", None),
            memory=getattr(op, "transformed_activation_memory", None),
            presence_flag=getattr(op, "has_saved_activation", False),
            presence_flag_name="has_saved_activation",
            payload_name="transformed_out",
        )
        _check_live_payload_metadata(
            name,
            label,
            payload=_live_payload_value(op, "grad"),
            shape=getattr(op, "grad_shape", None),
            dtype=getattr(op, "grad_dtype", None),
            memory=getattr(op, "gradient_memory", None),
            presence_flag=getattr(op, "has_grad", False),
            presence_flag_name="has_grad",
            payload_name="grad",
        )
        _check_live_payload_metadata(
            name,
            label,
            payload=_live_payload_value(op, "transformed_grad"),
            shape=getattr(op, "transformed_grad_shape", None),
            dtype=getattr(op, "transformed_grad_dtype", None),
            memory=getattr(op, "transformed_gradient_memory", None),
            presence_flag=getattr(op, "has_grad", False),
            presence_flag_name="has_grad",
            payload_name="transformed_grad",
        )
        for record in getattr(op, "_grad_records", ()) or ():
            record_label = f"{label}.grad_record[{getattr(record, 'backward_pass_index', '?')}]"
            _check_live_payload_metadata(
                name,
                record_label,
                payload=_live_payload_value(record, "grad"),
                shape=getattr(record, "shape", None),
                dtype=getattr(record, "dtype", None),
                memory=getattr(record, "memory", None),
                presence_flag=getattr(record, "is_saved", False),
                presence_flag_name="is_saved",
                payload_name="grad",
            )
            _check_live_payload_metadata(
                name,
                record_label,
                payload=_live_payload_value(record, "transformed_grad"),
                shape=getattr(record, "transformed_grad_shape", None),
                dtype=getattr(record, "transformed_grad_dtype", None),
                memory=getattr(record, "transformed_gradient_memory", None),
                presence_flag=getattr(record, "is_saved", False),
                presence_flag_name="is_saved",
                payload_name="transformed_grad",
            )


def _live_payload_value(owner: object, payload_name: str) -> object | None:
    """Return an already-live payload without invoking guarded payload accessors.

    Parameters
    ----------
    owner:
        Object that owns the payload field.
    payload_name:
        Name of the payload field to inspect.

    Returns
    -------
    object or None
        The live payload object, or ``None`` when no payload is currently attached.
    """

    slot_getter = getattr(owner, "_slot", None)
    if callable(slot_getter):
        return slot_getter(payload_name, None)
    return getattr(owner, payload_name, None)


def _check_live_payload_metadata(
    name: str,
    label: str,
    *,
    payload: object | None,
    shape: object,
    dtype: object,
    memory: object,
    presence_flag: bool,
    presence_flag_name: str,
    payload_name: str,
) -> None:
    """Check metadata for one live tensor-like payload.

    Parameters
    ----------
    name:
        Invariant name used in raised errors.
    label:
        Owner label for diagnostics.
    payload:
        Live payload object, or ``None`` when absent.
    shape:
        Recorded shape metadata.
    dtype:
        Recorded dtype metadata.
    memory:
        Recorded memory metadata.
    presence_flag:
        Boolean metadata that should be true when payload is present.
    presence_flag_name:
        Name of ``presence_flag`` for diagnostics.
    payload_name:
        Payload field name for diagnostics.

    Raises
    ------
    MetadataInvariantError
        If present payload metadata disagrees with the payload.
    """

    if payload is None:
        return
    if not presence_flag:
        raise MetadataInvariantError(
            name,
            f"{label} has live {payload_name} payload but {presence_flag_name}=False",
        )
    actual_shape = _payload_shape(payload)
    if actual_shape is not None and shape != actual_shape:
        raise MetadataInvariantError(
            name,
            f"{label} {payload_name} shape metadata {shape!r} != payload shape {actual_shape!r}",
        )
    actual_dtype = _payload_dtype(payload)
    if (
        actual_dtype is not None
        and dtype is not None
        and not _dtype_values_match(dtype, actual_dtype)
    ):
        raise MetadataInvariantError(
            name,
            f"{label} {payload_name} dtype metadata {dtype!r} != payload dtype {actual_dtype!r}",
        )
    actual_memory = _payload_memory(payload)
    if actual_memory is not None and memory is not None:
        if not isinstance(memory, int):
            raise MetadataInvariantError(
                name,
                f"{label} {payload_name} memory metadata {memory!r} is not an integer",
            )
        if memory != actual_memory:
            raise MetadataInvariantError(
                name,
                f"{label} {payload_name} memory metadata {memory!r} != payload memory "
                f"{actual_memory!r}",
            )


def _payload_shape(payload: object) -> tuple[int, ...] | None:
    """Return a tuple shape for a tensor-like payload.

    Parameters
    ----------
    payload:
        Candidate tensor-like payload.

    Returns
    -------
    tuple[int, ...] | None
        Shape tuple when available.
    """

    shape = getattr(payload, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except TypeError:
        return None


def _payload_dtype(payload: object) -> object | None:
    """Return dtype metadata from a tensor-like payload.

    Parameters
    ----------
    payload:
        Candidate tensor-like payload.

    Returns
    -------
    object | None
        Payload dtype when available.
    """

    return getattr(payload, "dtype", None)


def _payload_memory(payload: object) -> int | None:
    """Return byte memory for a tensor-like payload.

    Parameters
    ----------
    payload:
        Candidate tensor-like payload.

    Returns
    -------
    int | None
        Number of bytes when ``nelement`` and ``element_size`` are available.
    """

    nelement = getattr(payload, "nelement", None)
    element_size = getattr(payload, "element_size", None)
    if not callable(nelement) or not callable(element_size):
        return None
    return int(nelement() * element_size())
